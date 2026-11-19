"""Plan and execute contact-based sorting in the tabletop simulation."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from io import StringIO

import mujoco as mj
import numpy as np

from .control import DOWNWARD, PandaController
from .scene import BOX_FLOOR_HEIGHT, BOX_RIM_HEIGHT, HOME, evaluate


def tune_gripper(model):
    gripper = model.actuator("actuator8").id
    model.actuator_gainprm[gripper, 0] *= 8
    model.actuator_biasprm[gripper, 1] *= 8
    fingers = [model.body(name).id for name in ("left_finger", "right_finger")]
    model.geom_friction[np.isin(model.geom_bodyid, fingers), 0] = 3.


class ContactCheck:
    def __init__(self, model, data, target):
        self.model, self.data = model, data
        self.target = model.body(target).id
        self.fingers = {model.body(name).id for name in ("left_finger", "right_finger")}
        base = model.body("link0").id
        robot = {base}
        for body in range(base + 1, model.nbody):
            if model.body_parentid[body] in robot:
                robot.add(body)
        self.robot = robot - {base}

    def check(self):
        for contact in self.data.contact:
            if contact.dist > -.0005:
                continue
            a, b = map(int, self.model.geom_bodyid[[contact.geom1, contact.geom2]])
            if b in self.robot and a not in self.robot:
                a, b = b, a
            if a in self.fingers and (b in self.fingers or (b == 0 and contact.dist > -.002)):
                continue
            if a in self.robot and not (a in self.fingers and b == self.target):
                raise RuntimeError(f"Collision: {self.model.body(a).name} / {self.model.body(b).name or 'table'}")


def _roll(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, s], [0., -s, c]])


def _yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def move_joint_path(arm, position, rotation, seconds, stage):
    """Use a joint-space path for free-space motion between task waypoints."""
    start = arm.data.qpos[arm.qadr].copy()
    try:
        end = arm.solve_ik(position, rotation, seed=start)
    except RuntimeError:
        end = arm.solve_ik(position, rotation, seed=HOME[:7])
    _move_joints(arm, start, end, seconds)
    error = np.linalg.norm(arm.tcp_position() - position)
    if error > .009:
        raise RuntimeError(f"{stage}: TCP tracking error {error:.4f} m")
    print(f"[{stage}] TCP={arm.tcp_position().round(4)}, error={error * 1000:.2f} mm", flush=True)


def _move_joints(arm, start, end, seconds):
    steps = max(1, round(.02 / arm.model.opt.timestep))
    count = round(seconds / (steps * arm.model.opt.timestep))
    for t in np.linspace(0, 1, count + 1)[1:]:
        blend = t**3 * (10 + t * (-15 + 6 * t))
        arm.data.ctrl[arm.actuators] = start + blend * (end - start)
        arm.step(steps)
    arm.hold(.25)


def park_arm(arm):
    _move_joints(arm, arm.data.qpos[arm.qadr].copy(), HOME[:7], 1.7)


def operate_lid(arm, category, opening):
    model, data = arm.model, arm.data
    name = f"storage_box_lid_{category}"
    joint = model.joint(name)
    angle_index = joint.qposadr[0]
    anchor = data.xanchor[joint.id].copy()
    handle = "handle" if opening else "closing_handle"
    center = data.geom(f"{name}_{handle}").xpos.copy()
    offset = center - anchor
    start = float(data.qpos[angle_index])
    finish = np.deg2rad(52 if opening else 29)

    def orientation(angle):
        if opening:
            return DOWNWARD
        pitch = np.clip(np.rad2deg(angle) - 84, -70, -12)
        return _roll(np.deg2rad(pitch)) @ _yaw(np.pi / 2) @ DOWNWARD

    rotation = orientation(start)
    clearance = -(.043 if opening else .055) * rotation[:, 2]
    arm.move(np.array([.42, .10, .30]), 1.0, f"{category} lid clearance")
    move_joint_path(arm, center + clearance, rotation, 1.4, f"{category} lid approach")
    arm.move(center, .8, f"{category} lid handle", rotation=rotation)
    arm.close()
    if len(arm.finger_contacts(name)) != 2:
        raise RuntimeError(f"The gripper missed the {category} lid handle")

    for angle in np.linspace(start, finish, 13)[1:]:
        rotation = orientation(angle)
        arm.move(anchor + _roll(angle - start) @ offset, .24, f"{category} lid turn",
                 tolerance=.020, rotation=rotation)
        if abs(data.qpos[angle_index] - angle) > np.deg2rad(14):
            raise RuntimeError(f"The {category} lid slipped out of the gripper")

    arm.open()
    retreat = np.array([0., -.12, .10]) if opening else -.105 * rotation[:, 2]
    arm.move(arm.tcp_position() + retreat, .9, f"{category} lid retreat", rotation=rotation)
    arm.move(np.array([.42, .10, .30]), 1.0, f"{category} lid clearance")
    arm.hold(1.8)
    angle = float(data.qpos[angle_index])
    if abs(angle - joint.range[int(opening)]) > np.deg2rad(3):
        raise RuntimeError(f"The {category} lid stopped at {np.rad2deg(angle):.1f}°")
    return float(np.rad2deg(angle))


@dataclass(frozen=True)
class Transfer:
    candidate: int
    score: float
    pick: np.ndarray
    rotation: np.ndarray
    release: np.ndarray
    release_rotation: np.ndarray
    height: float


def _transfers(poses, scores, points, box, lane, height):
    points_xy = points[:, :2]
    low, high = np.percentile(points_xy, [1, 99], axis=0)
    center = (low + high) / 2
    eigenvalues, axes = np.linalg.eigh(np.cov(points_xy, rowvar=False))
    angle = np.arctan2(axes[1, -1], axes[0, -1])
    turn = np.pi / 2 - (angle + np.pi / 2) % np.pi if eigenvalues[-1] > 1.25 * eigenvalues[0] else 0.
    slot = np.asarray(box["center_xy_m"]) + [0., lane * .047]
    order = np.argsort(-scores * np.clip(-poses[:, 2, 2], .05, 1.) ** .65)
    choices = []
    for index in order:
        pose = poses[index]
        position = pose[:3, 3]
        if pose[2, 2] > -.56 or not .005 <= position[2] <= .13:
            continue
        if np.any(position[:2] < low - .022) or np.any(position[:2] > high + .022):
            continue
        choices.append(index)
    for offset, vertical, flip, other_yaw in ((-.004, False, False, False), (0., True, False, False),
                                              (-.002, True, True, False), (0., False, False, True),
                                              (-.004, True, False, True), (-.007, False, True, False),
                                              (.004, True, True, True)):
        align = _yaw(turn + np.pi * other_yaw)
        for index in choices[:26]:
            pose = poses[index]
            pick = pose[:3, 3] + [0., 0., offset]
            if pick[2] < .008:
                continue
            rotation = pose[:3, :3]
            if vertical:
                closing = rotation[:2, 1]
                rotation = _yaw(np.arctan2(closing[0], -closing[1])) @ DOWNWARD
            if flip:
                rotation = rotation @ np.diag([-1., -1., 1.])
            release = np.r_[slot + align[:2, :2] @ (pick[:2] - center),
                            pick[2] + BOX_FLOOR_HEIGHT + .018]
            yield Transfer(int(index), float(scores[index]), pick, rotation, release, align @ rotation, height)


def _pick_object(arm, name, transfer):
    pick, rotation = transfer.pick, transfer.rotation
    above = pick - .075 * rotation[:, 2]
    source = arm.data.body(name)
    start = source.xpos.copy()
    move_joint_path(arm, above, rotation, 1.6, f"{name} pregrasp")
    arm.move(pick, .9, f"{name} approach", rotation=rotation)
    if np.linalg.norm(source.xpos - start) > .015:
        raise RuntimeError("Target moved during approach")
    arm.close()
    if len(arm.finger_contacts(name)) != 2:
        raise RuntimeError("Both fingers must contact the object")

    lifted = np.r_[pick[:2], min(.21, transfer.height)]
    arm.move(lifted, 1.3, f"{name} lift", rotation=rotation)
    if len(arm.finger_contacts(name)) != 2 or source.xpos[2] - start[2] < .055:
        raise RuntimeError("Object slipped during lift")


def transfer_object(arm, name, transfer, settle_seconds):
    _pick_object(arm, name, transfer)
    _place_object(arm, name, transfer, settle_seconds)


def _place_object(arm, name, transfer, settle_seconds):
    release, release_rotation = transfer.release, transfer.release_rotation
    rotation = transfer.rotation
    carrying_height = transfer.height
    transit = np.r_[.46, -.03, carrying_height]
    move_joint_path(arm, transit, rotation, 1.35, f"{name} transit")
    move_joint_path(arm, transit, release_rotation, .6, f"{name} align")
    over_box = np.r_[release[:2], min(carrying_height, .235)]
    move_joint_path(arm, over_box, release_rotation, 1.3, f"{name} over box")
    arm.move(release, 1.15, f"{name} lower", rotation=release_rotation)
    if len(arm.finger_contacts(name)) != 2:
        raise RuntimeError("Object slipped before release")
    arm.open()
    arm.move(over_box, .95, f"{name} retreat", rotation=release_rotation)
    park_arm(arm)
    arm.hold(settle_seconds)


def plan_transfer(arm, config, obj, lane, poses, scores, points, idle: Callable[[], None]):
    model = arm.model
    scratch = mj.MjData(model)
    held = mj.MjData(model)
    name = obj["name"]
    contact = ContactCheck(model, scratch, name)
    box = config["trays"][obj["category"]]
    settle = config["evaluation"]["settle_seconds"]
    already_sorted = {item["name"] for item in evaluate(model, arm.data, config)["objects"] if item["success"]}
    errors: dict[str, int] = {}
    for proposal in _transfers(poses, scores, points, box, lane, config["execution"]["transport_clearance_m"]):
        idle()
        trial = PandaController(model, scratch, on_step=contact.check)
        mj.mj_copyData(scratch, model, arm.data)
        try:
            with redirect_stdout(StringIO()):
                _pick_object(trial, name, proposal)
        except RuntimeError as error:
            message = str(error).split(" at [", 1)[0]
            errors[message] = errors.get(message, 0) + 1
            continue

        mj.mj_copyData(held, model, scratch)
        # Try the low release first. Higher/front-biased releases leave more
        # clearance for fingertips and the open lid above the rear slot.
        for dx, dy, height in ((0., 0., proposal.release[2]), (0., -.020, BOX_RIM_HEIGHT + .025),
                               (0., 0., BOX_RIM_HEIGHT + .022), (0., 0., BOX_RIM_HEIGHT + .047),
                               (-.020, -.012, BOX_RIM_HEIGHT + .025), (0., -.025, BOX_RIM_HEIGHT + .060),
                               (-.020, -.012, BOX_RIM_HEIGHT + .055),
                               (.015, -.025, BOX_RIM_HEIGHT + .060)):
            idle()
            release = proposal.release.copy()
            release[:2] += [dx, dy]
            release[2] = max(height, proposal.release[2])
            placement = replace(proposal, release=release)
            mj.mj_copyData(scratch, model, held)
            try:
                with redirect_stdout(StringIO()):
                    _place_object(trial, name, placement, settle)
                sorted_names = {item["name"] for item in evaluate(model, scratch, config)["objects"] if item["success"]}
                if already_sorted | {name} <= sorted_names:
                    return placement, errors
                message = "Object did not settle completely inside its box"
            except RuntimeError as error:
                message = str(error).split(" at [", 1)[0]
            errors[message] = errors.get(message, 0) + 1
    return None, errors
