"""Jacobian IK and smooth position control; the object moves only by contact."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np


# 绕基座X轴转180°，使工具坐标系的正Z轴指向桌面。
DOWNWARD = np.diag([1.0, -1.0, -1.0])


@dataclass
class MotionResult:
    stage: str
    target: list[float]
    position_error_m: float


class PandaController:
    """Targets are expressed in link0 coordinates, in metres."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 on_step: Callable[[], None] | None = None):
        self.model, self.data, self.on_step = model, data, on_step
        self.joints = np.array([model.joint(f"joint{i}").id for i in range(1, 8)])
        # 关节位置和速度在MuJoCo中使用不同的地址表；雅可比矩阵按速度维度取列。
        self.qadr = model.jnt_qposadr[self.joints]
        self.dadr = model.jnt_dofadr[self.joints]
        self.actuators = np.array([model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.gripper = model.actuator("actuator8").id
        self.tcp = model.site("tcp").id
        self.base = model.body("link0").id
        # 逆运动学在独立状态里试算，关节执行器仍跟踪真实仿真状态的目标。
        self.ik_data = mujoco.MjData(model)
        self.ik_data.qpos[:] = data.qpos
        self.limits = model.jnt_range[self.joints]
        self.results: list[MotionResult] = []
        self.data.ctrl[self.actuators] = data.qpos[self.qadr]
        self.data.ctrl[self.gripper] = 255.0

    def base_to_world(self, point: np.ndarray) -> np.ndarray:
        return self.data.xpos[self.base] + self.data.xmat[self.base].reshape(3, 3) @ point

    def tcp_position(self) -> np.ndarray:
        rotation = self.data.xmat[self.base].reshape(3, 3)
        return rotation.T @ (self.data.site_xpos[self.tcp] - self.data.xpos[self.base])

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            # 仅对机械臂关节补偿重力和科里奥利等偏置力，夹爪和物体继续由仿真接触驱动。
            self.data.qfrc_applied[self.dadr] = self.data.qfrc_bias[self.dadr]
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise RuntimeError("Non-finite simulation state")
            if self.on_step is not None:
                self.on_step()

    def hold(self, seconds: float) -> None:
        self.step(round(seconds / self.model.opt.timestep))

    def solve_ik(self, position: np.ndarray, rotation: np.ndarray = DOWNWARD,
                 seed: np.ndarray | None = None) -> np.ndarray:
        """Damped least squares with joint limits and full orientation error."""
        d, m = self.ik_data, self.model
        d.qpos[:] = self.data.qpos
        if seed is not None:
            d.qpos[self.qadr] = seed
        target_p = self.base_to_world(np.asarray(position, dtype=float))
        target_r = self.data.xmat[self.base].reshape(3, 3) @ rotation
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        quat, rotvec = np.empty(4), np.empty(3)
        for _ in range(350):
            mujoco.mj_forward(m, d)
            ep = target_p - d.site_xpos[self.tcp]
            # 将当前姿态到目标姿态的相对旋转转为短弧旋转向量，与位置误差一起求解。
            delta_r = target_r @ d.site_xmat[self.tcp].reshape(3, 3).T
            mujoco.mju_mat2Quat(quat, delta_r.ravel())
            if quat[0] < 0:
                quat *= -1
            mujoco.mju_quat2Vel(rotvec, quat, 1.0)
            if np.linalg.norm(ep) < 0.00015 and np.linalg.norm(rotvec) < 0.001:
                return d.qpos[self.qadr].copy()
            mujoco.mj_jacSite(m, d, jp, jr, self.tcp)
            # 降低姿态项权重，并用阻尼抑制接近奇异姿态时过大的关节更新。
            jac = np.vstack([jp[:, self.dadr], 0.4 * jr[:, self.dadr]])
            error = np.concatenate([ep, 0.4 * rotvec])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.002**2 * np.eye(6), error)
            dq *= min(1.0, 0.12 / max(np.max(np.abs(dq)), 1e-9))
            # 每轮限制最大步长，并为关节限位预留少量余量。
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + dq,
                                       self.limits[:, 0] + .002, self.limits[:, 1] - .002)
        raise RuntimeError(f"IK failed at {position}: position error {np.linalg.norm(ep):.4f} m, "
                           f"orientation error {np.linalg.norm(rotvec):.4f} rad")

    def move(self, target: np.ndarray, duration: float, stage: str,
             tolerance: float = .008, rotation: np.ndarray = DOWNWARD) -> MotionResult:
        target = np.asarray(target, dtype=float)
        start = self.tcp_position()
        base_r = self.data.xmat[self.base].reshape(3, 3)
        start_r = base_r.T @ self.data.site_xmat[self.tcp].reshape(3, 3)
        start_q, end_q = np.empty(4), np.empty(4)
        mujoco.mju_mat2Quat(start_q, start_r.ravel())
        mujoco.mju_mat2Quat(end_q, rotation.ravel())
        if start_q @ end_q < 0:
            end_q *= -1
        angle = np.arccos(np.clip(start_q @ end_q, -1., 1.))
        seed = self.data.qpos[self.qadr].copy()
        # 先确认终点可达，再按平滑时间曲线插值位置和四元数姿态。
        self.solve_ik(target, rotation, seed=seed)
        steps_per_update = max(1, round(.02 / self.model.opt.timestep))
        updates = max(1, round(duration / (steps_per_update * self.model.opt.timestep)))
        planned: list[np.ndarray] = []
        for i in range(1, updates + 1):
            t = i / updates
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            desired = start + blend * (target - start)
            if angle < 1e-5:
                quat = (1 - blend) * start_q + blend * end_q
            else:
                quat = (np.sin((1 - blend) * angle) * start_q + np.sin(blend * angle) * end_q) / np.sin(angle)
            quat /= np.linalg.norm(quat)
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, quat)
            # 用上一采样点的关节解作为初值，让相邻位姿沿同一解附近求解。
            seed = self.solve_ik(desired, rotation.reshape(3, 3), seed=seed)
            planned.append(seed)
        # 整段都有逆运动学解后才下发关节目标，避免中间点无解时机械臂停在半路。
        for seed in planned:
            self.data.ctrl[self.actuators] = seed
            self.step(steps_per_update)
        self.hold(.25)
        error = float(np.linalg.norm(target - self.tcp_position()))
        result = MotionResult(stage, target.tolist(), error)
        self.results.append(result)
        print(f"[{stage}] TCP={self.tcp_position().round(4)}, error={error*1000:.2f} mm", flush=True)
        if error > tolerance:
            raise RuntimeError(f"{stage}: TCP tracking error {error:.4f} m exceeds {tolerance:.4f} m")
        return result

    def open(self) -> None:
        # 夹爪执行器以255和0分别表示张开和闭合目标，留出仿真时间让双指到位。
        self.data.ctrl[self.gripper] = 255.0
        self.hold(.6)

    def close(self) -> None:
        self.data.ctrl[self.gripper] = 0.0
        self.hold(.8)

    def finger_contacts(self, object_body: str = "blue_cube") -> list[str]:
        """Contact diagnostics used to detect missed grasps, never to position them."""
        body = self.model.body(object_body).id
        fingers = {self.model.body(name).id: name for name in ["left_finger", "right_finger"]}
        touching = set()
        for contact in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if b1 == body and b2 in fingers:
                touching.add(fingers[b2])
            if b2 == body and b1 in fingers:
                touching.add(fingers[b1])
        return sorted(touching)


def pick_and_place(controller: PandaController, coor1: np.ndarray, coor2: np.ndarray,
                   cube_size: float = .04, clearance_z: float = .28,
                   on_stage: Callable[[str], None] | None = None) -> dict:
    """Pick from top-surface Coor1; place above interior-floor Coor2."""
    # 单立方体流程从两个表面坐标和边长估算抓取中心与释放高度。
    c = controller
    coor1, coor2 = np.asarray(coor1), np.asarray(coor2)
    if not all(np.isfinite(p).all() and .15 < p[0] < .75 and abs(p[1]) < .45
               and -.02 < p[2] < .16 for p in [coor1, coor2]):
        raise ValueError("Detected target is outside the configured tabletop workspace")
    grasp = coor1 - np.array([0., 0., cube_size / 2])
    release = coor2 + np.array([0., 0., cube_size / 2 + .035])
    above_cube = np.array([grasp[0], grasp[1], clearance_z])
    above_basket = np.array([release[0], release[1], clearance_z])
    # 靠近基座时将手掌向外倾斜，让下降路径可达，并保留与立方体对齐的闭合轴。
    pitch = np.deg2rad(45. if np.linalg.norm(grasp[:2]) < .30 else 0.)
    cp, sp = np.cos(pitch), np.sin(pitch)
    grasp_rotation = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]]) @ DOWNWARD

    def stage(name: str) -> None:
        if on_stage:
            on_stage(name)

    c.open()
    c.move(above_cube, 2.0, "approach", rotation=grasp_rotation)
    c.move(grasp, 2.0, "descend", rotation=grasp_rotation)
    c.close()
    contacts = c.finger_contacts()
    if len(contacts) != 2:
        raise RuntimeError(f"Grasp failed: expected both finger contacts, got {contacts}")
    stage("grasp")
    c.move(above_cube, 2.0, "lift", rotation=grasp_rotation)
    if len(c.finger_contacts()) != 2:
        raise RuntimeError("Cube slipped during lift")
    stage("lift")
    c.move(above_basket, 2.8, "transfer")
    c.move(release, 2.0, "lower")
    c.open()
    stage("release")
    c.move(above_basket, 1.8, "retreat")
    c.hold(1.5)
    stage("finished")
    return {"grasp_contacts": contacts, "grasp_tcp_base": grasp.tolist(),
            "grasp_pitch_degrees": float(np.rad2deg(pitch)),
            "release_tcp_base": release.tolist(),
            "motions": [vars(result) for result in c.results]}
