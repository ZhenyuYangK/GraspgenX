"""Procedural object prototypes and reproducible, collision-free initial layouts.

All six objects are free rigid bodies. Primitive compounds keep the cup and
tape ring hollow for contact simulation; no object is attached to the robot.
Lengths below are explicit prototype dimensions, in metres.
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HOME = np.array([-1., 0., 0., -1.57079, 0., 1.57079, -.7853, .04, .04])
SILVER = (.63, .67, .72, 1.)
BLACK = (.08, .09, .11, 1.)
RED = (.77, .055, .035, 1.)
CREAM = (.90, .86, .70, 1.)


def _s(values):
    return " ".join(f"{float(x):.8g}" for x in values)


def load_config(path: Path | str | None = None) -> dict:
    with Path(path or ROOT / "configs/task.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    names = [obj["name"] for obj in config["objects"]]
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError("Expected six unique object names")
    if set(names) != {"cup", "spoon", "wrench", "screwdriver", "stapler", "tape_dispenser"}:
        raise ValueError("The configured objects must match the six prototype builders")
    for obj in config["objects"]:
        if obj["category"] not in config["trays"] or obj["radius_m"] <= 0:
            raise ValueError(f"Invalid category or bounding radius: {obj['name']}")
    for tray in config["trays"].values():
        if len(tray["center_xy_m"]) != 2 or min(tray["inner_half_size_xy_m"]) <= 0:
            raise ValueError("Invalid tray dimensions")
    return config


def sample_layout(config: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    region = config["layout"]
    lo = np.array([region["x_m"][0], region["y_m"][0]])
    hi = np.array([region["x_m"][1], region["y_m"][1]])
    clearance = region["gripper_clearance_m"]
    objects = sorted(config["objects"], key=lambda item: -item["radius_m"])
    # Restart the whole packing to avoid a late-object dead end at this density.
    budget = int(region["max_attempts"])
    for _ in range(100):
        placed = {}
        for obj in objects:
            radius = obj["radius_m"]
            if np.any(lo + radius >= hi - radius):
                raise ValueError(f"Sampling region is too small for {obj['name']}")
            for _ in range(max(1, budget // (100 * len(objects)))):
                xy = rng.uniform(lo + radius, hi - radius)
                if any(np.linalg.norm(xy - np.array(p["xyz_m"][:2])) <
                       radius + p["radius_m"] + clearance for p in placed.values()):
                    continue
                yaw = float(rng.uniform(-np.pi, np.pi))
                placed[obj["name"]] = {"xyz_m": [*map(float, xy), .001],
                                       "yaw_rad": yaw, "radius_m": radius}
                break
            else:
                break
        if len(placed) == len(objects):
            return {"seed": int(seed), "objects": placed}
    raise RuntimeError("Could not place six objects with the configured clearance")


def _geom(body, kind, *, size=None, pos=None, rgba=SILVER, **extra):
    attrs = {"type": kind, "rgba": _s(rgba), "density": "450", "condim": "4",
             "friction": "1.2 .01 .001"}
    if size is not None:
        attrs["size"] = _s(size)
    if pos is not None:
        attrs["pos"] = _s(pos)
    attrs.update({key: _s(value) if isinstance(value, (list, tuple, np.ndarray)) else str(value)
                  for key, value in extra.items()})
    return ET.SubElement(body, "geom", attrs)


def _ring(body, center, radius, radial_half, axial_half, plane="xy", rgba=SILVER, n=24):
    """Tangential boxes form an open ring, including its concave collision space."""
    center = np.array(center)
    for i in range(n):
        angle = 2 * np.pi * i / n
        c, s = np.cos(angle), np.sin(angle)
        if plane == "xy":
            p = center + radius * np.array([c, s, 0])
            _geom(body, "box", size=[radial_half, radius * np.tan(np.pi/n) + .0003, axial_half],
                  pos=p, euler=[0, 0, angle], rgba=rgba)
        else:
            p = center + radius * np.array([c, 0, s])
            _geom(body, "box", size=[radial_half, axial_half, radius * np.tan(np.pi/n) + .0003],
                  pos=p, euler=[0, -angle, 0], rgba=rgba)


def _cup(body):
    _geom(body, "cylinder", pos=[0, 0, .004], size=[.033, .004], rgba=CREAM)
    _ring(body, [0, 0, .042], .030, .003, .035, rgba=CREAM)
    _ring(body, [.044, 0, .040], .022, .004, .005, plane="xz", rgba=CREAM, n=20)


def _spoon(body):
    _geom(body, "capsule", size=[.012], fromto=[-.061, 0, .013, .023, 0, .013])
    _geom(body, "ellipsoid", size=[.030, .021, .012], pos=[.042, 0, .013])
    # A darker inset makes the shallow bowl legible without a texture label.
    _geom(body, "ellipsoid", size=[.022, .014, .002], pos=[.044, 0, .024],
          rgba=[.40, .44, .48, 1], contype="0", conaffinity="0", density="0")


def _wrench(body):
    _geom(body, "box", size=[.051, .012, .013], pos=[-.004, 0, .013])
    _ring(body, [-.064, 0, .013], .017, .005, .013, n=12)
    _geom(body, "box", size=[.013, .025, .013], pos=[.049, 0, .013])
    for side in [-1, 1]:
        _geom(body, "box", size=[.013, .007, .013], pos=[.066, side*.019, .013])


def _screwdriver(body):
    _geom(body, "capsule", size=[.017], fromto=[-.064, 0, .018, -.018, 0, .018], rgba=RED)
    _geom(body, "cylinder", size=[.012, .008], pos=[-.01, 0, .018], euler=[0, np.pi/2, 0], rgba=BLACK)
    _geom(body, "capsule", size=[.005], fromto=[-.005, 0, .018, .066, 0, .018])
    _geom(body, "box", size=[.008, .008, .003], pos=[.072, 0, .018])


def _stapler(body):
    _geom(body, "box", size=[.053, .022, .007], pos=[0, 0, .007], rgba=BLACK)
    _geom(body, "box", size=[.049, .018, .004], pos=[-.001, 0, .019])
    _geom(body, "box", size=[.050, .021, .012], pos=[-.002, 0, .035],
          euler=[0, .16, 0], rgba=[.055, .19, .43, 1])
    _geom(body, "cylinder", size=[.016, .022], pos=[-.038, 0, .027], euler=[np.pi/2, 0, 0], rgba=BLACK)
    _geom(body, "box", size=[.006, .02, .008], pos=[.043, 0, .027])


def _tape_dispenser(body):
    _geom(body, "box", size=[.054, .027, .008], pos=[0, 0, .008], rgba=BLACK)
    for y in [-.021, .021]:
        _geom(body, "box", size=[.022, .004, .017], pos=[-.023, y, .027], rgba=BLACK)
    _ring(body, [-.021, 0, .046], .021, .007, .016, plane="xz", rgba=[.83, .60, .22, 1])
    _geom(body, "box", size=[.018, .016, .0015], pos=[.018, 0, .050],
          euler=[0, .56, 0], rgba=[.90, .78, .48, 1], contype="0", conaffinity="0", density="0")
    _geom(body, "box", size=[.008, .025, .005], pos=[.044, 0, .026], rgba=BLACK)
    _geom(body, "box", size=[.002, .023, .003], pos=[.049, 0, .034])


BUILDERS = {"cup": _cup, "spoon": _spoon, "wrench": _wrench, "screwdriver": _screwdriver,
            "stapler": _stapler, "tape_dispenser": _tape_dispenser}


def _camera_axes(position, target):
    direction = np.array(target) - np.array(position)
    direction /= np.linalg.norm(direction)
    right = np.cross(direction, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    return _s([*right, *up])


def build_scene(config: dict, layout: dict, output: Path) -> Path:
    """Write an independent MJCF beside the copied Panda include."""
    root = ET.Element("mujoco", model="panda_semantic_tidying")
    # Absolute include makes outputs portable within this installation.
    ET.SubElement(root, "include", file=str(ROOT / "assets/panda_scene.xml"))
    ET.SubElement(root, "compiler", angle="radian", autolimits="true",
                  meshdir=str(ROOT / "assets/robots/franka_emika_panda/assets"))
    ET.SubElement(root, "option", timestep=".002", integrator="implicitfast", gravity="0 0 -9.81",
                  cone="elliptic", iterations="80", tolerance="1e-9")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", ambient=".25 .25 .25", diffuse=".35 .35 .35")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos=".4 -.3 1.8", dir="0 0 -1", diffuse=".55 .55 .55")
    _geom(world, "plane", pos=[0, 0, -.75], size=[3, 3, .1], rgba=[.20, .22, .25, 1])
    _geom(world, "box", pos=[.40, 0, -.035], size=[.65, .57, .035], rgba=[.48, .46, .42, 1])
    for name, pos, target, fovy in [
            ("eye_to_hand", [.95, -.25, 1.15], [.48, -.03, 0], 50),
            ("overview", [1.25, -1.35, 1.10], [.43, -.01, .13], 45)]:
        ET.SubElement(world, "camera", name=name, pos=_s(pos),
                      xyaxes=_camera_axes(pos, target), fovy=str(fovy))
    for category, tray in config["trays"].items():
        body = ET.SubElement(world, "body", name=f"tray_{category}", pos=_s([*tray["center_xy_m"], 0]))
        hx, hy = tray["inner_half_size_xy_m"]
        color = {"kitchen": [.27, .47, .42, 1], "tools": [.36, .42, .54, 1], "office": [.58, .40, .28, 1]}[category]
        _geom(body, "box", pos=[0, 0, .004], size=[hx+.006, hy+.006, .004], rgba=color)
        for side in [-1, 1]:
            _geom(body, "box", pos=[side*(hx+.003), 0, .018], size=[.003, hy+.006, .010], rgba=color)
            _geom(body, "box", pos=[0, side*(hy+.003), .018], size=[hx, .003, .010], rgba=color)
    for obj in config["objects"]:
        item = layout["objects"][obj["name"]]
        body = ET.SubElement(world, "body", name=obj["name"], pos=_s(item["xyz_m"]),
                             euler=_s([0, 0, item["yaw_rad"]]))
        ET.SubElement(body, "freejoint", name=f"{obj['name']}_joint")
        BUILDERS[obj["name"]](body)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="unicode")
    return output


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    import mujoco
    for index in range(7):
        joint = model.joint(f"joint{index+1}")
        data.qpos[joint.qposadr] = HOME[index]
        data.ctrl[model.actuator(f"actuator{index+1}").id] = HOME[index]
    for name in ["finger_joint1", "finger_joint2"]:
        data.qpos[model.joint(name).qposadr] = .04
    data.ctrl[model.actuator("actuator8").id] = 255.
    mujoco.mj_forward(model, data)


def body_bounds(model, data, name: str, *, collision_only=True):
    body_id = model.body(name).id
    indices = np.flatnonzero(model.geom_bodyid == body_id)
    corners = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    points = []
    for index in indices:
        if collision_only and not (model.geom_contype[index] or model.geom_conaffinity[index]):
            continue
        local = model.geom_aabb[index, :3] + corners * model.geom_aabb[index, 3:]
        points.append(local @ data.geom_xmat[index].reshape(3, 3).T + data.geom_xpos[index])
    if not points:
        raise ValueError(f"No geometry for {name}")
    vertices = np.concatenate(points)
    return vertices.min(axis=0), vertices.max(axis=0)


def evaluate(model, data, config) -> dict:
    """Independent ground-truth evaluator; never supplies perception targets."""
    import mujoco
    details = []
    limits = config["evaluation"]
    for obj in config["objects"]:
        low, high = body_bounds(model, data, obj["name"], collision_only=False)
        tray = config["trays"][obj["category"]]
        center, half = np.array(tray["center_xy_m"]), np.array(tray["inner_half_size_xy_m"])
        inside = bool(np.all(low[:2] >= center-half) and np.all(high[:2] <= center+half))
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                model.body(obj["name"]).id, velocity, 0)
        linear = float(np.linalg.norm(velocity[3:]))
        angular = float(np.linalg.norm(velocity[:3]))
        # A tray floor is 8 mm above the table. This also rejects a held object.
        supported = bool(abs(low[2] - .008) < limits["floor_gap_m"])
        stable = linear < limits["max_linear_speed_m_s"] and angular < limits["max_angular_speed_rad_s"]
        details.append({"name": obj["name"], "category": obj["category"],
                        "success": bool(inside and supported and stable), "inside_correct_tray": inside,
                        "supported": supported, "stable": bool(stable),
                        "linear_speed_m_s": linear, "angular_speed_rad_s": angular,
                        "bounds_min_m": low.tolist(), "bounds_max_m": high.tolist()})
    return {"success": all(item["success"] for item in details),
            "sorted_count": sum(item["success"] for item in details), "objects": details}
