"""Procedural object prototypes and reproducible, collision-free initial layouts.

All six objects are free rigid bodies. Compound collision shapes preserve
the hollow parts of the objects.
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
BOX_FLOOR_HEIGHT = .008
BOX_RIM_HEIGHT = .105
BOX_OPEN_ANGLE = np.deg2rad(110)
BOX_COLORS = {"kitchen": (.27, .47, .42), "tools": (.36, .42, .54),
              "office": (.58, .40, .28)}


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
    attrs = {"type": kind, "density": "450", "condim": "4",
             "friction": "1.2 .01 .001"}
    if rgba is not None:
        attrs["rgba"] = _s(rgba)
    if size is not None:
        attrs["size"] = _s(size)
    if pos is not None:
        attrs["pos"] = _s(pos)
    attrs.update({key: _s(value) if isinstance(value, (list, tuple, np.ndarray)) else str(value)
                  for key, value in extra.items()})
    return ET.SubElement(body, "geom", attrs)


def _loft_faces(sections, *, loop=False):
    """Triangulate a surface between cross sections with matching vertices."""
    count, width = sections.shape[:2]
    faces = []
    for section in range(count if loop else count - 1):
        start = section * width
        end = (section + 1) % count * width
        for index in range(width):
            following = (index + 1) % width
            faces.extend([(start + index, start + following, end + following),
                          (start + index, end + following, end + index)])
    if not loop:
        end = (count - 1) * width
        for index in range(1, width - 1):
            faces.extend([(0, index + 1, index),
                          (end, end + index, end + index + 1)])
    return faces


def _mesh_part(body, assets, name, sections, collision_parts, *, material, loop=False, density=7800):
    vertices = _s(sections.ravel())
    faces = " ".join(str(i) for face in _loft_faces(sections, loop=loop) for i in face)
    ET.SubElement(assets, "mesh", name=name, vertex=vertices, face=faces, smoothnormal="true")
    _geom(body, "mesh", mesh=name, rgba=None, material=material,
          contype="0", conaffinity="0", density="0", group="2")
    # MuJoCo computes a convex hull for each collision mesh without faces.
    for index, part in enumerate(collision_parts):
        hull = f"{name}_collision_{index}"
        ET.SubElement(assets, "mesh", name=hull, vertex=_s(part.ravel()))
        _geom(body, "mesh", mesh=hull, density=density, group="3")


def _cup(body, assets):
    angle = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    circle = np.column_stack((np.cos(angle), np.sin(angle)))
    # Revolve the ceramic profile from the recessed underside to the inner floor.
    foot = [(.005, .0025), (.021, .0025), (.023, .0018), (.024, .0005),
            (.025, 0), (.027, 0), (.028, .0009), (.0285, .0036)]
    wall = [(.030, .005), (.032, .0075), (.034, .012), (.035, .021),
            (.0357, .042), (.0369, .065), (.0374, .080), (.0374, .082),
            (.0370, .0834), (.0360, .0842), (.0348, .0844), (.0336, .0840),
            (.0329, .0830), (.0327, .0817), (.0324, .066), (.0312, .044),
            (.0304, .024), (.0298, .017), (.0285, .013), (.0257, .0103)]
    floor = [(.022, .0093), (.012, .0090), (.005, .0090)]
    shell = np.array([np.column_stack((radius * circle, np.full(len(angle), z)))
                      for radius, z in foot + wall + floor])

    # Short wedges leave the cavity open. A flat circular contact supports the foot.
    wall_sections = shell[len(foot):len(foot) + len(wall)]
    collision = [wall_sections[:, np.arange(start, start + 4) % len(angle)]
                 for start in range(0, len(angle), 3)]
    collision.append(shell[[len(foot) - 1, len(foot) + 1, -3], ::3])
    _mesh_part(body, assets, "cup_shell", shell, collision,
               material="cup_glaze", density=2350)
    _geom(body, "cylinder", size=[.027, .0018], pos=[0, 0, .0018],
          group="3", density=2350)

    # The C-shaped handle widens into the wall at both roots.
    arc = np.linspace(-np.pi / 2, np.pi / 2, 25)
    cos_arc, sin_arc = np.cos(arc), np.sin(arc)
    centers = np.column_stack((.033 + .033 * cos_arc, np.zeros(len(arc)),
                               .044 + .023 * sin_arc))
    normals = np.column_stack((cos_arc / .033, np.zeros(len(arc)), sin_arc / .023))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    section_angle = angle[::4]
    thickness = .0048 + .0015 * sin_arc**4
    width = .0060 + .0020 * sin_arc**4
    handle = (centers[:, None, :]
              + (thickness[:, None] * np.cos(section_angle))[:, :, None] * normals[:, None, :]
              + (width[:, None] * np.sin(section_angle))[:, :, None] * [0, 1, 0])
    _mesh_part(body, assets, "cup_handle", handle,
               [handle[start:start + 5] for start in range(0, len(arc) - 1, 4)],
               material="cup_glaze", density=2350)


def _spoon(body, assets):
    angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    c, s = np.cos(angle), np.sin(angle)
    radius = np.array([.14, .34, .55, .73, .87, .96, 1.])[:, None]
    x = .042 + .030 * radius * c
    y = .019 * radius * s * (1 - .14 * radius * c)
    depth = np.maximum(0, (radius**2 - .14**2) / (1 - .14**2))
    surface = .0018 + (.011 - .0015 * c) * depth
    bottom = np.stack((x, y, surface - .0015), axis=-1)
    top = np.stack((x, y, surface), axis=-1)
    rim = (top[-1] + bottom[-1]) / 2
    rim[:, :2] += np.column_stack((c, s)) * .0004
    bowl = np.concatenate((bottom, rim[None], top[::-1]))

    # The center patch and short radial wedges follow the bowl on both sides.
    bowl_collision = [np.concatenate((bottom[0], top[0]))[::2]]
    for inner, outer in zip((0, 2, 4), (2, 4, 6)):
        for start in range(0, len(angle), 4):
            arc = np.arange(start, start + 5, 2) % len(angle)
            bowl_collision.append(np.concatenate((bottom[[inner, outer]][:, arc],
                                                  top[[inner, outer]][:, arc])))
    _mesh_part(body, assets, "spoon_bowl", bowl, bowl_collision, material="flatware_steel")

    # Axial stations give the handle a rounded butt and a raised, narrower neck.
    stations = [
        (-.073, .0004, .0017, .0003), (-.072, .0030, .0017, .0011),
        (-.069, .0061, .0017, .0016), (-.063, .0072, .0018, .0017),
        (-.050, .0068, .0028, .0017), (-.033, .0054, .0044, .0016),
        (-.015, .0040, .0069, .0015), ( .000, .0034, .0091, .0015),
        ( .010, .0041, .0119, .0014), ( .012, .0043, .0119, .0013),
        ( .015, .0046, .0105, .0012), ( .019, .0029, .0080, .0007),
    ]
    angle = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    handle = np.array([np.column_stack((np.full(len(angle), x), width * np.cos(angle),
                                        z + height * np.sin(angle)))
                       for x, width, z, height in stations])
    _mesh_part(body, assets, "spoon_handle", handle,
               [handle[start:end] for start, end in zip((0, 3, 5, 7), (4, 6, 8, len(handle)))],
               material="flatware_steel")


def _wrench_head_sections(outer, inner, height):
    """Bevel the outer edge and the working faces of a wrench head."""
    radial = outer - inner
    bevel = .00065 * radial / np.linalg.norm(radial, axis=1, keepdims=True)
    columns = [(outer, height - .00065), (outer, .00065), (outer - bevel, 0),
               (inner + bevel, 0), (inner, .00065), (inner, height - .00065),
               (inner + bevel, height), (outer - bevel, height)]
    return np.stack([np.column_stack((xy, np.full(len(xy), z))) for xy, z in columns], axis=1)


def _wrench(body, assets):
    # A 12-point box end. Short convex sections leave the bore open.
    angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    circle = np.column_stack((np.cos(angle), np.sin(angle)))
    points = circle[::2] * np.tile([.0118, .0100], 12)[:, None]
    inner = np.stack((points, (points + np.roll(points, -1, axis=0)) / 2), axis=1).reshape(-1, 2)
    center = np.array([-.063, 0])
    ring = _wrench_head_sections(center + .019 * circle, center + inner, .009)
    _mesh_part(body, assets, "wrench_box_end", ring,
               [ring[[i, (i+1) % len(ring), (i+2) % len(ring)]] for i in range(0, len(ring), 2)],
               material="tool_steel", loop=True)

    # Parallel jaws transition to the relieved throat, rotated 15 degrees from the handle.
    jaw = np.array([[.018, .009], [-.002, .009], [-.009, .003],
                    [-.009, -.003], [-.002, -.009], [.018, -.009]])
    inner = np.concatenate([np.linspace(a, b, 7, endpoint=False) for a, b in zip(jaw, jaw[1:])]
                           + [jaw[-1:]])
    angle = np.unwrap(np.arctan2(inner[:, 1], inner[:, 0]))
    angle = np.interp(angle, [angle[0], angle[-1]], np.deg2rad([34, 326]))
    outer = np.column_stack((.022 * np.cos(angle), .022 * np.sin(angle)))
    c, s = np.cos(np.pi / 12), np.sin(np.pi / 12)
    rotation = np.array([[c, s], [-s, c]])
    center = np.array([.060, 0])
    head = _wrench_head_sections(center + outer @ rotation, center + inner @ rotation, .008)
    _mesh_part(body, assets, "wrench_open_end", head,
               [head[i:i+4] for i in range(0, len(head) - 1, 3)], material="tool_steel")

    # The forged shank flares and thickens where it meets each head.
    stations = [(-.050, .0100, .002, .0078), (-.041, .0097, .0017, .0072),
                (-.029, .0075, .0017, .0064), ( .010, .0074, .0017, .0064),
                ( .029, .0085, .0015, .0069), ( .042, .0100, .0013, .0070),
                ( .049, .0115, .0013, .0070)]
    handle = []
    for x, width, bottom, top in stations:
        bevel = .0009
        profile = [(width, bottom + bevel), (width, top - bevel), (width - bevel, top),
                   (-width + bevel, top), (-width, top - bevel), (-width, bottom + bevel),
                   (-width + bevel, bottom), (width - bevel, bottom)]
        handle.append([(x, y, z) for y, z in profile])
    handle = np.array(handle)
    _mesh_part(body, assets, "wrench_handle", handle,
               [handle[i:i+2] for i in range(len(handle) - 1)], material="tool_steel")


def _screwdriver(body, assets):
    angle = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    # Palm swell, finger waist and flared guard of a two-material moulded handle.
    # The slightly flattened six-lobed section also keeps the tool from rolling.
    stations = [
        (-.080, .0030, .0030), (-.079, .0068, .0065),
        (-.076, .0108, .0105), (-.070, .0138, .0135),
        (-.060, .0150, .0148), (-.047, .0150, .0148),
        (-.035, .0135, .0132), (-.024, .0108, .0105),
        (-.019, .0100, .0097), (-.016, .0125, .0122),
        (-.013, .0125, .0122), (-.011, .0095, .0092),
    ]
    center_z = .0153

    def profile(x, width, height, theta):
        lobes = 1 + .035 * np.cos(6 * theta)
        return np.column_stack((np.full(len(theta), x), width * lobes * np.cos(theta),
                                center_z + height * lobes * np.sin(theta)))

    handle = np.array([profile(x, w, h, angle) for x, w, h in stations])
    _mesh_part(body, assets, "screwdriver_handle", handle,
               [handle[i:i+3] for i in range(0, len(handle) - 2, 2)] + [handle[-2:]],
               material="screwdriver_red", density=1050)

    # Six rubber inserts follow the handle rather than floating over a capsule.
    grip_x = np.array([-.073, -.070, -.060, -.047, -.035, -.025, -.022])
    widths = np.interp(grip_x, np.array(stations)[:, 0], np.array(stations)[:, 1])
    heights = np.interp(grip_x, np.array(stations)[:, 0], np.array(stations)[:, 2])
    spans = [.055, .22, .24, .24, .22, .18, .045]
    for index in range(6):
        sections = []
        for x, w, h, span in zip(grip_x, widths, heights, spans):
            arc = index * np.pi / 3 + np.linspace(-span, span, 9)
            outer = profile(x, w + .00018, h + .00018, arc)
            inner = profile(x, w - .00025, h - .00025, arc[::-1])
            sections.append(np.concatenate((outer, inner)))
        _mesh_part(body, assets, f"screwdriver_grip_{index}", np.array(sections), [],
                   material="grip_rubber")

    collar = np.array([profile(x, r, r, angle) for x, r in
                       [(-.012, .0086), (-.0105, .0089), (-.008, .0089), (-.0065, .0068)]])
    _mesh_part(body, assets, "screwdriver_collar", collar, [collar],
               material="grip_rubber", density=1100)
    ferrule = np.array([profile(x, r, r, angle) for x, r in
                        [(-.007, .0047), (-.0063, .0049), (-.0035, .0049), (-.0028, .0031)]])
    _mesh_part(body, assets, "screwdriver_ferrule", ferrule, [ferrule], material="tool_steel")

    # The working end is a forged, tapered flat blade, narrower than the handle.
    _geom(body, "cylinder", size=[.0031, .0325], pos=[.0285, 0, center_z],
          euler=[0, np.pi / 2, 0], rgba=None, material="driver_chrome",
          contype=0, conaffinity=0, density=0, group=2)
    _geom(body, "cylinder", size=[.0031, .0325], pos=[.0285, 0, center_z],
          euler=[0, np.pi / 2, 0], group=3, density=7800)
    blade = _rounded_sections([
        (.059, .0029, center_z - .0029, center_z + .0029),
        (.064, .0036, center_z - .0022, center_z + .0022),
        (.070, .0038, center_z - .0013, center_z + .0013),
        (.078, .0036, center_z - .00055, center_z + .00055),
        (.081, .0034, center_z - .00045, center_z + .00045),
    ], .00035)
    _mesh_part(body, assets, "screwdriver_blade", blade,
               [blade[i:i+2] for i in range(len(blade) - 1)], material="driver_chrome")


def _rounded_sections(stations, corner_radius):
    """Rounded rectangular sections for a part whose silhouette varies along x."""
    angle = np.linspace(0, np.pi / 2, 5) + (np.pi / 2) * np.arange(4)[:, None]
    arcs = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    sections = []
    for x, width, bottom, top in stations:
        r = min(corner_radius, width * .8, (top - bottom) * .4)
        corners = np.array([(width - r, top - r), (-width + r, top - r),
                            (-width + r, bottom + r), (width - r, bottom + r)])
        yz = (corners[:, None, :] + r * arcs).reshape(-1, 2)
        sections.append(np.column_stack((np.full(len(yz), x), yz)))
    return np.array(sections)


def _stapler(body, assets):
    # Each station gives x, half width, and the lower and upper heights.
    pad = _rounded_sections([
        (-.051, .009, .0007, .0021), (-.048, .015, .0002, .0026),
        (-.043, .018, 0, .003), (.044, .018, 0, .003),
        (.049, .015, .0002, .0026), (.052, .009, .0007, .0021),
    ], .001)
    _mesh_part(body, assets, "stapler_pad", pad, [], material="stapler_rubber")
    _geom(body, "box", size=[.0435, .017, .0015], pos=[.0005, 0, .0015],
          group=3, density=1100)

    # The front of the magazine is raised above the clinching plate.
    parts = [
        ("base", "stapler_base", 2100, .0026, [
            (-.054, .008, .0034, .0064), (-.052, .014, .0024, .0086),
            (-.047, .018, .002, .0100), (-.040, .019, .002, .0104),
            (.041, .019, .002, .0104), (.048, .0175, .0023, .0099),
            (.052, .014, .0031, .0086), (.054, .008, .0042, .0072),
        ]),
        ("hinge", "stapler_base", 2100, .003, [
            (-.050, .008, .008, .024), (-.047, .0135, .008, .030),
            (-.041, .0145, .008, .0315), (-.033, .014, .008, .028),
            (-.026, .011, .008, .023), (-.022, .008, .008, .016),
        ]),
        ("magazine", "stapler_steel", 3100, .0012, [
            (-.046, .009, .019, .025), (-.040, .0115, .0195, .028),
            (-.018, .0120, .0215, .030), (.033, .0123, .027, .035),
            (.047, .0118, .0285, .0365), (.051, .010, .030, .0363),
        ]),
        ("cover", "stapler_blue", 850, .0045, [
            (-.052, .007, .027, .035), (-.049, .012, .024, .038),
            (-.043, .015, .024, .042), (-.032, .016, .026, .045),
            (-.010, .0165, .029, .048), (.012, .018, .031, .0505),
            (.031, .0185, .0325, .051), (.043, .0178, .0335, .0495),
            (.050, .0145, .035, .0465), (.053, .010, .0365, .0435),
            (.054, .005, .038, .0415),
        ]),
        ("anvil", "stapler_steel", 7800, .0008, [
            (.024, .005, .0092, .0115), (.026, .008, .0092, .013),
            (.030, .009, .0092, .0132), (.047, .009, .0092, .0132),
            (.050, .007, .0092, .0125), (.051, .0045, .0092, .0113),
        ]),
    ]
    for name, material, density, radius, stations in parts:
        sections = _rounded_sections(stations, radius)
        _mesh_part(body, assets, f"stapler_{name}", sections, [sections],
                   material=material, density=density)

    _geom(body, "cylinder", pos=[-.041, 0, .0185], size=[.0042, .0148],
          euler=[np.pi / 2, 0, 0], rgba=None, material="stapler_steel",
          contype=0, conaffinity=0, density=0, group=2)
    for x in [.036, .042]:
        _geom(body, "capsule", size=[.00075], fromto=[x, -.0055, .01255, x, .0055, .01255],
              rgba=[.22, .24, .27, 1], contype=0, conaffinity=0, density=0, group=2)


def _tape_dispenser(body, assets):
    pad = _rounded_sections([
        (-.056, .016, .0005, .0025), (-.052, .023, 0, .003),
        (.049, .023, 0, .003), (.055, .016, .0005, .0025),
    ], .001)
    _mesh_part(body, assets, "tape_pad", pad, [pad], material="grip_rubber", density=1100)
    base = _rounded_sections([
        (-.057, .014, .003, .008), (-.055, .021, .002, .012),
        (-.050, .026, .002, .015), (-.039, .027, .002, .016),
        (.033, .026, .002, .016), (.050, .024, .003, .015),
        (.056, .019, .004, .012), (.058, .010, .005, .009),
    ], .003)
    _mesh_part(body, assets, "tape_base", base,
               [base[i:i+3] for i in range(0, len(base) - 2, 2)] + [base[-2:]],
               material="dispenser_shell", density=2300)

    # Sculpted cheeks support the hollow spool and leave the reel bore visible.
    stations = [
        (-.055, .0015, .010, .022), (-.052, .0030, .009, .033),
        (-.047, .0037, .009, .039), (-.040, .0040, .009, .041),
        (-.031, .0040, .009, .039), (-.022, .0040, .009, .0365),
        (-.013, .0040, .009, .033), (-.003, .0040, .009, .025),
        (.012, .0040, .009, .020), (.027, .0040, .009, .022),
        (.041, .0040, .009, .030), (.050, .0035, .010, .032),
        (.055, .0020, .013, .029),
    ]
    for side in [-1, 1]:
        cheek = _rounded_sections(stations, .0025)
        cheek[:, :, 1] += side * .021
        _mesh_part(body, assets, f"tape_cheek_{side}", cheek,
                   [cheek[i:i+2] for i in range(len(cheek) - 1)],
                   material="dispenser_shell", density=1100)

    angle = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    center = np.array([-.022, .047])

    def reel(name, profile, material, *, collision=False, density=950):
        sections = np.array([[(center[0] + r * np.cos(a), y,
                               center[1] + r * np.sin(a)) for r, y in profile] for a in angle])
        hulls = [sections[np.arange(i, i+5) % len(angle)]
                 for i in range(0, len(angle), 4)] if collision else []
        _mesh_part(body, assets, name, sections, hulls,
                   material=material, loop=True, density=density)

    reel("tape_spool", [(.0085, -.020), (.0102, -.020), (.0109, -.0193),
                        (.0109, .0193), (.0102, .020), (.0085, .020)],
         "dispenser_shell", collision=True, density=1100)
    reel("tape_paper_core", [(.0109, -.0168), (.0138, -.0168), (.0144, -.0162),
                             (.0144, .0162), (.0138, .0168), (.0109, .0168)],
         "tape_cardboard", collision=True, density=650)
    reel("tape_roll", [(.0143, -.0159), (.0255, -.0159), (.0263, -.0154),
                       (.0266, -.0146), (.0266, .0146), (.0263, .0154),
                       (.0255, .0159), (.0143, .0159)],
         "tape_amber", collision=True)
    # Thin concentric seams suggest wound film at both cut faces of the roll.
    for side in [-1, 1]:
        for index, radius in enumerate([.016, .0185, .021, .0235, .025]):
            y = side * .01596
            reel(f"tape_winding_{side}_{index}",
                 [(radius - .00007, y - .00004), (radius + .00007, y - .00004),
                  (radius + .00007, y + .00004), (radius - .00007, y + .00004)],
                 "tape_edge")

    nose = _rounded_sections([
        (.035, .014, .012, .021), (.043, .019, .012, .031),
        (.050, .021, .013, .033), (.055, .020, .015, .032),
    ], .002)
    _mesh_part(body, assets, "tape_nose", nose,
               [nose[i:i+2] for i in range(len(nose) - 1)],
               material="dispenser_shell", density=1100)
    plate = _rounded_sections([
        (.048, .020, .032, .0333), (.050, .021, .032, .0343),
        (.0547, .021, .032, .0343),
    ], .0004)
    _mesh_part(body, assets, "tape_cutter", plate, [plate], material="driver_chrome")
    for index, y in enumerate(np.linspace(-.0195, .0195, 19)):
        tooth = np.array([[(x, y - .00095, .0341), (x, y + .00095, .0341),
                           (x, y, .0362)] for x in [.0538, .0547]])
        _mesh_part(body, assets, f"tape_tooth_{index}", tooth, [], material="driver_chrome")

    # Analytic tangent joins the wound reel to the cutting edge without a floating strip.
    end = np.array([.0538, .0354])
    delta = end - center
    tangent_angle = np.arctan2(delta[1], delta[0]) + np.arccos(.0266 / np.linalg.norm(delta))
    start = center + .0266 * np.array([np.cos(tangent_angle), np.sin(tangent_angle)])
    strip = _rounded_sections([
        (x, .015, z - .00012, z + .00012)
        for x, z in np.linspace(start, end, 4)
    ], .00008)
    _mesh_part(body, assets, "tape_feed", strip, [strip], material="tape_film", density=950)


BUILDERS = {"cup": _cup, "spoon": _spoon, "wrench": _wrench, "screwdriver": _screwdriver,
            "stapler": _stapler, "tape_dispenser": _tape_dispenser}


def _camera_axes(position, target):
    direction = np.array(target) - np.array(position)
    direction /= np.linalg.norm(direction)
    right = np.cross(direction, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    return _s([*right, *up])


def _box_label(body, assets, category, half_width, front_y):
    name = f"storage_box_label_{category}"
    width, height, thickness = .86 * half_width, .022, .0005
    # Explicit UVs keep the text upright on the outward-facing plate.
    vertices = [(x, y, z) for y in (-thickness, thickness)
                for x, z in [(-width, -height), (width, -height),
                             (width, height), (-width, height)]]
    ET.SubElement(assets, "mesh", name=name, vertex=_s(np.ravel(vertices)),
                  texcoord="0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0",
                  face="0 1 2 0 2 3 4 7 6 4 6 5 0 4 5 0 5 1 "
                       "3 2 6 3 6 7 1 5 6 1 6 2 0 3 7 0 7 4")
    ET.SubElement(assets, "texture", name=name, type="2d",
                  file=str(ROOT / "assets" / "box_labels" / f"{category}.png"))
    ET.SubElement(assets, "material", name=name, texture=name,
                  texuniform="false", specular="0", shininess="0")
    _geom(body, "mesh", mesh=name, material=name, rgba=None,
          pos=[0, front_y - thickness, .056], contype="0", conaffinity="0", density="0")


def _storage_box(world, assets, actuators, category, box, *, powered_lid):
    body = ET.SubElement(world, "body", name=f"storage_box_{category}",
                         pos=_s([*box["center_xy_m"], 0]))
    hx, hy = box["inner_half_size_xy_m"]
    wall = .005
    outer_x, outer_y = hx + wall, hy + wall
    rgb = np.array(BOX_COLORS[category])
    shell, rim, hardware = [*rgb, 1], [*(rgb * .72), 1], [.09, .10, .11, 1]
    floor_half = BOX_FLOOR_HEIGHT / 2
    wall_half = (BOX_RIM_HEIGHT - BOX_FLOOR_HEIGHT) / 2
    wall_center = BOX_FLOOR_HEIGHT + wall_half
    _geom(body, "box", pos=[0, 0, floor_half], size=[outer_x, outer_y, floor_half],
          rgba=[*(rgb * .78), 1])
    for side in (-1, 1):
        x, y = side * (hx + wall / 2), side * (hy + wall / 2)
        _geom(body, "box", pos=[x, 0, wall_center], size=[wall / 2, outer_y, wall_half], rgba=shell)
        _geom(body, "box", pos=[0, y, wall_center], size=[hx, wall / 2, wall_half], rgba=shell)
        for center, size in [([x, 0], [wall / 2 + .0007, outer_y + .0007]),
                             ([0, y], [hx * .99, wall / 2 + .0007])]:
            _geom(body, "box", pos=[*center, BOX_RIM_HEIGHT - .002], size=[*size, .002],
                  rgba=rim, contype="0", conaffinity="0", density="0")
    _box_label(body, assets, category, hx, -outer_y)

    name = f"storage_box_lid_{category}"
    lid = ET.SubElement(body, "body", name=name, pos=_s([0, outer_y, BOX_RIM_HEIGHT + .005]))
    ET.SubElement(lid, "joint", name=name, type="hinge", axis="-1 0 0",
                  range=_s([0, BOX_OPEN_ANGLE]), damping=".04" if powered_lid else ".012")
    if not powered_lid:
        # A counterweighted lid stays against the nearest stop once released
        # on either side of its over-center point (about 36 degrees).
        ET.SubElement(lid, "inertial", pos="0 -.035 .048", mass=".16", diaginertia=".001 .0011 .0011")
    _geom(lid, "box", name=f"{name}_panel", pos=[0, -outer_y, 0],
          size=[outer_x + .001, outer_y + .001, .004], rgba=rim)
    _geom(lid, "box", pos=[0, -outer_y, .0045], size=[hx * .99, hy * .99, .0005],
          rgba=shell, contype="0", conaffinity="0", density="0")
    hinge_collision = {} if powered_lid else {"contype": "0", "conaffinity": "0"}
    for x in (-.68 * hx, .68 * hx):
        _geom(lid, "cylinder", pos=[x, 0, 0], size=[.006, .011],
              quat=[.70710678, 0, .70710678, 0], rgba=hardware, **hinge_collision)
    if powered_lid:
        handle_y = -2 * outer_y + .028
        for x in (-.032, .032):
            _geom(lid, "box", pos=[x, handle_y, .010], size=[.005, .006, .006], rgba=hardware)
        _geom(lid, "capsule", name=f"{name}_handle", fromto=[-.034, handle_y, .018, .034, handle_y, .018],
              size=[.005], rgba=hardware)
        ET.SubElement(actuators, "position", name=name, joint=name,
                      ctrlrange=_s([0, BOX_OPEN_ANGLE]), forcerange="-1.2 1.2", kp="18", kv=".35")
    else:
        _geom(lid, "cylinder", pos=[-.065, -.070, .021], size=[.006, .021], rgba=hardware)
        _geom(lid, "sphere", name=f"{name}_handle", pos=[-.065, -.070, .043], size=[.019], rgba=hardware)
        # The underside pull is accessible from the front when the lid is open.
        handle_x = .075 if category == "kitchen" else -.082
        _geom(lid, "cylinder", pos=[handle_x, -.145, -.0185], size=[.006, .0185], rgba=hardware)
        _geom(lid, "sphere", name=f"{name}_closing_handle", pos=[handle_x, -.145, -.037],
              size=[.017], rgba=hardware)


def build_scene(config: dict, layout: dict, output: Path, *, powered_lids: bool = True) -> Path:
    """Write an independent MJCF beside the copied Panda include."""
    root = ET.Element("mujoco", model="panda_semantic_tidying")
    # Absolute include makes outputs portable within this installation.
    ET.SubElement(root, "include", file=str(ROOT / "assets/panda_scene.xml"))
    ET.SubElement(root, "compiler", angle="radian", autolimits="true",
                  meshdir=str(ROOT / "assets/robots/franka_emika_panda/assets"))
    ET.SubElement(root, "option", timestep=".002", integrator="implicitfast", gravity="0 0 -9.81",
                  cone="elliptic", iterations="80", tolerance="1e-9")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "material", name="flatware_steel", rgba=".58 .61 .65 1",
                  specular=".40", shininess=".42")
    ET.SubElement(assets, "material", name="tool_steel", rgba=".46 .49 .53 1",
                  specular=".27", shininess=".28")
    ET.SubElement(assets, "material", name="cup_glaze", rgba=".91 .85 .70 1",
                  specular=".48", shininess=".58")
    ET.SubElement(assets, "material", name="stapler_blue", rgba=".055 .20 .43 1",
                  specular=".32", shininess=".43")
    ET.SubElement(assets, "material", name="stapler_base", rgba=".09 .10 .12 1",
                  specular=".13", shininess=".20")
    ET.SubElement(assets, "material", name="stapler_rubber", rgba=".045 .047 .052 1",
                  specular=".035", shininess=".08")
    ET.SubElement(assets, "material", name="stapler_steel", rgba=".60 .64 .70 1",
                  specular=".47", shininess=".50")
    for name, rgba, specular, shininess in [
        ("screwdriver_red", ".68 .035 .023 1", ".25", ".32"),
        ("grip_rubber", ".035 .039 .044 1", ".045", ".08"),
        ("driver_chrome", ".66 .69 .73 1", ".65", ".70"),
        ("dispenser_shell", ".065 .075 .086 1", ".28", ".38"),
        ("tape_cardboard", ".66 .52 .34 1", ".035", ".08"),
        ("tape_amber", ".79 .63 .34 1", ".30", ".48"),
        ("tape_edge", ".68 .53 .28 1", ".12", ".25"),
        ("tape_film", ".85 .73 .47 1", ".42", ".58"),
    ]:
        ET.SubElement(assets, "material", name=name, rgba=rgba,
                      specular=specular, shininess=shininess)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", ambient=".25 .25 .25", diffuse=".35 .35 .35")
    world = ET.SubElement(root, "worldbody")
    actuators = ET.SubElement(root, "actuator")
    ET.SubElement(world, "light", pos=".4 -.3 1.8", dir="0 0 -1", diffuse=".55 .55 .55")
    _geom(world, "plane", pos=[0, 0, -.75], size=[3, 3, .1], rgba=[.20, .22, .25, 1])
    _geom(world, "box", pos=[.40, 0, -.035], size=[.65, .57, .035], rgba=[.48, .46, .42, 1])
    for name, pos, target, fovy in [
            ("eye_to_hand", [.95, -.25, 1.15], [.48, -.03, 0], 50),
            ("overview", [1.25, -1.35, 1.10], [.43, -.01, .13], 45)]:
        ET.SubElement(world, "camera", name=name, pos=_s(pos),
                      xyaxes=_camera_axes(pos, target), fovy=str(fovy))
    for category, box in config["trays"].items():
        _storage_box(world, assets, actuators, category, box, powered_lid=powered_lids)
    for obj in config["objects"]:
        item = layout["objects"][obj["name"]]
        body = ET.SubElement(world, "body", name=obj["name"], pos=_s(item["xyz_m"]),
                             euler=_s([0, 0, item["yaw_rad"]]))
        ET.SubElement(body, "freejoint", name=f"{obj['name']}_joint")
        BUILDERS[obj["name"]](body, assets)
        if not powered_lids:
            scale = config["execution"].get("pickup_thickness_scale", {}).get(obj["name"], 1.)
            if scale != 1.:
                meshes = {geom.attrib["mesh"] for geom in body.findall("geom")}
                for mesh in assets.findall("mesh"):
                    if mesh.attrib["name"] in meshes:
                        mesh.set("scale", _s([1., 1., scale]))
                for geom in body.findall("geom"):
                    geom.set("density", str(float(geom.attrib["density"]) / scale))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="unicode")
    return output


def initialize(model: mujoco.MjModel, data: mujoco.MjData, *, lids_open: bool = True) -> None:
    import mujoco
    for index in range(7):
        joint = model.joint(f"joint{index+1}")
        data.qpos[joint.qposadr] = HOME[index]
        data.ctrl[model.actuator(f"actuator{index+1}").id] = HOME[index]
    for name in ["finger_joint1", "finger_joint2"]:
        data.qpos[model.joint(name).qposadr] = .04
    data.ctrl[model.actuator("actuator8").id] = 255.
    for index in range(model.njnt):
        joint = model.joint(index)
        if joint.name.startswith("storage_box_lid_"):
            angle = joint.range[int(lids_open)]
            data.qpos[joint.qposadr] = angle
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint.name)
            if actuator_id >= 0:
                data.ctrl[actuator_id] = angle
    mujoco.mj_forward(model, data)


def body_bounds(model, data, name: str, *, collision_only=True):
    import mujoco

    body_id = model.body(name).id
    indices = np.flatnonzero(model.geom_bodyid == body_id)
    corners = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    points = []
    for index in indices:
        if collision_only and not (model.geom_contype[index] or model.geom_conaffinity[index]):
            continue
        if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH:
            # A rotated mesh bounding box can put the spoon's curved shell below the table.
            mesh = model.geom_dataid[index]
            start = model.mesh_vertadr[mesh]
            local = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        else:
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
        box = config["trays"][obj["category"]]
        center, half = np.array(box["center_xy_m"]), np.array(box["inner_half_size_xy_m"])
        inside = bool(np.all(low[:2] >= center-half) and np.all(high[:2] <= center+half))
        fits_below_rim = bool(high[2] <= BOX_RIM_HEIGHT)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                model.body(obj["name"]).id, velocity, 0)
        linear = float(np.linalg.norm(velocity[3:]))
        angular = float(np.linalg.norm(velocity[:3]))
        supported = bool(abs(low[2] - BOX_FLOOR_HEIGHT) < limits["floor_gap_m"])
        stable = linear < limits["max_linear_speed_m_s"] and angular < limits["max_angular_speed_rad_s"]
        details.append({"name": obj["name"], "category": obj["category"],
                        "success": bool(inside and fits_below_rim and supported and stable),
                        "inside_correct_tray": inside, "fits_below_rim": fits_below_rim,
                        "supported": supported, "stable": bool(stable),
                        "linear_speed_m_s": linear, "angular_speed_rad_s": angular,
                        "bounds_min_m": low.tolist(), "bounds_max_m": high.tolist()})
    return {"success": all(item["success"] for item in details),
            "sorted_count": sum(item["success"] for item in details), "objects": details}
