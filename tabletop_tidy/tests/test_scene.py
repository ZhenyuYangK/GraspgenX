import os
os.environ.setdefault("MUJOCO_GL", "egl")

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from tabletop_tidy.scene import (
    ROOT, body_bounds, build_scene, evaluate, initialize, load_config, sample_layout,
)
from tabletop_tidy.control import PandaController
from tabletop_tidy.camera import RGBDCamera


@pytest.fixture(scope="module")
def simulation(tmp_path_factory):
    config = load_config()
    path = build_scene(config, sample_layout(config, 42), tmp_path_factory.mktemp("scene") / "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    return config, model


def settled(simulation):
    config, model = simulation
    data = mujoco.MjData(model)
    initialize(model, data)
    controller = PandaController(model, data)
    controller.hold(1.5)
    return config, model, data, controller


def put_in_trays(config, model, data, controller):
    counts = dict.fromkeys(config["trays"], 0)
    for obj in config["objects"]:
        tray = config["trays"][obj["category"]]
        position = [*tray["center_xy_m"], .009]
        position[1] += -.050 if counts[obj["category"]] == 0 else .050
        counts[obj["category"]] += 1
        joint = model.joint(f"{obj['name']}_joint")
        data.qpos[joint.qposadr[0]:joint.qposadr[0]+7] = [*position, 1, 0, 0, 0]
        data.qvel[joint.dofadr[0]:joint.dofadr[0]+6] = 0
    mujoco.mj_forward(model, data)
    controller.hold(1.5)


def test_random_layouts_are_reproducible_and_separated():
    config = load_config()
    assert sample_layout(config, 42) == sample_layout(config, 42)
    assert sample_layout(config, 42) != sample_layout(config, 43)
    region = config["layout"]
    for seed in range(50):
        layout = sample_layout(config, seed)
        objects = list(layout["objects"].values())
        for index, obj in enumerate(objects):
            xy, radius = np.array(obj["xyz_m"][:2]), obj["radius_m"]
            assert np.all(xy-radius >= [region["x_m"][0], region["y_m"][0]])
            assert np.all(xy+radius <= [region["x_m"][1], region["y_m"][1]])
            for other in objects[index+1:]:
                assert np.linalg.norm(xy-np.array(other["xyz_m"][:2])) >= (
                    radius+other["radius_m"]+region["gripper_clearance_m"])


def test_copied_assets_are_complete_and_unchanged():
    manifest = json.loads((ROOT / "assets/source_manifest.json").read_text())
    for entry in manifest["files"]:
        target = ROOT / entry["target"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"]
    assert (ROOT / "assets/robots/franka_emika_panda/LICENSE").is_file()


def test_six_objects_are_free_stable_and_within_sampling_disks(simulation):
    config, model, data, _ = settled(simulation)
    layout = sample_layout(config, 42)
    assert np.isfinite(data.qpos).all()
    for obj in config["objects"]:
        joint = model.joint(f"{obj['name']}_joint")
        assert model.jnt_type[joint.id] == mujoco.mjtJoint.mjJNT_FREE
        low, high = body_bounds(model, data, obj["name"])
        assert -.002 < low[2] < .002
        assert high[2] < .11
        center = np.array(layout["objects"][obj["name"]]["xyz_m"][:2])
        # All primitive corners fit in the circle used for initial clearance.
        indices = np.flatnonzero(model.geom_bodyid == model.body(obj["name"]).id)
        signs = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
        for index in indices:
            points = model.geom_aabb[index, :3]+signs*model.geom_aabb[index, 3:]
            points = points @ data.geom_xmat[index].reshape(3, 3).T+data.geom_xpos[index]
            assert np.max(np.linalg.norm(points[:, :2]-center, axis=1)) <= obj["radius_m"]+.002
    assert evaluate(model, data, config)["sorted_count"] == 0


def test_camera_depth_roundtrip_and_object_visibility(simulation):
    config, model, data, _ = settled(simulation)
    with RGBDCamera(model, data, width=640, height=480) as camera:
        frame = camera.capture()
        assert frame.rgb.shape == (480, 640, 3)
        assert np.isfinite(frame.depth).all()
        assert np.all(frame.depth > 0)
        for obj in config["objects"]:
            point = data.xpos[model.body(obj["name"]).id]+[0, 0, .015]
            uv, depth = camera.calibration.project(point)
            assert 1 < uv[0] < 639 and 1 < uv[1] < 479
            assert np.allclose(camera.calibration.backproject(uv, depth), point, atol=1e-10)
        # Visibility is checked independently with renderer IDs, never fed to SAM3.
        camera.renderer.enable_segmentation_rendering()
        camera.renderer.update_scene(data, camera="eye_to_hand")
        segmentation = camera.renderer.render().copy()
        camera.renderer.disable_segmentation_rendering()
        for obj in config["objects"]:
            geoms = np.flatnonzero(model.geom_bodyid == model.body(obj["name"]).id)
            visible = np.isin(segmentation[..., 0], geoms) & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
            assert visible.sum() > 40, obj["name"]


@pytest.mark.parametrize("geom_group", [2, 3], ids=["rendering", "collision"])
def test_spoon_bowl_and_wrench_openings(simulation, geom_group):
    _, model = simulation
    data = mujoco.MjData(model)
    initialize(model, data)
    groups = np.zeros(6, dtype=np.uint8)
    groups[geom_group] = 1

    def ray_distance(name, x, y):
        body = model.body(name).id
        rotation = data.xmat[body].reshape(3, 3)
        origin = data.xpos[body] + rotation @ [x, y, .030]
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(model, data, origin, -rotation[:, 2], groups, False, -1, hit)
        if distance >= 0:
            assert model.geom_bodyid[hit[0]] == body
        return distance

    floor = ray_distance("spoon", .042, .001)
    wall = ray_distance("spoon", .066, .002)
    assert 0 < wall < floor < .030
    assert floor - wall > .004

    assert ray_distance("wrench", -.063, 0) == -1  # box-end bore
    assert ray_distance("wrench", .071, .003) == -1  # open jaw gap
    assert 0 < ray_distance("wrench", -.078, .001) < .030
    assert 0 < ray_distance("wrench", .060, .015) < .030


def test_evaluator_accepts_complete_stable_correct_trays(simulation):
    config, model, data, controller = settled(simulation)
    put_in_trays(config, model, data, controller)
    report = evaluate(model, data, config)
    assert report["success"], report
    assert report["sorted_count"] == 6


def test_evaluator_rejects_wrong_tray_and_hovering(simulation):
    config, model, data, controller = settled(simulation)
    put_in_trays(config, model, data, controller)
    joint = model.joint("spoon_joint")
    adr = joint.qposadr[0]
    data.qpos[adr:adr+3] = [*config["trays"]["tools"]["center_xy_m"], .009]
    mujoco.mj_forward(model, data)
    result = evaluate(model, data, config)
    assert not result["success"]
    assert not next(item for item in result["objects"] if item["name"] == "spoon")["inside_correct_tray"]
    data.qpos[adr:adr+3] = [*config["trays"]["kitchen"]["center_xy_m"], .20]
    mujoco.mj_forward(model, data)
    item = next(item for item in evaluate(model, data, config)["objects"] if item["name"] == "spoon")
    assert item["inside_correct_tray"] and not item["supported"] and not item["success"]


def test_evaluator_rejects_moving_object(simulation):
    config, model, data, controller = settled(simulation)
    put_in_trays(config, model, data, controller)
    data.qvel[model.joint("spoon_joint").dofadr[0]] = .2
    mujoco.mj_forward(model, data)
    item = next(item for item in evaluate(model, data, config)["objects"] if item["name"] == "spoon")
    assert not item["stable"] and not item["success"]
