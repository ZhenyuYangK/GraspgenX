import os
# 在导入MuJoCo之前设定默认渲染后端，测试可在无图形桌面的环境运行。
os.environ.setdefault("MUJOCO_GL", "egl")

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from tabletop_tidy.scene import (
    BOX_FLOOR_HEIGHT, ROOT, body_bounds, build_scene, evaluate, initialize, load_config, sample_layout,
)
from tabletop_tidy.control import PandaController
from tabletop_tidy.camera import RGBDCamera
from tabletop_tidy.sorting import ContactCheck, operate_lid, park_arm, tune_gripper


@pytest.fixture(scope="module")
def simulation(tmp_path_factory):
    # 同一测试模块复用种子42的模型，减少重复生成和编译场景的开销。
    config = load_config()
    path = build_scene(config, sample_layout(config, 42), tmp_path_factory.mktemp("scene") / "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    return config, model


def settled(simulation):
    config, model = simulation
    # 每次分配独立的仿真状态并等待物品落稳，避免测试之间共享上一次的物体位置。
    data = mujoco.MjData(model)
    initialize(model, data)
    controller = PandaController(model, data)
    controller.hold(1.5)
    return config, model, data, controller


def put_in_trays(config, model, data, controller):
    # 直接布置验收用场景。同一盒的两件物品分前后放，姿态设为单位四元数并清零速度。
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
    # 对50个种子检查整个包围圆都在区域内，并验证任意一对物品的夹爪间隙。
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
    # sha256用于核对项目内副本；已加注释的副本另用source_sha256记录来源文件的哈希。
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
        # 用各几何体包围盒的角点验证配置的包围圆确实覆盖了物体。
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
        # 用每件物品附近的三维点检查视野和投影到像素再反投影的精度。
        for obj in config["objects"]:
            point = data.xpos[model.body(obj["name"]).id]+[0, 0, .015]
            uv, depth = camera.calibration.project(point)
            assert 1 < uv[0] < 639 and 1 < uv[1] < 479
            assert np.allclose(camera.calibration.backproject(uv, depth), point, atol=1e-10)
        # 旁路读取渲染器的几何体ID来统计可见像素，独立检查每件物品是否入镜。
        camera.renderer.enable_segmentation_rendering()
        camera.renderer.update_scene(data, camera="eye_to_hand")
        segmentation = camera.renderer.render().copy()
        camera.renderer.disable_segmentation_rendering()
        for obj in config["objects"]:
            geoms = np.flatnonzero(model.geom_bodyid == model.body(obj["name"]).id)
            visible = np.isin(segmentation[..., 0], geoms) & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
            assert visible.sum() > 40, obj["name"]


@pytest.fixture(params=[2, 3], ids=["rendering", "collision"])
def object_ray(simulation, request):
    # 每组射线断言分别对第2组外观网格和第3组碰撞几何体运行。
    _, model = simulation
    data = mujoco.MjData(model)
    initialize(model, data)
    groups = np.zeros(6, dtype=np.uint8)
    groups[request.param] = 1

    def ray_distance(name, local_origin, local_direction):
        body = model.body(name).id
        # 在物体局部坐标定义射线，再变换到世界坐标，便于直接按原型尺寸选取探测点。
        rotation = data.xmat[body].reshape(3, 3)
        origin = data.xpos[body] + rotation @ local_origin
        direction = rotation @ local_direction
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(model, data, origin, direction, groups, False, -1, hit)
        if distance >= 0:
            assert model.geom_bodyid[hit[0]] == body
        return distance

    return ray_distance


def test_spoon_bowl_and_wrench_openings(object_ray):
    def ray_distance(name, x, y):
        return object_ray(name, [x, y, .030], [0, 0, -1])

    floor = ray_distance("spoon", .042, .001)
    wall = ray_distance("spoon", .066, .002)
    # 从同一高度向下发射，勺碗中心的命中距离应大于侧壁。
    assert 0 < wall < floor < .030
    assert floor - wall > .004

    assert ray_distance("wrench", -.063, 0) == -1  # 梅花端通孔
    assert ray_distance("wrench", .071, .003) == -1  # 开口端缺口
    assert 0 < ray_distance("wrench", -.078, .001) < .030
    assert 0 < ray_distance("wrench", .060, .015) < .030


def test_cup_cavity_handle_and_stapler_gap(object_ray):
    # 竖直射线量杯壁高度，横向射线分别穿过把手孔洞和订书机的进纸间隙。
    down, across = [0, 0, -1], [0, 1, 0]
    floor = object_ray("cup", [0, 0, .095], down)
    rim = object_ray("cup", [.035, .002, .095], down)
    assert 0 < rim < .02
    assert .065 < floor - rim < .085
    assert object_ray("cup", [.051, -.020, .044], across) == -1
    assert 0 < object_ray("cup", [.066, -.020, .044], across) < .020

    assert object_ray("stapler", [.039, -.025, .020], across) == -1
    assert 0 < object_ray("stapler", [.039, -.025, .032], across) < .025
    assert 0 < object_ray("stapler", [.039, 0, .020], down) < .010


def test_screwdriver_tapered_blade_and_handle(object_ray):
    down = [0, 0, -1]
    handle = object_ray("screwdriver", [-.050, 0, .040], down)
    shaft = object_ray("screwdriver", [.040, 0, .040], down)
    tip = object_ray("screwdriver", [.079, 0, .040], down)
    assert 0 < handle < shaft < tip < .040
    assert shaft - handle > .009
    assert tip - shaft > .002
    # 相同高度的横向射线能打到圆杆，但能从更薄的刀尖上方通过。
    assert object_ray("screwdriver", [.079, -.025, .0173], [0, 1, 0]) == -1
    assert 0 < object_ray("screwdriver", [.040, -.025, .0173], [0, 1, 0]) < .025


def test_tape_bore_reel_and_feed_clearance(object_ray):
    across = [0, 1, 0]
    assert object_ray("tape_dispenser", [-.022, -.045, .047], across) == -1
    assert 0 < object_ray("tape_dispenser", [-.0095, -.045, .047], across) < .030
    assert 0 < object_ray("tape_dispenser", [-.022, -.045, .068], across) < .030
    # 从上方应打到拉出的胶带，胶带下方的空隙允许侧向射线穿过。
    feed = object_ray("tape_dispenser", [.023, 0, .085], [0, 0, -1])
    assert .025 < feed < .040
    assert object_ray("tape_dispenser", [.018, -.045, .028], across) == -1
    assert 0 < object_ray("tape_dispenser", [.049, -.045, .029], across) < .045


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
    # 先制造放错盒，再让物体悬在正确盒子的正上方，分别检验水平范围和盒底支撑条件。
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
    # 保持入盒位置，单独给勺子线速度，验证验收还要求物品静稳。
    data.qvel[model.joint("spoon_joint").dofadr[0]] = .2
    mujoco.mj_forward(model, data)
    item = next(item for item in evaluate(model, data, config)["objects"] if item["name"] == "spoon")
    assert not item["stable"] and not item["success"]


def test_storage_box_lids_open_independently_and_close_over_contents(simulation):
    config, model = simulation
    data = mujoco.MjData(model)
    initialize(model, data, lids_open=False)
    controller = PandaController(model, data)
    names = [f"storage_box_lid_{category}" for category in config["trays"]]
    angles = np.array([model.joint(name).qposadr[0] for name in names])
    controls = np.array([model.actuator(name).id for name in names])
    np.testing.assert_allclose(data.qpos[angles], 0)
    kitchen_lid = model.body("storage_box_lid_kitchen").id
    kitchen_box = model.body("storage_box_kitchen").id
    x, y = config["trays"]["kitchen"]["center_xy_m"]
    hit = np.empty(1, dtype=np.int32)
    # 从盒子正上方发射同一条射线，检查合盖时先碰到盖，开盖后能碰到盒底。
    mujoco.mj_ray(model, data, [x, y, .35], [0, 0, -1], None, True, -1, hit)
    assert hit[0] >= 0 and model.geom_bodyid[hit[0]] == kitchen_lid

    opened = model.actuator_ctrlrange[controls, 1]
    data.ctrl[controls[0]] = opened[0]
    controller.hold(1)
    np.testing.assert_allclose(data.qpos[angles], [opened[0], 0, 0], atol=.02)
    mujoco.mj_ray(model, data, [x, y, .35], [0, 0, -1], None, True, -1, hit)
    assert hit[0] >= 0 and model.geom_bodyid[hit[0]] == kitchen_box

    data.ctrl[controls] = opened
    controller.hold(1)
    np.testing.assert_allclose(data.qpos[angles], opened, atol=.02)
    put_in_trays(config, model, data, controller)
    data.ctrl[controls] = 0
    controller.hold(1)
    np.testing.assert_allclose(data.qpos[angles], 0, atol=.02)
    assert evaluate(model, data, config)["sorted_count"] == 6


def test_unpowered_storage_box_lids_settle_at_the_hinge_limits(tmp_path):
    config = load_config()
    scene = build_scene(config, sample_layout(config, 42), tmp_path / "passive.xml", powered_lids=False)
    model = mujoco.MjModel.from_xml_path(str(scene))
    # 确认没有盒盖执行器，然后验证打开和闭合两种初始角度都能稳定在各自限位。
    names = [f"storage_box_lid_{category}" for category in config["trays"]]
    assert all(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) == -1 for name in names)

    for lids_open in (False, True):
        data = mujoco.MjData(model)
        initialize(model, data, lids_open=lids_open)
        PandaController(model, data).hold(1.5)
        for name in names:
            joint = model.joint(name)
            np.testing.assert_allclose(data.qpos[joint.qposadr], joint.range[int(lids_open)], atol=.02)


def test_panda_opens_and_closes_all_storage_boxes_by_their_handles(tmp_path):
    config = load_config()
    scene = build_scene(config, sample_layout(config, 42), tmp_path / "passive.xml", powered_lids=False)
    model = mujoco.MjModel.from_xml_path(str(scene))
    tune_gripper(model)
    data = mujoco.MjData(model)
    initialize(model, data, lids_open=False)
    arm = PandaController(model, data)
    arm.hold(1.5)
    names = {category: f"storage_box_lid_{category}" for category in config["trays"]}
    joints = [model.joint(name) for name in names.values()]
    angle_indices = [joint.qposadr[0] for joint in joints]

    for opening in (True, False):
        if not opening:
            # 合盖前预先放好物品，以便同时检查机械臂开合和盒内物品是否仍通过验收。
            park_arm(arm)
            put_in_trays(config, model, data, arm)
            assert evaluate(model, data, config)["success"]
        for category, name in names.items():
            # 测试在每个仿真步检查接触，本次操作允许双指接触目标盒盖。
            arm.on_step = ContactCheck(model, data, name).check
            operate_lid(arm, category, opening)
        arm.on_step = None
        np.testing.assert_allclose(data.qpos[angle_indices],
                                   [joint.range[int(opening)] for joint in joints], atol=np.deg2rad(3))

    assert evaluate(model, data, config)["success"]


def test_evaluator_rejects_object_protruding_above_box(simulation):
    config, model = simulation
    data = mujoco.MjData(model)
    initialize(model, data)
    wrench = model.joint("wrench_joint").qposadr[0]
    x, y = config["trays"]["tools"]["center_xy_m"]
    # 竖起扳手后按几何包围范围调整高度，让底端贴着盒底而顶端露出盒沿。
    data.qpos[wrench:wrench+7] = [x, y, 0, np.sqrt(.5), 0, np.sqrt(.5), 0]
    mujoco.mj_forward(model, data)
    low, _ = body_bounds(model, data, "wrench", collision_only=False)
    data.qpos[wrench+2] = BOX_FLOOR_HEIGHT - low[2]
    mujoco.mj_forward(model, data)
    item = next(item for item in evaluate(model, data, config)["objects"] if item["name"] == "wrench")
    assert item["inside_correct_tray"] and item["supported"] and item["stable"]
    assert not item["fits_below_rim"] and not item["success"]
