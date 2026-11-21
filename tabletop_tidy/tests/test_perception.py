import numpy as np
import pytest

from tabletop_tidy.camera import CameraCalibration
from tabletop_tidy.grasps import panda_tcp_poses
from tabletop_tidy.perception import ObjectObservation, mask_to_points, match_observations


def test_mask_depth_uses_pixel_centers_and_filters_invalid_workspace_points():
    # 用可手算的标定构造小图，同时包含掩膜外、桌面、NaN和工作区外的像素。
    camera_to_base = np.array([[1., 0., 0., .25], [0., -1., 0., .1],
                               [0., 0., -1., 1.], [0., 0., 0., 1.]])
    intrinsics = np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 1.]])
    calibration = CameraCalibration(3, 2, intrinsics, camera_to_base)
    depth = np.array([[.8, .8, 1.], [.8, np.nan, .8]])
    mask = np.array([[False, True, True], [True, True, True]])
    layout = {"x_m": [0., .9], "y_m": [-.5, .5]}

    points = mask_to_points(mask, depth, calibration, layout)

    np.testing.assert_allclose(points, [[.65, .1, .2], [.25, -.3, .2]])


def test_panda_tcp_offset_follows_the_grasp_pose_orientation():
    # 抓取的局部Z轴指向基座X轴，所以沿手掌Z轴的偏移应加在基座X坐标上。
    grasp = np.array([[0., 0., 1., .5], [0., 1., 0., .2],
                      [-1., 0., 0., .3], [0., 0., 0., 1.]])
    hand_to_tcp = np.eye(4)
    hand_to_tcp[2, 3] = .1034

    tcp, = panda_tcp_poses(grasp[None], hand_to_tcp)

    np.testing.assert_allclose(tcp[:3, 3], [.6034, .2, .3])
    # 同时检查双指坐标轴经局部Z轴旋转后与抓取X轴反向对齐。
    np.testing.assert_allclose(tcp[:3, 1], -grasp[:3, 0])


def test_competing_masks_are_assigned_to_distinct_objects():
    pytest.importorskip("scipy")
    stapler = np.array([[True, False]])
    dispenser = np.array([[False, True]])
    points = np.ones((1, 3))
    # 最高分的订书机候选其实占用了胶带座的掩膜；选次高分才能让两件物品都匹配。
    candidates = [
        ObjectObservation("stapler", "stapler", .69, dispenser, points),
        ObjectObservation("tape_dispenser", "desk tape dispenser", .63, dispenser, points),
        ObjectObservation("stapler", "stapler", .52, stapler, points),
    ]

    result = match_observations(candidates, ["stapler", "tape_dispenser"])

    assert result["stapler"] is candidates[2]
    assert result["tape_dispenser"] is candidates[1]
