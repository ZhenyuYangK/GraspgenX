"""GraspGenX inference and Franka Panda frame conversion."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def panda_tcp_poses(grasp_poses: np.ndarray, hand_to_tcp: np.ndarray) -> np.ndarray:
    """Map GraspGenX grasp poses to the Panda TCP in the same reference frame."""
    # 右乘局部Z轴的90°旋转以对齐Panda双指，再将手掌到TCP的偏移带入当前抓取方向。
    grasp_to_hand = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.],
                              [0., 0., 1., 0.], [0., 0., 0., 1.]])
    return grasp_poses @ grasp_to_hand @ hand_to_tcp


class GraspGenX:
    def __init__(self, checkpoints: Path, grippers: Path, config: dict):
        if not all((checkpoints / "release" / kind / "config.yaml").is_file()
                   for kind in ("gen", "dis")):
            raise FileNotFoundError(f"GraspGenX release configs are missing under {checkpoints}")
        gripper = config["gripper_name"]
        assets = grippers / "gripper_descriptions/assets/x_grippers"
        if not (assets / gripper / "config.json").is_file():
            raise FileNotFoundError(f"GraspGenX gripper {gripper!r} is missing under {assets}")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GraspGenX inference requires access to a CUDA device")
        # GraspGenX在导入时读取路径环境变量，因此先设置权重和夹爪描述的位置。
        os.environ["GRASPGENX_CHECKPOINT_DIR"] = str(checkpoints.resolve())
        os.environ["GRASPGENX_GRIPPER_CFG_DIR"] = str(grippers.resolve())
        from graspgenx._constants import MIN_OBJECT_POINT_COUNT
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.samplers.graspmoe import run_graspmoe
        from graspgenx.utils.checkpoint_io import load_model_cfg

        release = checkpoints / "release"
        model = load_model_cfg(str(release / "gen"), str(release / "dis"))
        self.min_points = MIN_OBJECT_POINT_COUNT
        self.num_grasps = config["num_grasps"]
        self.sampler = GraspGenXSampler(model, gripper, diffusion_grasps=self.num_grasps)
        self.grasp_moe = run_graspmoe
        self.torch = torch

    def generate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Expected finite XYZ points with shape (N, 3)")
        if len(points) < self.min_points:
            raise ValueError(f"GraspGenX requires at least {self.min_points} points, received {len(points)}")
        poses, scores = self.sampler.run_inference(points, self.sampler, num_grasps=self.num_grasps,
                                                   topk_num_grasps=self.num_grasps)
        poses = poses.cpu().numpy().reshape(-1, 4, 4)
        scores = scores.cpu().numpy()
        # 位姿和分数一起过滤、排序，保持每个位姿对应原来的模型分数。
        keep = np.isfinite(poses).all(axis=(1, 2)) & np.isfinite(scores)
        order = np.argsort(-scores[keep], kind="stable")[:self.num_grasps]
        poses, scores = poses[keep][order], scores[keep][order]
        if np.max(points[:, 2]) < .045:
            # 为贴近桌面的物品补充有模型评分的俯抓候选，结束后恢复Torch随机状态。
            with self.torch.random.fork_rng(devices=[self.torch.cuda.current_device()]):
                result = self.grasp_moe(points, self.sampler, num_grasps=48, obb_density="dense",
                                        z_offsets_cm=(-1.5, -.8, 0.))
            extra_poses, extra_scores = result["grasps_obb"], result["scores_obb"]
            keep = np.isfinite(extra_poses).all(axis=(1, 2)) & np.isfinite(extra_scores)
            order = np.argsort(-extra_scores[keep], kind="stable")[:self.num_grasps // 3]
            poses = np.concatenate([poses, extra_poses[keep][order]])
            scores = np.concatenate([scores, extra_scores[keep][order]])
        order = np.argsort(-scores, kind="stable")
        return poses[order], scores[order]
