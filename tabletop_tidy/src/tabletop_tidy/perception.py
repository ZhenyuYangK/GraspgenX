"""Text-prompted SAM3 segmentation and metric object point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import CameraCalibration, RGBDFrame


@dataclass(frozen=True)
class ObjectObservation:
    # 保留命中的提示词、置信度、二维掩膜和对应的基座坐标系点云。
    name: str
    prompt: str
    score: float
    mask: np.ndarray
    points: np.ndarray


def mask_to_points(mask: np.ndarray, depth: np.ndarray, calibration: CameraCalibration,
                   layout: dict) -> np.ndarray:
    """Backproject valid masked depth within the tabletop pickup area."""
    if mask.shape != depth.shape or depth.shape != (calibration.height, calibration.width):
        raise ValueError("Mask, depth, and calibration dimensions must match")
    rows, cols = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    # 数组索引按行、列排列，反投影接口按(u, v)即列、行接收像素。
    points = calibration.backproject(np.column_stack((cols, rows)), depth[rows, cols])
    xmin, xmax = layout["x_m"]
    ymin, ymax = layout["y_m"]
    x, y, z = points.T
    # 桌面是基座坐标系的Z=0，保留1毫米余量以滤掉深度误差带来的桌面点。
    inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) & (z > .001)
    return points[inside].astype(np.float32)


def match_observations(candidates: list[ObjectObservation],
                       names: list[str]) -> dict[str, ObjectObservation]:
    """Assign at most one configured name to each observed instance."""
    from scipy.optimize import linear_sum_assignment

    instances: list[dict[str, ObjectObservation]] = []
    # 按分数从高到低聚类掩膜；交并比超过0.5视为同一实例，每个名称保留最高分。
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        for instance in instances:
            other = next(iter(instance.values())).mask
            intersection = np.count_nonzero(candidate.mask & other)
            if intersection > .5 * np.count_nonzero(candidate.mask | other):
                instance.setdefault(candidate.name, candidate)
                break
        else:
            instances.append({candidate.name: candidate})

    # 行是配置名称，列是真实实例和未检测占位。有效匹配加上名称总数的权重，优先多匹配名称。
    # 匹配件数相同时，再比较置信度总和；每个实例至多分配给一个名称。
    weights = np.zeros((len(names), len(instances) + len(names)))
    for row, name in enumerate(names):
        for column, instance in enumerate(instances):
            if name in instance:
                weights[row, column] = len(names) + instance[name].score
    rows, columns = linear_sum_assignment(weights, maximize=True)
    return {names[row]: instances[column][names[row]] for row, column in zip(rows, columns)
            if weights[row, column] > 0}


class Sam3Segmenter:
    def __init__(self, checkpoint: Path, confidence_threshold: float):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint does not exist: {checkpoint}")
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 inference requires access to a CUDA device")
        self.torch = torch
        model = build_sam3_image_model(device="cuda", load_from_HF=False)
        # 内存映射本地归档，只取detector前缀下的张量加载到GPU模型，降低CPU内存峰值。
        archive = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=True)
        weights = archive.get("model", archive)
        detector = {name.removeprefix("detector."): tensor for name, tensor in weights.items()
                    if name.startswith("detector.")}
        missing, _ = model.load_state_dict(detector, strict=False)
        if missing:
            raise RuntimeError(f"SAM3 detector weights missing: {missing}")
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

    def detect(self, frame: RGBDFrame, calibration: CameraCalibration,
               config: dict) -> dict[str, ObjectObservation]:
        from PIL import Image

        if frame.rgb.shape != (*frame.depth.shape, 3):
            raise ValueError("Expected aligned RGB and depth images")
        candidates = []
        with self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16):
            # 对每帧只设置一次图像，再用配置中的多种提示词查询同一图像状态。
            state = self.processor.set_image(Image.fromarray(frame.rgb))
            for obj in config["objects"]:
                for prompt in obj["prompts"]:
                    result = self.processor.set_text_prompt(prompt, state)
                    masks = result["masks"].cpu().numpy()[:, 0]
                    scores = result["scores"].float().cpu().numpy()
                    for mask, score in zip(masks, scores):
                        cloud = mask_to_points(mask, frame.depth, calibration, config["layout"])
                        if len(cloud) and np.isfinite(score):
                            candidates.append(ObjectObservation(obj["name"], prompt, float(score), mask, cloud))

        return match_observations(candidates, [obj["name"] for obj in config["objects"]])
