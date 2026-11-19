"""RGB-D acquisition and exact simulator eye-to-hand camera calibration.

Pixel coordinates are ``(u, v)`` at pixel centers. Camera coordinates follow
OpenCV: x right, y down, z forward. Returned 3-D coordinates are in the Panda
``link0`` frame, in meters. MuJoCo's metric depth is axial z, not ray length.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    K: np.ndarray
    T_base_camera: np.ndarray
    camera_name: str = "eye_to_hand"
    base_body: str = "link0"

    @classmethod
    def from_mujoco(
        cls,
        model: Any,
        data: Any,
        width: int,
        height: int,
        camera_name: str = "eye_to_hand",
        base_body: str = "link0",
    ) -> "CameraCalibration":
        """Read ideal intrinsics/extrinsics after ``mj_forward``.

        This is ground-truth *camera* calibration supplied by the simulator,
        not a noisy image-based calibration or an object pose lookup.
        The scene uses a perspective camera specified by vertical ``fovy``.
        """
        import mujoco

        if width <= 0 or height <= 0:
            raise ValueError("Camera image dimensions must be positive")
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        if camera_id < 0 or base_id < 0:
            raise ValueError(f"Missing camera {camera_name!r} or base body {base_body!r}")
        if hasattr(model, "cam_orthographic") and model.cam_orthographic[camera_id]:
            raise ValueError("This calibration requires a perspective camera")
        if np.any(model.cam_sensorsize[camera_id] > 0):
            raise ValueError("Use the scene's fovy camera; physical sensor intrinsics are unsupported")
        focal = 0.5 * height / np.tan(np.deg2rad(float(model.cam_fovy[camera_id])) / 2)
        K = np.array([[focal, 0, (width - 1) / 2],
                      [0, focal, (height - 1) / 2], [0, 0, 1]], dtype=float)
        # MuJoCo local camera: +x right, +y up, viewing direction -z.
        R_world_camera = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
        R_world_base = data.xmat[base_id].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R_world_base.T @ R_world_camera
        T[:3, 3] = R_world_base.T @ (data.cam_xpos[camera_id] - data.xpos[base_id])
        return cls(width, height, K, T, camera_name, base_body)

    def backproject(self, pixels_uv: np.ndarray, depth: np.ndarray | float) -> np.ndarray:
        """Convert pixel centers and their metric z-depth to base-frame points."""
        uv = np.asarray(pixels_uv, dtype=float)
        if uv.shape[-1:] != (2,):
            raise ValueError("pixels_uv must have shape (..., 2)")
        z = np.broadcast_to(np.asarray(depth, dtype=float), uv.shape[:-1])
        if not np.all(np.isfinite(uv)) or not np.all(np.isfinite(z) & (z > 0)):
            raise ValueError("Pixels must be finite and depth must be finite and positive")
        camera_points = np.stack(((uv[..., 0] - self.K[0, 2]) * z / self.K[0, 0],
                                  (uv[..., 1] - self.K[1, 2]) * z / self.K[1, 1], z), axis=-1)
        return camera_points @ self.T_base_camera[:3, :3].T + self.T_base_camera[:3, 3]

    def project(self, points_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return projected ``(u, v)`` and camera-axis depth, including off-image pixels."""
        points = np.asarray(points_base, dtype=float)
        if points.shape[-1:] != (3,) or not np.all(np.isfinite(points)):
            raise ValueError("points_base must contain finite 3-D points")
        camera_points = (points - self.T_base_camera[:3, 3]) @ self.T_base_camera[:3, :3]
        z = camera_points[..., 2]
        if np.any(z <= 0):
            raise ValueError("Cannot project points behind the camera")
        uv = np.stack((camera_points[..., 0] / z * self.K[0, 0] + self.K[0, 2],
                       camera_points[..., 1] / z * self.K[1, 1] + self.K[1, 2]), axis=-1)
        return uv, z

    def alignment_report(self) -> dict[str, float]:
        """Check rotation and viewing direction relative to the robot base/table."""
        rotation = self.T_base_camera[:3, :3]
        downward = np.array([0., 0., -1.])
        angle = np.degrees(np.arccos(np.clip(rotation[:, 2] @ downward, -1., 1.)))
        return {"rotation_orthogonality_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
                "rotation_determinant": float(np.linalg.det(rotation)),
                "angle_from_table_normal_degrees": float(angle)}

    def as_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "K": self.K.tolist(),
                "T_base_camera": self.T_base_camera.tolist(), "camera_name": self.camera_name,
                "base_body": self.base_body, "camera_convention": "OpenCV x-right y-down z-forward",
                "depth_convention": "metric axial z", "units": "meters",
                "calibration_source": "exact MuJoCo camera and robot base transforms",
                "alignment": self.alignment_report()}

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


class RGBDCamera:
    """A context-managed offscreen renderer for the fixed overhead camera."""

    def __init__(self, model: Any, data: Any, width: int = 640, height: int = 480,
                 camera_name: str = "eye_to_hand", base_body: str = "link0") -> None:
        import mujoco

        self.model, self.data = model, data
        self.camera_name = camera_name
        mujoco.mj_forward(model, data)
        self.calibration = CameraCalibration.from_mujoco(model, data, width, height, camera_name, base_body)
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def capture(self) -> RGBDFrame:
        self.renderer.update_scene(self.data, camera=self.camera_name)
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        return RGBDFrame(rgb=rgb, depth=depth)

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> "RGBDCamera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
