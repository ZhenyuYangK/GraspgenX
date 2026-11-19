"""Scene preparation and diagnostic entry point.

Scene preview is intentionally distinct from the future SAM3/GraspGenX task
runner; it never reports a successful tidying task.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from .scene import ROOT, build_scene, evaluate, initialize, load_config, sample_layout


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare and inspect the six-object Panda scene")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/scene")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--check-sam3", action="store_true", help="Inspect local SAM3 checkpoint files and exit")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.checkpoint is not None and not args.check_sam3:
        parser.error("--checkpoint requires --check-sam3; this entry point currently previews the scene")
    if args.check_sam3:
        from .weights import inspect_checkpoint, scan_checkpoints
        files = [inspect_checkpoint(args.checkpoint)] if args.checkpoint else scan_checkpoints(ROOT.parent)
        report = {"mode": "checkpoint_structure_check", "model_loaded": False,
                  "inference_run": False, "files": files}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if any(item["is_sam3_candidate"] for item in files) else 2
    # Import rendering-related modules after selecting the GL backend.
    if not args.viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import cv2
    from .camera import RGBDCamera
    from .control import PandaController

    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    layout = sample_layout(config, args.seed)
    scene_path = build_scene(config, layout, args.output / "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    initialize(model, data)
    controller = PandaController(model, data)
    controller.hold(1.5)
    import numpy as np
    with RGBDCamera(model, data, **config["camera"]) as camera:
        frame = camera.capture()
        cv2.imwrite(str(args.output / "rgb.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        np.save(args.output / "depth_m.npy", frame.depth)
        camera.calibration.save(args.output / "calibration.json")
    with RGBDCamera(model, data, width=960, height=720, camera_name="overview") as camera:
        frame = camera.capture()
        cv2.imwrite(str(args.output / "overview.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
    report = {"mode": "scene_preview", "task_executed": False,
              "layout": layout, "initial_evaluation": evaluate(model, data, config)}
    for name, value in [("layout.json", layout), ("scene_report.json", report)]:
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": "scene_preview", "seed": args.seed,
                      "output": str(args.output), "task_executed": False}, ensure_ascii=False))
    if args.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [.43, -.03, .15]
            viewer.cam.distance = 1.55
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -35
            while viewer.is_running():
                start = time.monotonic()
                controller.step()
                viewer.sync()
                time.sleep(max(0, model.opt.timestep-(time.monotonic()-start)))
    return 0
