"""Run the two perception stages on a saved tabletop RGB-D frame."""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabletop_tidy.camera import CameraCalibration, RGBDFrame
from tabletop_tidy.grasps import GraspGenX, panda_tcp_poses
from tabletop_tidy.perception import Sam3Segmenter
from tabletop_tidy.scene import load_config


def segment(args, config):
    with Image.open(args.scene / "rgb.png") as image:
        rgb = np.asarray(image.convert("RGB"))
    frame = RGBDFrame(rgb, np.load(args.scene / "depth_m.npy"))
    data = json.loads((args.scene / "calibration.json").read_text(encoding="utf-8"))
    calibration = CameraCalibration(data["width"], data["height"], np.asarray(data["K"]),
                                    np.asarray(data["T_base_camera"]), data["camera_name"], data["base_body"])
    model = Sam3Segmenter(args.checkpoint, config["sam3"]["confidence_threshold"])
    observations = model.detect(frame, calibration, config)
    args.output.mkdir(parents=True)
    objects = []
    for obj in config["objects"]:
        name = obj["name"]
        observation = observations.get(name)
        entry = {"name": name, "detected": observation is not None}
        if observation is not None:
            filename = f"{name}.npz"
            np.savez_compressed(args.output / filename, mask=observation.mask, points=observation.points)
            entry.update(prompt=observation.prompt, score=observation.score,
                         point_count=len(observation.points), file=filename)
        objects.append(entry)
    return {"stage": "sam3", "point_frame": calibration.base_body, "units": "meters",
            "checkpoint": str(args.checkpoint), "peak_cuda_bytes": model.torch.cuda.max_memory_allocated(),
            "objects": objects}


def grasp(args, config):
    import torch

    report = json.loads((args.segmentation / "sam3.json").read_text(encoding="utf-8"))
    torch.manual_seed(args.seed)
    model = GraspGenX(args.checkpoints, args.grippers, config["graspgenx"])
    tcp = ET.parse(ROOT / "assets/panda_scene.xml").find(".//site[@name='tcp']")
    hand_to_tcp = np.eye(4)
    hand_to_tcp[:3, 3] = np.fromstring(tcp.attrib["pos"], sep=" ")
    args.output.mkdir(parents=True)
    objects = []
    for obj in report["objects"]:
        name = obj["name"]
        entry = {"name": name, "candidate_count": 0}
        points = np.empty((0, 3))
        if obj["detected"]:
            with np.load(args.segmentation / obj["file"]) as observation:
                points = observation["points"]
        if len(points) >= model.min_points:
            poses, scores = model.generate(points)
            entry["candidate_count"] = len(poses)
            if len(poses):
                filename = f"{name}.npz"
                np.savez_compressed(args.output / filename, grasps=poses, scores=scores,
                                    tcp_poses=panda_tcp_poses(poses, hand_to_tcp))
                entry["file"] = filename
            else:
                entry["reason"] = "no_grasp_candidates"
        else:
            entry["reason"] = "insufficient_points" if obj["detected"] else "object_not_detected"
        objects.append(entry)
    return {"stage": "graspgenx", "pose_frame": report["point_frame"], "units": "meters",
            "seed": args.seed, "gripper": config["graspgenx"]["gripper_name"],
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "objects": objects}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    stages = parser.add_subparsers(dest="stage", required=True)

    segment_parser = stages.add_parser("segment", help="Convert a saved RGB-D capture into object point clouds")
    segment_parser.add_argument("--scene", type=Path, required=True)
    segment_parser.add_argument("--output", type=Path, required=True)
    segment_parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam3/sam3.pt")
    segment_parser.set_defaults(run=segment, report_name="sam3.json")

    grasp_parser = stages.add_parser("grasp", help="Generate grasp candidates from SAM3 point clouds")
    grasp_parser.add_argument("--segmentation", type=Path, required=True)
    grasp_parser.add_argument("--output", type=Path, required=True)
    grasp_parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints/graspgenx")
    grasp_parser.add_argument("--grippers", type=Path, default=ROOT / "checkpoints/gripper_descriptions")
    grasp_parser.add_argument("--seed", type=int, default=42)
    grasp_parser.set_defaults(run=grasp, report_name="graspgenx.json")

    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}")
    try:
        report = args.run(args, load_config(args.config))
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"{args.stage}: {error}\n")
    output = args.output / args.report_name
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
