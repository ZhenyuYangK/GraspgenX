"""Run SAM3 and GraspGenX and watch the Panda sort the tabletop in MuJoCo."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
import time

import mujoco as mj
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabletop_tidy.camera import RGBDCamera
from tabletop_tidy.control import PandaController
from tabletop_tidy.scene import build_scene, evaluate, initialize, load_config, sample_layout
from tabletop_tidy.sorting import ContactCheck, operate_lid, park_arm, plan_transfer, transfer_object, tune_gripper


class ViewerClosed(Exception):
    pass


class LiveView:
    # 窗口使用单独的MjData，按约30帧每秒从仿真状态同步位置、速度和时间。
    def __init__(self, viewer, model, display, data):
        self.viewer, self.model, self.display, self.data = viewer, model, display, data
        self.next_frame = 0.
        self.last_update = 0.
        if viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [.43, -.03, .15]
                viewer.cam.distance = 1.40
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -37

    def step(self):
        if not self.viewer or self.data.time < self.next_frame:
            return
        if not self.viewer.is_running():
            raise ViewerClosed
        delay = self.last_update + 1 / 30 - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        with self.viewer.lock():
            self.display.qpos[:] = self.data.qpos
            self.display.qvel[:] = self.data.qvel
            self.display.time = self.data.time
            mj.mj_forward(self.model, self.display)
        self.viewer.sync()
        self.last_update = time.monotonic()
        self.next_frame += 1 / 30

    def idle(self):
        # 推理和规划等耗时阶段也让窗口处理刷新，并及时响应关闭操作。
        if not self.viewer:
            return
        if not self.viewer.is_running():
            raise ViewerClosed
        if time.monotonic() - self.last_update >= 1 / 30:
            self.viewer.sync()
            self.last_update = time.monotonic()

    def announce(self, stage, detail):
        print(f"[{stage}] {detail}", flush=True)
        if self.viewer and hasattr(self.viewer, "set_texts"):
            self.viewer.set_texts((None, None, stage, detail))


def inference_worker(python, worker_args, idle):
    # 用推理环境的Python启动独立进程，将标准输出和错误合并，失败时读出日志。
    worker = ROOT / "scripts/infer_scene.py"
    with TemporaryFile() as log:
        with subprocess.Popen([str(python), str(worker), *map(str, worker_args)],
                              stdout=log, stderr=subprocess.STDOUT, cwd=ROOT) as process:
            try:
                while process.poll() is None:
                    idle()
                    time.sleep(.05)
            finally:
                # 等待期间如果关闭窗口，结束仍在运行的推理进程。
                if process.poll() is None:
                    process.terminate()
            if process.returncode != 0:
                log.seek(0)
                detail = log.read().decode(errors="replace").strip()
                raise RuntimeError(f"Inference worker exited with status {process.returncode}\n{detail}")


def observe(model, data, config, args, view, report):
    # 两个推理子进程通过临时文件交换数据，退出时随目录清理。
    with TemporaryDirectory(prefix="tabletop-tidy-") as directory:
        work = Path(directory)
        frame_dir = work / "frame"
        frame_dir.mkdir()
        with RGBDCamera(model, data, **config["camera"]) as camera:
            frame = camera.capture()
            Image.fromarray(frame.rgb).save(frame_dir / "rgb.png")
            np.save(frame_dir / "depth_m.npy", frame.depth)
            camera.calibration.save(frame_dir / "calibration.json")

        config_args = ["--config", args.config] if args.config else []
        segmentation, grasps = work / "segmentation", work / "grasps"
        view.announce("SAM3", "Segmenting the live RGB-D frame")
        inference_worker(args.inference_python, [*config_args, "segment", "--scene", frame_dir,
                         "--output", segmentation], view.idle)
        view.announce("GraspGenX", "Generating Panda grasp candidates")
        inference_worker(args.inference_python, [*config_args, "grasp", "--segmentation", segmentation,
                         "--output", grasps, "--seed", args.seed], view.idle)

        segmentation_report = json.loads((segmentation / "sam3.json").read_text())
        grasp_report = json.loads((grasps / "graspgenx.json").read_text())
        # 持久报告保留推理摘要；候选位姿、分数和点云在临时文件清理前读入内存。
        report.append({stage["stage"]: {**stage, "objects": [
            {key: value for key, value in obj.items() if key != "file"} for obj in stage["objects"]]}
            for stage in (segmentation_report, grasp_report)})
        candidates = {}
        for item in grasp_report["objects"]:
            if not item["candidate_count"]:
                continue
            name = item["name"]
            with np.load(grasps / item["file"]) as result, np.load(segmentation / f"{name}.npz") as obs:
                candidates[name] = (result["tcp_poses"], result["scores"], obs["points"])
        return candidates


def run_task(arm, config, args, view, report):
    # 先打开收纳盒，再采集场景和逐件整理，物品验收通过后才合盖。
    model, data = arm.model, arm.data
    settle = config["evaluation"]["settle_seconds"]
    arm.hold(settle)
    try:
        for category in config["trays"]:
            view.announce("Panda", f"Opening {category} box")
            angle = operate_lid(arm, category, opening=True)
            report["lids"].append({"category": category, "action": "open", "angle_degrees": angle})
    except RuntimeError as error:
        report["lids"].append({"category": category, "action": "open", "error": str(error)})
        raise

    # 采集RGB-D前将机械臂停回初始关节姿态，减小对固定相机视野的遮挡。
    park_arm(arm)
    candidates = observe(model, data, config, args, view, report["inference"])
    lanes = dict.fromkeys(config["trays"], -1)

    def object_height(obj):
        # 用点云高度的第95百分位排序，缺少候选的物品排到最后。
        observed = candidates.get(obj["name"])
        return float(np.percentile(observed[2][:, 2], 95)) if observed else float("inf")

    # 每个盒子先用前方槽位放较矮的物品，为贴近盒沿下降的手腕留出空间。
    for obj in sorted(config["objects"], key=object_height):
        name, category = obj["name"], obj["category"]
        result = {"name": name, "category": category, "attempts": []}
        report["objects"].append(result)
        for attempt in range(1, config["execution"]["max_attempts_per_object"] + 1):
            view.announce("Panda", f"Sorting {name} into {category} ({attempt})")
            entry = {"attempt": attempt}
            result["attempts"].append(entry)
            if name in candidates:
                plan, failures = plan_transfer(arm, config, obj, lanes[category], *candidates[name], view.idle)
            else:
                plan, failures = None, {"No GraspGenX candidates from this RGB-D observation": 1}
            entry["rejected_candidates"] = failures
            if plan is None:
                entry["error"] = "No feasible grasp and placement"
                # 候选只在仿真副本中试跑。缺少目标候选时，重拍才可能提供新的输入。
                if name not in candidates and attempt < config["execution"]["max_attempts_per_object"]:
                    candidates.update(observe(model, data, config, args, view, report["inference"]))
                    continue
                break
            entry.update(candidate=plan.candidate, score=plan.score)
            contact = ContactCheck(model, data, name)

            def move_step():
                # 执行已选方案时，每个物理步都处理画面更新和接触检查。
                view.step()
                contact.check()

            arm.on_step = move_step
            try:
                transfer_object(arm, name, plan, settle)
            except RuntimeError as error:
                entry["error"] = str(error)
            finally:
                arm.on_step = view.step
            result["evaluation"] = next(item for item in evaluate(model, data, config)["objects"]
                                        if item["name"] == name)
            if result["evaluation"]["success"]:
                lanes[category] = 1
                break
            # 实际抓放可能改变物品姿态，先松手抬高，再为下一次尝试重新观测。
            arm.open()
            retreat = arm.tcp_position()
            retreat[2] = max(.23, retreat[2])
            arm.move(retreat, 1.0, "recovery retreat")
            if attempt < config["execution"]["max_attempts_per_object"]:
                park_arm(arm)
                candidates.update(observe(model, data, config, args, view, report["inference"]))

    report["before_closing"] = evaluate(model, data, config)
    if not report["before_closing"]["success"]:
        return
    try:
        for category in config["trays"]:
            view.announce("Panda", f"Closing {category} box")
            angle = operate_lid(arm, category, opening=False)
            report["lids"].append({"category": category, "action": "close", "angle_degrees": angle})
    except RuntimeError as error:
        report["lids"].append({"category": category, "action": "close", "error": str(error)})
        raise
    arm.hold(settle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--headless", action="store_true", help="Verify the full task without opening a window")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inference-python", type=Path, default=ROOT / ".venv-inference/bin/python")
    args = parser.parse_args()
    if args.config is not None:
        args.config = args.config.resolve()
    args.inference_python = args.inference_python.absolute()
    output = args.output or ROOT / "outputs" / f"sorting-{args.seed}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    output.mkdir(parents=True, exist_ok=args.output is not None)
    config = load_config(args.config)
    report = {"seed": args.seed, "success": False, "lids": [], "objects": [], "inference": []}

    with TemporaryDirectory(prefix="tabletop-scene-") as scratch:
        # 完整任务使用无盒盖执行器的场景，盒盖由机械臂抓住把手开合。
        scene = build_scene(config, sample_layout(config, args.seed), Path(scratch) / "scene.xml", powered_lids=False)
        model = mj.MjModel.from_xml_path(str(scene))
        tune_gripper(model)
        data, display = mj.MjData(model), mj.MjData(model)
        for state in (data, display):
            initialize(model, state, lids_open=False)
        if args.headless:
            window = nullcontext()
        else:
            from mujoco.viewer import launch_passive
            window = launch_passive(model, display, show_left_ui=False, show_right_ui=False)

        with window as viewer:
            view = LiveView(viewer, model, display, data)
            arm = PandaController(model, data, on_step=view.step)
            try:
                run_task(arm, config, args, view, report)
            except ViewerClosed:
                report["error"] = "Viewer closed before the task finished"
            except (RuntimeError, OSError) as error:
                report["error"] = str(error)
                print(f"[ERROR] {error}", file=sys.stderr, flush=True)

            # 即使任务中途停止也记录最终状态。成功要求物品合格、全部盒盖闭合且合盖动作齐全。
            report["final_evaluation"] = evaluate(model, data, config)
            report["final_lids"] = []
            for category in config["trays"]:
                joint = model.joint(f"storage_box_lid_{category}")
                angle = float(data.qpos[joint.qposadr[0]])
                report["final_lids"].append({"category": category, "angle_degrees": float(np.rad2deg(angle)),
                                             "closed": bool(abs(angle - joint.range[0]) <= np.deg2rad(3))})
            report["success"] = ("error" not in report and report["final_evaluation"]["success"]
                                 and all(lid["closed"] for lid in report["final_lids"])
                                 and sum(lid["action"] == "close" for lid in report["lids"]) == len(config["trays"]))
            path = output / "sorting_report.json"
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            count = report["final_evaluation"]["sorted_count"]
            view.announce("Result", f"{count}/{len(config['objects'])} sorted")
            print(f"Report: {path}", flush=True)
            if viewer:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(.05)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
