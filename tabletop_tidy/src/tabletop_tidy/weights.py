"""Inspect local checkpoint files without unpickling or contacting a model host."""
from __future__ import annotations

import json
import os
from pathlib import Path
import zipfile


def inspect_checkpoint(path: Path) -> dict:
    path = Path(path)
    report = {"path": str(path.absolute()), "is_sam3_candidate": False,
              "model_loaded": False, "inference_run": False}
    if not path.is_file():
        return {**report, "status": "missing_file"}
    size = path.stat().st_size
    report["size_bytes"] = size
    if path.name.startswith("sam_vit_"):
        return {**report, "status": "sam1_weights"}
    if size < 1024:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeError:
            text = ""
        if text.startswith("version https://git-lfs.github.com/spec/"):
            return {**report, "status": "git_lfs_pointer"}
        if text.startswith("/"):
            return {**report, "status": "path_reference_text", "referenced_path": text,
                    "referenced_path_exists": Path(text).exists()}
    if path.suffix == ".safetensors":
        try:
            with path.open("rb") as stream:
                header_size = int.from_bytes(stream.read(8), "little")
                if not 0 < header_size <= min(size-8, 100_000_000):
                    raise ValueError("Invalid safetensors header size")
                header = json.loads(stream.read(header_size))
            tensors = {key: tensor for key, tensor in header.items() if key != "__metadata__"}
            if not tensors:
                raise ValueError("No tensors")
            payload = size-8-header_size
            intervals = []
            dtype_sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
                           "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
            for tensor in tensors.values():
                start, end = tensor["data_offsets"]
                if not 0 <= start <= end <= payload:
                    raise ValueError("Truncated or invalid tensor payload")
                shape = tensor["shape"]
                count = 1
                for dimension in shape:
                    if not isinstance(dimension, int) or dimension < 0:
                        raise ValueError("Invalid tensor shape")
                    count *= dimension
                if tensor["dtype"] in dtype_sizes and end-start != count*dtype_sizes[tensor["dtype"]]:
                    raise ValueError("Tensor byte count does not match shape")
                intervals.append((start, end))
            previous_end = 0
            for start, end in sorted(intervals):
                if start != previous_end:
                    raise ValueError("Overlapping tensors or gaps in payload")
                previous_end = end
            if previous_end != payload:
                raise ValueError("Trailing bytes outside tensor payload")
            if any(key.startswith("detector_model.") for key in tensors):
                return {**report, "status": "transformers_format_requires_separate_adapter"}
            if not any(key.startswith("detector.") for key in tensors):
                return {**report, "status": "unrecognized_safetensors_model"}
            return {**report, "status": "native_sam3_safetensors_structure",
                    "is_sam3_candidate": True, "tensor_count": len(tensors)}
        except (ValueError, KeyError, TypeError, OSError) as error:
            return {**report, "status": "invalid_safetensors", "error": str(error)}
    if path.suffix in {".pt", ".pth"} and path.name.startswith("sam3"):
        # Archive structure is checked here. Keys and model completeness are
        # checked by the actual model loader, never inferred from the filename.
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                if any(name.endswith("/data.pkl") for name in archive.namelist()):
                    return {**report, "status": "sam3_named_pytorch_archive_unloaded",
                            "is_sam3_candidate": True}
        return {**report, "status": "invalid_pytorch_archive"}
    return {**report, "status": "unrecognized_file"}


def scan_checkpoints(root: Path) -> list[dict]:
    candidates = []
    ignored = {".git", ".cache", "vendor", "outputs", "__pycache__", "node_modules"}
    for directory, dirs, filenames in os.walk(Path(root)):
        dirs[:] = [name for name in dirs if name not in ignored and not name.startswith(".venv")]
        for name in sorted(filenames):
            path = Path(directory) / name
            if name == "sam3-mirror" or path.suffix in {".pt", ".pth", ".safetensors"}:
                candidates.append(inspect_checkpoint(path))
    return sorted(candidates, key=lambda item: item["path"])
