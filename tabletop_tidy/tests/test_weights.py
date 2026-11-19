import json
from pathlib import Path
import pytest
from tabletop_tidy.weights import inspect_checkpoint, scan_checkpoints


def test_preview_rejects_unused_checkpoint_argument(capsys):
    from tabletop_tidy.cli import main
    with pytest.raises(SystemExit) as error:
        main(["--checkpoint", "sam3.pt"])
    assert error.value.code == 2
    assert "--checkpoint requires --check-sam3" in capsys.readouterr().err


def write_safetensors(path, key, *, truncated=False):
    header = json.dumps({key: {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little")+header+b"\0"*(4 if truncated else 8))


def test_path_reference_and_sam1_are_not_sam3_weights(tmp_path):
    reference = tmp_path / "sam3-mirror"
    reference.write_text("/missing/old_machine/sam3")
    result = inspect_checkpoint(reference)
    assert result["status"] == "path_reference_text"
    assert not result["is_sam3_candidate"]
    sam1 = tmp_path / "sam_vit_b_01ec64.pth"
    sam1.write_bytes(b"not-loaded")
    assert inspect_checkpoint(sam1)["status"] == "sam1_weights"


def test_truncated_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "sam3.safetensors"
    write_safetensors(path, "detector.weight", truncated=True)
    result = inspect_checkpoint(path)
    assert result["status"] == "invalid_safetensors"
    assert not result["is_sam3_candidate"]


def test_structural_check_distinguishes_model_formats(tmp_path):
    path = tmp_path / "sam3.safetensors"
    write_safetensors(path, "detector.weight")
    native = inspect_checkpoint(path)
    assert native["is_sam3_candidate"]
    assert not native["model_loaded"] and not native["inference_run"]
    write_safetensors(path, "detector_model.weight")
    assert not inspect_checkpoint(path)["is_sam3_candidate"]


def test_lfs_pointer_and_dependency_files_are_not_loaded(tmp_path):
    path = tmp_path / "sam3.pt"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 4000000000\n")
    assert inspect_checkpoint(path)["status"] == "git_lfs_pointer"
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "sam3.pt").write_bytes(b"unrelated")
    assert len(scan_checkpoints(tmp_path)) == 1
