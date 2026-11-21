"""The integrity gate rejects corrupt downloads even at the correct file size."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

# 按文件路径导入下载脚本，直接测试校验函数，无需登录或下载权重。
script = Path(__file__).resolve().parents[1] / "scripts/download_sam3.py"
spec = importlib.util.spec_from_file_location("download_sam3", script)
download_sam3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_sam3)


def test_checkpoint_integrity_rejects_same_size_corruption(tmp_path):
    path = tmp_path / "weights"
    original = b"downloaded checkpoint payload"
    digest = hashlib.sha256(original).hexdigest()
    path.write_bytes(original)
    assert download_sam3.verify_file(path, len(original), digest) == digest
    # 改首字节保留长度，再删一个字节，分别触发哈希错误和大小错误。
    path.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError, match="SHA-256"):
        download_sam3.verify_file(path, len(original), digest)
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="大小"):
        download_sam3.verify_file(path, len(original), digest)
