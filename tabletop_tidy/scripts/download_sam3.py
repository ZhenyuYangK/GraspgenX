#!/usr/bin/env python3
"""Download the native SAM3 checkpoint from Meta's authorized official repo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = "facebook/sam3"
ENDPOINT = "https://huggingface.co"
sys.path.insert(0, str(ROOT / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, size: int, expected_sha256: str) -> str:
    if path.stat().st_size != size:
        raise ValueError("文件大小与官方元数据不一致")
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise ValueError("文件SHA-256与官方元数据不一致")
    return actual


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-access", action="store_true", help="Check official access without downloading weights")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/sam3")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi, get_token, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from httpx import TransportError

    if not get_token():
        print("请先在本机终端登录HuggingFace：", file=sys.stderr)
        print(f"{ROOT / '.venv-download/bin/hf'} auth login", file=sys.stderr)
        return 3

    try:
        # Token is resolved by the official client, never saved in project files.
        info = HfApi(endpoint=ENDPOINT).model_info(REPO, files_metadata=True, token=True)
        checkpoint = next(item for item in info.siblings if item.rfilename == "sam3.pt")
        lfs = checkpoint.lfs
        expected_sha = getattr(lfs, "sha256", None)
        expected_size = checkpoint.size or getattr(lfs, "size", None)
        if not info.sha or expected_size is None or not expected_sha:
            raise ValueError("官方仓库未返回完整的权重版本、大小和SHA-256元数据")
        metadata = {"repo_id": REPO, "revision": info.sha, "filename": "sam3.pt",
                    "size_bytes": expected_size, "expected_sha256": expected_sha,
                    "model_loaded": False, "inference_run": False}
        if args.check_access:
            # HEAD verifies access to the gated blob, rather than just its card.
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
            get_hf_file_metadata(hf_hub_url(REPO, "sam3.pt", revision=info.sha, endpoint=ENDPOINT), token=True)
            print(json.dumps({"access_verified": True, **metadata}, indent=2, ensure_ascii=False))
            return 0

        args.output.mkdir(parents=True, exist_ok=True)
        # Downloads are pinned to the metadata's commit; local_dir supports resume.
        for filename in ["config.json", "sam3.pt"]:
            hf_hub_download(REPO, filename, revision=info.sha, token=True,
                            endpoint=ENDPOINT, local_dir=args.output)
        path = args.output / "sam3.pt"
        print("下载完成，正在核对文件大小和SHA-256。", flush=True)
        metadata["sha256"] = verify_file(path, expected_size, expected_sha)
        from tabletop_tidy.weights import inspect_checkpoint
        structure = inspect_checkpoint(path)
        if not structure["is_sam3_candidate"]:
            raise ValueError("下载文件的PyTorch归档结构检查失败")
        metadata.update({"path": str(path.resolve()), "integrity_verified": True,
                         "archive_status": structure["status"]})
        report_path = args.output / "download_manifest.json"
        report_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0
    except GatedRepoError:
        print("SAM3权重访问尚未获批，或当前Token未获得该模型的读取权限。请在官方模型页完成授权。", file=sys.stderr)
        return 3
    except HfHubHTTPError as error:
        status = getattr(error.response, "status_code", "unknown")
        print(f"官方仓库请求失败，HTTP状态码：{status}。请检查本机登录、模型授权和网络后重试。", file=sys.stderr)
        return 4
    except TransportError:
        print("连接官方仓库失败。请检查网络后重试；已下载的片段可继续使用。", file=sys.stderr)
        return 4
    except (ValueError, OSError, StopIteration) as error:
        print(f"下载或校验失败：{type(error).__name__}。请检查输出目录和官方文件元数据。", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
