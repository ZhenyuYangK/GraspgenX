from pathlib import Path
import sys

# 直接运行源码时，把src加入模块搜索路径，以便导入本地包。
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tabletop_tidy.cli import main

if __name__ == "__main__":
    # 此入口用于场景预览和权重结构检查；完整整理从仓库根目录的run.sh启动。
    raise SystemExit(main())
