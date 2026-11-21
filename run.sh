#!/usr/bin/env bash
set -euo pipefail

# 以脚本所在目录定位仿真环境，从其他目录启动时也能找到入口。
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/tabletop_tidy" && pwd)
export PYTHONDONTWRITEBYTECODE=1
# 优先使用调用者指定的渲染后端；默认用GLFW开窗口，离屏运行用EGL。
if [[ -z ${MUJOCO_GL:-} ]]; then
    MUJOCO_GL=glfw
    for argument in "$@"; do
        if [[ $argument == --headless ]]; then
            MUJOCO_GL=egl
            break
        fi
    done
fi
export MUJOCO_GL
# 用仿真进程替换当前Shell，原样传入参数并由它返回退出状态。
exec "$project_dir/.venv/bin/python" "$project_dir/scripts/run_sorting.py" "$@"
