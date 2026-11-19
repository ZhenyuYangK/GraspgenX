#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/tabletop_tidy" && pwd)
export PYTHONDONTWRITEBYTECODE=1
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
exec "$project_dir/.venv/bin/python" "$project_dir/scripts/run_sorting.py" "$@"
