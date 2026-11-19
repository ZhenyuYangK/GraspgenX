# GraspgenX桌面整理

使用MuJoCo、SAM3和NVIDIA GraspGenX完成Panda机械臂的桌面语义分类整理。项目代码位于[tabletop_tidy](tabletop_tidy/README.md)。

## 启动整理演示

从项目根目录运行`./run.sh`。MuJoCo窗口会打开，采集当前场景的RGB-D并调用SAM3和GraspGenX。Panda依次尝试开盖、分类抓放和合盖。终端显示进度，整轮结果写入`tabletop_tidy/outputs/sorting-*/sorting_report.json`。

## 安装与验证

仿真基线使用Python3.11，按源码检出和editable安装运行：

```bash
git clone https://github.com/ZhenyuYangK/GraspgenX.git
cd GraspgenX/tabletop_tidy
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-sim.txt
.venv/bin/python -m pip install --no-deps -e .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl .venv/bin/python -m pytest -q
.venv/bin/python run.py --seed 42
```

离屏渲染需要系统支持EGL。使用图形桌面时可执行`.venv/bin/python run.py --seed 42 --viewer`，然后按`1`、`2`、`3`分别开关餐厨用品、工具、办公用品收纳盒。

详细任务定义、配置、模型下载和源码来源见[项目说明](tabletop_tidy/README.md)。权重、虚拟环境、下载缓存、生成结果及上游源码目录均由Git忽略。
