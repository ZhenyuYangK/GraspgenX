# 桌面语义分类整理

项目使用MuJoCo、SAM3和NVIDIA GraspGenX，让Panda机械臂将六种物体放入三个翻盖收纳盒，失败时有限重试并记录结果。整理从三个闭盒启动，由Panda依次开盖，六件物品入盒并通过检查后依次合盖。

| 盒身标签 | 物体 | 盒盖开关键 |
|---|---|---|
| 餐厨用品 | 杯子、勺子 | `1` |
| 工具 | 扳手、螺丝刀 | `2` |
| 办公用品 | 订书机、胶带座 | `3` |

初始位置和桌面朝向由随机种子决定，物体之间预留夹爪空间。整轮验收要求六件物体完整位于各自的收纳盒内、低于盒沿、受到盒底支撑并保持静止，三个盒盖全部闭合。

## 实时桌面整理演示

配置好下文的仿真环境、推理环境和模型权重后，从项目根目录启动。

```bash
./run.sh
```

窗口会先显示当前桌面。Panda依次接触开盖，系统获取同帧RGB-D，SAM3分割物品，GraspGenX生成抓取候选。Panda执行经过可达性、接触和入盒检查的抓放，六件物品均通过入盒检查后继续合盖。鼠标可调整观察角度，演示结束后窗口保留最终场景。

终端显示运行阶段，逐件抓取、模型调用、盒盖动作及最终验收写入`outputs/sorting-*/sorting_report.json`。使用`./run.sh --seed 42 --headless`可离屏运行同一流程并输出报告。

使用带Matplotlib和中文字体的Python环境，从仓库根目录将报告绘制为PNG。默认把图片写在JSON旁，可通过`--output`指定保存路径，或通过`--font`指定中文字体文件。

```bash
python3 tabletop_tidy/scripts/plot_sorting_report.py /实际路径/sorting_report.json
```

整理模式使用带开合把手和配重的无电机盒盖。勺子和扳手经加厚以提高夹爪接触余量，并保留各自质量。两个模型按阶段运行。仿真物体真值用于接触诊断和独立入盒验收。

## 运行场景预览

仿真基线使用Python3.11。新克隆项目后，在源码目录建立独立环境并安装已验证版本：

```bash
cd GraspgenX/tabletop_tidy
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-sim.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python run.py --seed 42
```

仿真以源码检出和editable安装运行，`assets/`和`configs/`由项目直接读取。`run.py`提供场景预览和权重文件检查。

查看交互窗口：

```bash
.venv/bin/python run.py --seed 42 --viewer
```

盒盖默认打开。在窗口中按表中的数字键可以分别开关三个盒盖。加上`--boxes-closed`可以从三个盒盖合上的状态启动，也可以用于离屏预览。

预览输出位于`outputs/scene/`，包含`rgb.png`、`overview.png`、米制深度、相机标定、初始布局和场景检查记录。

布局、物体名称、分类映射、收纳盒位置及验收阈值位于`configs/task.json`。收纳盒配置使用`trays`字段。盒身和物体原型的几何尺寸位于`src/tabletop_tidy/scene.py`，单位为米。

螺丝刀采用带橡胶嵌条的收腰握柄、金属套环、圆杆和渐薄一字刀头。胶带座采用圆角底座、曲面侧壁、中空卷轴、纸芯、带卷绕纹理的胶带卷及锯齿切刀，胶带沿卷面切线延伸至切刀。两者的显示网格与碰撞几何分别构建，碰撞几何保留卷轴孔和出带下方的空隙。

## SAM3权重检查

先在[官方模型页](https://huggingface.co/facebook/sam3)申请访问。使用Python3.12建立独立下载环境，获批后在本机终端登录并下载：

```bash
python3.12 -m venv .venv-download
.venv-download/bin/python -m pip install -r requirements-download.txt
.venv-download/bin/hf auth login
.venv-download/bin/python scripts/download_sam3.py
```

登录在本机终端完成，Token不写入项目配置。下载脚本从`facebook/sam3`获取原生`sam3.pt`及`config.json`，固定仓库提交版本，并按官方大小和SHA-256校验。权重保存在`checkpoints/sam3/`，校验记录为`download_manifest.json`。

仅检查账号对官方权重的访问权限：

```bash
.venv-download/bin/python scripts/download_sam3.py --check-access
```

扫描当前工作区中的权重文件：

```bash
.venv/bin/python run.py --check-sam3
```

检查指定文件：

```bash
.venv/bin/python run.py --check-sam3 --checkpoint /实际路径/sam3.safetensors
```

文件检查识别SAM1权重、Git LFS指针、记录旧路径的文本文件及截断的safetensors。结构检查与真实模型加载分开记录。

官方SAM3权重统一保存在`checkpoints/sam3/`，下载完成后以`download_manifest.json`记录的大小和SHA-256校验结果为准。下载期间保留该目录中的缓存、锁文件和未完成文件，供后台进程继续使用。

## 分阶段推理

推理环境安装锁定版本的`vendor/sam3`和`vendor/GraspGenX`，以及与本机CUDA兼容的PyTorch和torchvision。两个模型依赖NumPy1.x，仿真环境使用NumPy2.x。以下命令从已有可用的CUDA版PyTorch环境创建独立推理环境，在激活该PyTorch环境后执行。

```bash
python -m venv --system-site-packages .venv-inference
.venv-inference/bin/python -m pip install 'setuptools<78' -e vendor/sam3 -e vendor/GraspGenX einops psutil pycocotools
```

使用下载环境获取GraspGenX权重和Panda夹爪资源。文件版本记录在`vendor-lock.json`。

```bash
.venv-download/bin/hf download adithyamurali/GraspGenXModel \
  --revision 7c834043c11a11417e31d6d5ea9355801e40a2c1 \
  --include 'release/**' --local-dir checkpoints/graspgenx
.venv-download/bin/hf download adithyamurali/gripper_descriptions --repo-type dataset \
  --revision 19a03c00d19aeaf052d0f6801f0041982d676e8a \
  --include 'gripper_descriptions/__init__.py' \
  --include 'gripper_descriptions/assets/x_grippers/franka_panda/**' \
  --local-dir checkpoints/gripper_descriptions
```

先保存同帧RGB-D和标定，再分别运行两个推理阶段。每个推理输出参数使用新的目录。

```bash
.venv/bin/python run.py --seed 42 --output outputs/debug-42/scene
.venv-inference/bin/python scripts/infer_scene.py segment \
  --scene outputs/debug-42/scene --output outputs/debug-42/segmentation
.venv-inference/bin/python scripts/infer_scene.py grasp \
  --segmentation outputs/debug-42/segmentation --output outputs/debug-42/grasps
```

分割结果包含每件物品的提示词、分数、掩膜和Panda基座坐标系下的点云。多个提示词返回相同实例时，联合匹配六类物品。抓取阶段输出GraspGenX原始位姿、分数和转换后的Panda工具中心点位姿，供后续可达性与碰撞检查使用。

种子`42`已在RTX4060 Laptop上完成分阶段推理。SAM3识别六件物品，与独立仿真掩膜的IoU为0.868至0.981。GraspGenX对每件物品生成100个候选。两阶段实测PyTorch峰值显存分配分别为3.79GiB和0.77GiB。

## 验证与来源

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl .venv/bin/python -m pytest -q
```

测试覆盖50组随机布局的间距、RGB-D标定与坐标回投、错盒和悬空等验收异常，以及Panda通过把手物理开合三个盒盖。完整整理演示使用种子`42`通过验收，六件物品正确入盒，三个盒盖全部闭合。

Panda资产、相机和控制代码的来源及SHA-256校验记录见`assets/source_manifest.json`。

模型来源和固定版本见`assets/THIRD_PARTY.md`、`vendor-lock.json`。完整整理演示使用`vendor/`中按记录版本获取的上游源码及已下载的模型权重。
