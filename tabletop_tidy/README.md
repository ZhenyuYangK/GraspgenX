# 桌面语义分类整理

项目目标是结合MuJoCo、SAM3和NVIDIA GraspGenX，让Panda机械臂自动将六种物体放入三个浅托盘，失败时有限重试并记录结果。当前版本为仿真基线，后续接入模型推理和自动抓放。

| 收纳类别 | 物体 |
|---|---|
| 餐厨用品 | 杯子、勺子 |
| 工具 | 扳手、螺丝刀 |
| 办公用品 | 订书机、胶带座 |

初始位置和桌面朝向随机，物体之间预留夹爪空间。物体保留可辨认外形，允许缩放和加厚。完成标准为六个物体分别完整位于对应托盘内，并在盘底稳定停留，不要求统一朝向。

## 已验证部分

- 独立Panda模型、六个自由刚体物体原型和三个浅托盘。
- 可复现的随机位置与朝向，布局按物体包围半径预留间距。
- RGB-D离屏采集、相机标定及坐标回投。
- 独立验收器检查类别、完整入盘、盘底支撑和静止状态。
- 50组随机布局的间距检查，以及错盘、悬空、运动中物体的拒绝检查。

当前入口用于场景预览和权重文件检查，输出中的`task_executed`为`false`。SAM3分割、GraspGenX抓取生成及六物体自动抓放属于后续实现阶段。

## 运行场景预览

仿真基线使用Python3.11。新克隆项目后，在源码目录建立独立环境并安装已验证版本：

```bash
cd GraspgenX/tabletop_tidy
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-sim.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python run.py --seed 42
```

当前以源码检出和editable安装运行，`assets/`和`configs/`保留在项目目录。SAM3推理环境另行配置。

查看交互窗口：

```bash
.venv/bin/python run.py --seed 42 --viewer
```

预览输出位于`outputs/scene/`，包含`rgb.png`、`overview.png`、米制深度、相机标定、初始布局和场景检查记录。窗口预览尚未进行人工验证。

布局、物体名称、分类映射、托盘位置及验收阈值位于`configs/task.json`。物体原型的几何尺寸位于`src/tabletop_tidy/scene.py`，单位为米。初始间距只约束物体之间的距离；实际抓取可达性需要在模型与运动规划接入后检查。

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

## 后续实现顺序

1. 补齐SAM3权重，验证六个物体的文本分割与深度点云。
2. 使用GraspGenX的`franka_panda`配置生成抓取候选，核对夹爪坐标系和TCP变换。
3. 加入可达性、碰撞和接触抓取检查，完成转运与盘内放置。
4. 加入逐物体重试、完整运行日志与六物体验收。

目标运行设备为RTX4060 Laptop，显存约8GB。是否采用分时加载，以两个模型的实际显存占用为准。运行时计划使用RGB-D图像获取物体信息；仿真物体真值用于场景诊断和独立验收。

## 验证与来源

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl .venv/bin/python -m pytest -q
```

本次提交前，在独立源码副本中使用已验证的本机依赖完成13项测试和离屏场景预览。全新虚拟环境的依赖解析与项目构建已通过，软件包下载因网速中断，完整安装验证另行进行。

Panda资产、相机和控制代码从原`../zts_mujoco_sort`复制，74个文件已逐一核对SHA-256，记录位于`assets/source_manifest.json`。独立运行和场景测试通过后，已按用户要求删除原目录。

未被新项目引用的旧`zts_mujoco_sam3`工程已按用户要求清理，其中的SAM1权重、重复机器人资产和历史输出一并删除。

模型来源和固定版本见`assets/THIRD_PARTY.md`、`vendor-lock.json`。`vendor/`中的上游源码由Git忽略，模型接入时按记录版本获取；运行当前仿真基线无需这两套上游源码或模型权重。历史工程的抓取结果未计入本项目验收。
