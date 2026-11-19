# 来源与许可证

Panda模型来自MuJoCo Menagerie的Franka Emika Panda，固定版本为`8161bba264d7fa7c99ca301e91e7fb44737676ad`，采用Apache-2.0许可证。原始模型文件、许可证、说明及资产清单保留在`robots/franka_emika_panda/`。

上游地址：https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda

`panda_scene.xml`、`src/tabletop_tidy/camera.py`及`src/tabletop_tidy/control.py`从用户提供的`zts_mujoco_sort`复制。Panda场景包含原项目增加的TCP、指腹名称和摩擦配置。复制来源及SHA-256记录在`source_manifest.json`。本项目通过主场景的compiler设置资产路径。

六种物体、桌面和浅托盘由本项目以MuJoCo几何组合生成。当前物体是可辨认外形的仿真原型，尺寸参数直接记录在场景生成代码中。

SAM3源码：https://github.com/facebookresearch/sam3

GraspGenX源码：https://github.com/NVlabs/GraspGenX

两套官方源码在本机保存于`vendor/`，该目录由Git忽略。模型接入所使用的上游版本记录在`vendor-lock.json`，各自许可证以对应仓库为准。当前仿真基线独立于这两套源码运行。
