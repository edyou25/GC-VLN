# HDF5 调试日志

默认开启，沿用原来的评估命令。安装依赖：

```bash
pip install 'h5py>=3.8,<4'
bash run_eval.sh rxr
```

每个 Episode 一个文件，重复运行不会覆盖旧日志：

```text
outputs/logs/<dataset>/<split>/<experiment_id>/hdf5/
  rank_<rank>_split_<split_index>/episode_<id>_<unique_id>.h5
```

`config/default.py` 的 `DEBUG_LOG.ENABLED` 和 `DEBUG_LOG.COMPRESSION`
控制默认行为。也可以在 `main.py` 命令的尾部配置项之前传入
`--no-debug-log`、`--debug-log` 或 `--debug-log-compression gzip`。
压缩支持 `lzf`（默认）、`gzip`、`none`，均不改变数组数值。

## 文件结构

```text
episode/
  id, scene, instruction, dag_raw, dag
  gt/{reference_path, goals, shortest_paths, trajectory}
  start_position, start_rotation
  config, config_yaml, worker_config, cameras
  experiment_id, environment_index
steps/000000/
  rgbd/{rgb, depth, rgb_30, depth_30, ..., gps, compass,
        panorama_rgb, panorama_depth, camera_matrix, pose_matrix}
  perception/
    detected
    detections/{boxes, labels, scores, masks, mask_scores}
    rooms/{boxes, labels, scores}
  mapping/{bev, wall, thin, fmm}
  scene_graph/{nodes, edges, attributes}
  planning/
    stage_before, stage, stage_result, constraints, masks
    candidates, final_points, selected_point, best_point, navigation_mode
    navigation_tree/{navigation_tree, waypoints_tree, sequence_tree, path, ...}
  action/{act, next_point, current_pos, current_heading, location}
  motion/
    action_valid
    frames/000000/
      rgbd/{rgb, depth, rgb_30, depth_30, ..., gps, compass}
      before/{position, rotation_xyzw, heading, sensors}
      after/{position, rotation_xyzw, heading, sensors}
      action OR reason OR stuck
      stuck_time, ...
  outcome, done
  error                         # 出错时存在
metrics                         # Episode 结束时存在
```

## 时序和坐标

- `steps` 是高层规划循环。每次更新 perception、mapping、scene graph 后保存，
  再保存选点结果并执行运动。规划异常会保存 traceback，尚未运行的模块标记为 `none`。
- `motion/frames` 是该 Step 内按顺序记录的底层事件。每个前进、左右转和 STOP 都保存
  动作执行后的全部 RGB-D 视角，以及执行前后 3D 位姿；没有开视频也会采集。
  `event` 属性为 `start`、`action`、`teleport` 或 `stuck`。
  `start` 没有执行动作，`teleport` 保存原因和目标，`stuck` 保存位移与规划结果。
  帧属性还包含 `step_index`、`frame_index`、`timestamp_ns` 和 `planning_context`。
- 底层运动期间**不重新运行感知/建图/规划**；这些帧的模块上下文通过
  `planning_context` 指向所属 Step。候选点包含参与聚类的原始点与 cluster IDs；
  聚类后累计点在 `final_points`，已有可选路点及回溯路径保存在 navigation tree 中。
- `rgbd/depth` 是 Habitat Depth Sensor 的原始输出，保留 shape、dtype、NaN/Inf 和数值，
  没有转成 PNG 或做日志专用归一化。当前 R2R/RxR 配置输出米制深度；
  Habitat 自身的 MIN_DEPTH / MAX_DEPTH 裁剪仍存在。
  `episode/cameras/<sensor_uuid>` 保存实际传感器的分辨率、HFOV、安装位置/角度和深度配置。
- `before/after/position` 是 Habitat 世界坐标中的三维位置（米），四元数顺序为 XYZW，
  heading 为弧度；`sensors` 保存每个相机的世界位置和旋转。
  Step 内 `camera_matrix` / `pose_matrix` 是建图实际使用的全景内参和变换。
  BEV 点使用现有代码的 `[row, column]` 网格坐标，分辨率和原点见 Config。
- GT 来自 Episode 的参考路径、目标、最短路径，以及已有 RxR GT 文件。
  数据集未提供的字段标记 `none`，不会生成替代 GT。

## 读取示例

```python
from pathlib import Path
import h5py

path = next(Path('outputs/logs').rglob('episode_*.h5'))
with h5py.File(path, 'r') as f:
    print(f['episode/id'].asstr()[()], f.attrs['status'])
    step = f['steps/000000']
    rgb = step['rgbd/rgb'][:]
    depth = step['rgbd/depth'][:]
    bev = step['mapping/bev'][:]
    boxes = step['perception/detections/boxes'][:]
    labels = step['perception/detections/labels'].asstr()[:]
    for node in step['scene_graph/nodes'].values():
        print(node['id'][()], list(node['attributes']))
    for frame in step['motion/frames'].values():
        print(frame.attrs['event'], frame['after/position'][:])
```

数字数组直接存为 HDF5 dataset；Unicode 使用 UTF-8。列表、元组、集合用
`000000`、`000001` 等子组保存，并通过 `type` 属性区分。普通字符串键字典直接用键名，
非字符串键或含 `/` 的字典使用编号 entry 的 `key` / `value`，避免类型和路径冲突。
`None` 是 `type=none` 的空组，空数组仍保留原始 shape / dtype。
Constraint 等对象保存完整字段及 `python_class`，不使用 pickle。

所有图保存完整节点 ID、边端点、图/节点/边属性；有向性和多重边信息在属性中。
不会为可视化截断节点、边或候选点。

## 写入和开销

主进程和环境 worker 依次打开同一个 Episode 文件，传递文件路径，不传递 HDF5 handle。
每个动作帧 flush，Step 写入后关闭文件。正常结束根属性为 `status=complete`；
达到规划上限为 `max_steps`；捕获到中断/异常为 `interrupted`。
motion 自己记录 `running` / `complete` / `error`。强制杀进程或断电不保证最后一帧完整。

全量 RGB-D（包括 12 个视角）、mask、地图和完整图会增加渲染时间与磁盘占用。
日志逐帧落盘，不在进程间传输整段视频。关闭日志后不额外采集底层动作帧。

CPU 测试不依赖 Habitat、GLIP 或 GSAM2 服务：

```bash
python -m unittest discover -s tests -v
```

## Matplotlib 逐帧查看器

在有桌面显示的环境中打开一个 Episode 文件：

```bash
pip install 'h5py>=3.8,<4' 'matplotlib>=3.5,<4' networkx numpy
python scripts/view_hdf5_log.py /path/to/episode_123_xxx.h5
```

无需运行 Habitat、模型服务或 GPU。查看器默认使用 Matplotlib 的交互后端；
如需指定，可加 `--backend TkAgg` 或 `--backend QtAgg`（相应 GUI 依赖需已安装）。
请在日志写完、文件关闭后打开；查看器不做实时追尾。

操作方式：

| 操作 | 功能 |
| --- | --- |
| 拖动底部 Frame 滑块 | 按整个 Episode 的事件顺序跳转到任意帧 |
| 左右键 / Previous、Next | 上一帧、下一帧 |
| 上下键 / Previous step、Next step | 跳到前后 Step 的首帧 |
| 空格 / Play / Pause | 播放、暂停，默认每秒 5 帧 |
| V / Camera + | 轮换查看 RGB-D 的各个相机视角 |
| Home / End | 第一帧、最后一帧 |
| I / Inspect HDF5 | 打开数据查看窗口，输入 HDF5 绝对路径并按 Enter |

上方显示**当前运动帧**的 RGB 和 Depth；检测面板单独显示**所属 Step 开始时**的
全景图、Boxes、Labels、Scores 和 Masks，避免把运动前的检测叠加到运动后的 RGB 上。
其余面板展示四张地图、Scene Graph、Navigation Tree、指令 DAG 和世界坐标轨迹/GT。
地图青色点和线表示当前 GPS 位置与朝向；FMM 上叠加绿色有效区域、橙色候选点和红色选中点。
候选点非常密集时，每次聚类结果最多抽样 3000 个点用于显示，HDF5 原数据不变。

Frame 索引从 0 开始，包含 `start`、`action`、`teleport`、`stuck` 事件。
没有 motion 的异常 Step 也会作为一帧显示，未记录模块显示 `Not recorded`。
节点较多时图中隐藏文字标签，节点和边仍全部绘制。

Inspect HDF5 窗口默认打开当前 Step 的约束。也可以输入这些路径：

```text
/episode                         # Instruction、Config、GT、相机参数
/steps/000000/planning/constraints
/steps/000000/scene_graph/nodes
/steps/000000/planning/navigation_tree
/steps/000000/motion/frames/000001
```

窗口中的 Line 滑块或鼠标滚轮可滚动数据摘要。子树默认展开三层，输入更深路径可继续检查。
大型数组显示 shape / dtype；输入数组本身的路径时，预览各维度前 4 项，避免加载全部内容。

可指定起始帧、播放速度、深度显示上限：

```bash
python scripts/view_hdf5_log.py episode.h5 --frame 120 --fps 10 --depth-max 8
```

无桌面的服务器可导出指定帧 PNG，不启动交互窗口：

```bash
python scripts/view_hdf5_log.py episode.h5 --frame 120 --save outputs/frame_120.png
```

图像按当前帧和相机视角读取；检测、地图和图结构只在切换 Step 时更新。
启动时只索引小型元数据与轨迹，不会把 Episode 的全部 RGB-D / Masks 读入内存。
