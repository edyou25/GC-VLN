## GC-VLN Full Experiment Logger

按 Episode 保存，每个 Frame / Step 记录：

| 模块 | 记录内容 |
|---|---|
| Episode | ID、Scene、Instruction、DAG、GT、Config |
| RGB-D | 每帧 RGB、原始 Depth、相机参数 |
| Perception | Boxes、Labels、Scores、Masks |
| Mapping | BEV、Wall、Thin Map、FMM Map |
| Scene Graph | 完整 Nodes、Edges、Attributes |
| Planning | Stage、Constraints、候选点、选中点、Navigation Tree |
| Motion | 每个 Action、3D Pose、Heading、Teleport/Stuck |
| Metrics & Debug | 每步 Distance-to-Goal、最终指标、耗时、异常 |

**保存格式：** RGB 用 JPG；Depth、Mask、Map 用 NPZ；其他数据用 JSONL/JSON。

**离线分析：** 轨迹、距离曲线、感知变化、地图演化、规划决策、不同 GPU 实验对比。

**原则：** 保留算法实际产生的完整中间量，不额外运行模型；通过 `episode_id / step_id / frame_id` 关联所有数据。