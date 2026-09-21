# Agent & Server

```text
① main / navigation 主流程

run_eval.sh
↓
main.py
↓
读取 config
↓
init_env()
↓
创建 Habitat VectorEnv
↓
Agent(env, config)
↓
Agent.act()
↓
每个 episode:
    env.reset()
    ↓
    instruction DAG → Instruction Graph
    ↓
    每一步获取 Habitat observation
    RGB / Depth / GPS / Compass
    ↓
    SceneGraph.get_scenegraph()
    ↓
    调用 GSAM2 Server 做感知
    ↓
    更新 3D SceneGraph + BEV Map
    ↓
    Region Solver
    ↓
    constraint + navigation tree
    ↓
    生成 next_point
    ↓
    env.step(action)
    ↓
    机器人移动
    ↓
    循环直到 STOP / 结束
↓
计算 SR / SPL / OSR 等
↓
保存结果
```

这条就是 **GC-VLN 的“大脑 + Habitat 执行”**。

---

另一条是独立的感知 server：

```text
② GSAM2 Server 流程

GSAM2.py --port 7000
↓
加载 Grounding DINO
↓
加载 SAM2
↓
启动 HTTP Model Server
↓
等待 main 发送请求
        ↓
收到:
    RGB image
    +
    text prompt
    例如:
    "sofa . doorway . table ."
        ↓
Grounding DINO
↓
text-conditioned object detection
↓
bbox + label + confidence
↓
SAM2
↓
根据 bbox 做精细 segmentation
↓
mask
↓
返回 main:
    masks
    boxes
    captions
    mask confidence
    caption confidence
↓
继续等待下一次请求
```

所以整个系统实际上是：

```text
                 ┌──────── GSAM2 Server ────────┐
                 │ Grounding DINO → SAM2         │
                 │           ↑        ↓          │
                 └────────── HTTP ────────────────┘
                             ↑
                             │ RGB + prompt
                             │ mask + caption
                             ↓
Habitat → main.py → Agent → SceneGraph
                           ↓
                       BEV + SG
                           ↓
                     Region Solver
                           ↓
                       waypoint
                           ↓
                        Habitat
```












# Region Solver（RS）

GC-VLN 整体

```text
指令
"Go past the sofa, then turn left at the table"

        ↓
① Instruction Graph
把句子拆成阶段和约束

Stage 1: pass sofa
Stage 2: left of table

        ↓
② SceneGraph
从相机里找到实际物体

sofa 在这里
table 在这里
door 在这里

        ↓
③ BEV Map
建立二维地图

哪里能走
哪里是墙
哪里是障碍物

        ↓
④ Region Solver  ← 你现在看的 RS
把：
「指令要求」
+
「物体位置」
+
「哪里能走」

结合起来

        ↓
找出一片“应该去的区域”

        ↓
⑤ 从区域里选 waypoint

        ↓
⑥ Navigation / Habitat
机器人真正走过去

        ↓
重新观察 → 再规划
```

- **根据语言关系生成允许区域**。
- **和 BEV 可通行区域求交集**。
- **从剩下区域选 waypoint**。

求满足语言约束和满足BEV约束的交集
