
import os
import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def prepare_rgb(rgb):
    image = to_numpy(rgb)

    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.size and image.max() <= 1.0:
            image *= 255
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def prepare_map(x):
    image = to_numpy(x).squeeze()
    if image.ndim != 2:
        raise ValueError(f"Expected 2D map, got {image.shape}")
    return image


def draw_detections(ax, rgb, detection):
    """当前帧 GSAM2 的 2D 检测框和分割 mask。"""

    image = prepare_rgb(rgb).astype(np.float32)

    if detection is None:
        ax.imshow(image.astype(np.uint8))
        ax.set_title("GSAM2: no detection")
        ax.axis("off")
        return

    boxes = to_numpy(detection["xyxy"]).reshape(-1, 4)
    masks = to_numpy(detection["mask"])
    captions = detection["caption"]
    scores = to_numpy(detection["caption_conf"]).reshape(-1)

    if masks.ndim == 2:
        masks = masks[None, ...]

    colors = plt.get_cmap("tab20")(
        np.arange(max(len(boxes), 1)) % 20
    )

    # 先混合 mask，再绘制检测框和标签
    for k, mask in enumerate(masks[:len(boxes)]):
        if mask.shape != image.shape[:2]:
            continue

        region = mask.astype(bool)
        color = np.array(colors[k, :3]) * 255

        image[region] = 0.65 * image[region] + 0.35 * color

    ax.imshow(np.clip(image, 0, 255).astype(np.uint8))

    for k, box in enumerate(boxes):
        x1, y1, x2, y2 = map(float, box)
        color = colors[k]

        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )

        label = f"{k}: {captions[k]} {scores[k]:.2f}"

        ax.text(
            x1,
            max(y1 - 4, 0),
            label,
            fontsize=7,
            color="white",
            bbox=dict(
                facecolor=color,
                edgecolor="none",
                alpha=0.85,
                pad=2,
            ),
        )

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_title(f"GSAM2: {len(boxes)} detections")
    ax.axis("off")


def get_map_bounds(bev, nodes, margin=35, min_span=200):
    """
    以已知可通行区域及 SceneGraph 物体中心为依据自动裁剪。
    不使用 FMM 的白色区域，因为它可能包含未知区域。
    """

    h, w = bev.shape

    ys, xs = np.where(bev > 0.5)

    if nodes:
        xs = np.concatenate([
            xs,
            np.array([n["x"] for n in nodes]),
        ])
        ys = np.concatenate([
            ys,
            np.array([n["y"] for n in nodes]),
        ])

    if len(xs) == 0:
        return (
            w // 2 - 100, w // 2 + 100,
            h // 2 - 100, h // 2 + 100,
        )

    x0, x1 = xs.min() - margin, xs.max() + margin
    y0, y1 = ys.min() - margin, ys.max() + margin

    span = max(x1 - x0, y1 - y0, min_span)

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    x0 = max(0, cx - span / 2)
    x1 = min(w, cx + span / 2)
    y0 = max(0, cy - span / 2)
    y1 = min(h, cy + span / 2)

    return x0, x1, y0, y1


def setup_map_axis(ax, title, bounds):
    x0, x1, y0, y1 = bounds

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X [grid]")
    ax.set_ylabel("Y [grid]")
    ax.tick_params(labelsize=8)


def visualize_maps(
    bev,
    bev_fmm,
    bev_wall,
    bev_thin,
    scene_graph,
    save_path,
    rgb=None,
    depth=None,
    detection=None,
    objects_3d=None,
    mesh_preview_path=None,
    crop_bounds=None,
    max_graph_nodes=25,
    max_points_per_object=2500,
):
    """
    GC-VLN diagnostic visualization.

    Outputs:
        1. PNG: RGB / detection / depth / maps / SceneGraph / 3D objects
        2. CSV: SceneGraph node metadata

    mesh_preview_path:
        Optional screenshot of the original MP3D mesh.
        Does not load or render .glb automatically.
    """

    bev_np = prepare_map(bev)
    fmm_np = prepare_map(bev_fmm)
    wall_np = prepare_map(bev_wall)
    thin_np = prepare_map(bev_thin)

    # --------------------------------------------------
    # 1. 收集 SceneGraph 节点
    # --------------------------------------------------

    nodes = []

    for idx, (node_id, data) in enumerate(
        scene_graph.nodes(data=True)
    ):
        center = data.get("center")

        if center is None:
            continue

        center = to_numpy(center).reshape(-1)

        if center.size < 2:
            continue

        # GC-VLN: center[0] = row, center[1] = column
        y = float(center[0])
        x = float(center[1])

        nodes.append({
            "index": idx,
            "node_id": str(node_id),
            "caption": str(data.get("caption", "unknown")),
            "x": x,
            "y": y,
            "num_detections": int(data.get("num_detections", 0)),
            "caption_conf": float(data.get("caption_conf", 0)),
            "degree": int(scene_graph.degree[node_id]),
        })

    bounds = (
        crop_bounds
        if crop_bounds is not None
        else get_map_bounds(bev_np, nodes)
    )

    # --------------------------------------------------
    # 2. 创建面板
    #
    # Row 0: RGB Panorama
    # Row 1: Detection Panorama
    # Row 2: Depth Panorama
    # Row 3: Four BEV maps
    # Row 4: SceneGraph / 3D / NetworkX / Mesh
    # --------------------------------------------------

    fig = plt.figure(figsize=(24, 18), constrained_layout=True)

    gs = fig.add_gridspec(
        5, 4,
        height_ratios=[0.8, 0.8, 0.8, 1.5, 1.5],
    )

    ax_rgb = fig.add_subplot(gs[0, :])
    ax_detection = fig.add_subplot(gs[1, :])
    ax_depth = fig.add_subplot(gs[2, :])

    map_axes = [
        fig.add_subplot(gs[3, j])
        for j in range(4)
    ]

    ax_sg = fig.add_subplot(gs[4, 0])
    ax_3d = fig.add_subplot(gs[4, 1], projection="3d")
    ax_graph = fig.add_subplot(gs[4, 2])
    ax_mesh = fig.add_subplot(gs[4, 3])

    # --------------------------------------------------
    # 3. 原始 RGB Panorama
    # --------------------------------------------------

    if rgb is not None:
        rgb_np = prepare_rgb(rgb)

        ax_rgb.imshow(rgb_np)
        ax_rgb.set_title(
            f"Original RGB Panorama ({rgb_np.shape[1]} x {rgb_np.shape[0]})"
        )
    else:
        ax_rgb.text(
            0.5, 0.5, "RGB not provided",
            ha="center", va="center",
            transform=ax_rgb.transAxes,
        )

    ax_rgb.axis("off")

    # --------------------------------------------------
    # 4. GSAM2 2D 检测结果
    # --------------------------------------------------

    if rgb is not None:
        draw_detections(ax_detection, rgb, detection)
    else:
        ax_detection.text(
            0.5, 0.5, "RGB not provided",
            ha="center", va="center",
            transform=ax_detection.transAxes,
        )
        ax_detection.axis("off")

    # --------------------------------------------------
    # 5. 原始 Depth Panorama
    # --------------------------------------------------

    if depth is not None:
        depth_np = prepare_map(depth).astype(np.float32)

        valid = (
            np.isfinite(depth_np)
            & (depth_np > 0)
        )

        depth_vis = np.ma.masked_where(~valid, depth_np)

        im = ax_depth.imshow(
            depth_vis,
            cmap="viridis",
            vmin=0,
            vmax=10,
            interpolation="nearest",
        )

        ax_depth.set_title("Original Depth Panorama [m]")

        fig.colorbar(
            im,
            ax=ax_depth,
            fraction=0.015,
            pad=0.01,
            label="Depth [m]",
        )
    else:
        ax_depth.text(
            0.5, 0.5, "Depth not provided",
            ha="center", va="center",
            transform=ax_depth.transAxes,
        )

    ax_depth.axis("off")

    # --------------------------------------------------
    # 6. 四张 BEV 地图
    # --------------------------------------------------

    maps = [
        ("BEV: Traversability", bev_np),
        ("BEV: FMM / Non-obstacle", fmm_np),
        ("BEV: Wall", wall_np),
        ("BEV: Skeleton", thin_np),
    ]

    for ax, (title, image) in zip(map_axes, maps):

        ax.imshow(
            image,
            cmap="gray",
            origin="upper",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )

        setup_map_axis(ax, title, bounds)

    # --------------------------------------------------
    # 7. SceneGraph 节点叠加 BEV
    # --------------------------------------------------

    ax_sg.imshow(
        bev_np,
        cmap="gray",
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    for n in nodes:

        ax_sg.scatter(
            n["x"],
            n["y"],
            s=45,
            c="red",
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

        # 使用编号，与 CSV 和 NetworkX 图一致
        ax_sg.annotate(
            str(n["node_id"]),
            (n["x"], n["y"]),
            xytext=(4, -5),
            textcoords="offset points",
            fontsize=7,
            color="yellow",
            fontweight="bold",
            zorder=4,
        )

    setup_map_axis(
        ax_sg,
        f"SceneGraph on BEV ({len(nodes)} objects)",
        bounds,
    )

    # --------------------------------------------------
    # 8. 累积 3D Object Point Clouds
    # --------------------------------------------------

    total_points = 0
    plotted_objects = 0

    if objects_3d is not None:

        # 只画在 SceneGraph 中存在的对象，避免显示被过滤的实例
        graph_ids = set(scene_graph.nodes())

        for obj_id in range(len(objects_3d)):

            if obj_id not in graph_ids:
                continue

            obj = objects_3d[obj_id]

            if "pcd" not in obj:
                continue

            points = np.asarray(obj["pcd"].points)

            if points.ndim != 2 or points.shape[1] < 3:
                continue

            if len(points) == 0:
                continue

            # 降采样：只用于绘图，不修改原始点云
            if len(points) > max_points_per_object:
                indices = np.linspace(
                    0,
                    len(points) - 1,
                    max_points_per_object,
                    dtype=int,
                )
                points = points[indices]

            ax_3d.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=0.5,
                alpha=0.6,
                label=f"ID {obj_id}",
            )

            total_points += len(points)
            plotted_objects += 1

        if plotted_objects:
            ax_3d.set_box_aspect((1, 1, 0.6))
            ax_3d.view_init(elev=55, azim=-70)

            # 太多物体时不显示 legend，避免遮挡
            if plotted_objects <= 12:
                ax_3d.legend(
                    fontsize=6,
                    loc="upper right",
                )

    if plotted_objects == 0:
        ax_3d.text2D(
            0.5, 0.5,
            "No 3D objects",
            ha="center",
            va="center",
            transform=ax_3d.transAxes,
        )

    ax_3d.set_title(
        f"Accumulated 3D Objects\n"
        f"{plotted_objects} objects / {total_points} plotted points"
    )

    ax_3d.set_xlabel("World X [m]", fontsize=8)
    ax_3d.set_ylabel("World Y [m]", fontsize=8)
    ax_3d.set_zlabel("World Z [m]", fontsize=8)

    # --------------------------------------------------
    # 9. NetworkX SceneGraph
    # --------------------------------------------------

    selected_ids = list(scene_graph.nodes())[:max_graph_nodes]
    subgraph = scene_graph.subgraph(selected_ids)

    if len(selected_ids):

        pos = nx.spring_layout(
            subgraph,
            seed=42,
            iterations=100,
        )

        labels = {
            node_id: (
                f"{node_id}: "
                f"{scene_graph.nodes[node_id].get('caption', '?')}\n"
                f"({scene_graph.nodes[node_id].get('num_detections', 0)})"
            )
            for node_id in selected_ids
        }

        nx.draw_networkx(
            subgraph,
            pos=pos,
            labels=labels,
            ax=ax_graph,
            node_size=1050,
            node_color="skyblue",
            edge_color="gray",
            width=0.6,
            font_size=6.5,
        )

    else:
        ax_graph.text(
            0.5, 0.5,
            "SceneGraph is empty",
            ha="center",
            va="center",
            transform=ax_graph.transAxes,
        )

    ax_graph.set_title(
        f"SceneGraph: {scene_graph.number_of_nodes()} nodes, "
        f"{scene_graph.number_of_edges()} edges\n"
        "Layout is NOT spatial",
    )

    ax_graph.axis("off")

    # --------------------------------------------------
    # 10. 可选：MP3D 原始 Mesh 预览图
    # --------------------------------------------------

    if (
        mesh_preview_path is not None
        and os.path.isfile(mesh_preview_path)
    ):
        mesh_image = plt.imread(mesh_preview_path)

        ax_mesh.imshow(mesh_image)
        ax_mesh.set_title("MP3D Original Mesh (reference)")

    else:
        ax_mesh.text(
            0.5, 0.5,
            "MP3D Mesh preview not provided\n\n"
            "Supply mesh_preview_path\n"
            "after rendering the original .glb",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax_mesh.transAxes,
        )

        ax_mesh.set_title("MP3D Original Mesh")

    ax_mesh.axis("off")

    # --------------------------------------------------
    # 11. 保存 PNG + CSV
    # --------------------------------------------------

    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(
        save_path,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(fig)

    csv_path = os.path.splitext(save_path)[0] + "_nodes.csv"

    fields = [
        "index",
        "node_id",
        "caption",
        "x",
        "y",
        "num_detections",
        "caption_conf",
        "degree",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(nodes)

    print(f"[DEBUG] Figure: {save_path}")
    print(f"[DEBUG] Nodes:  {csv_path}")