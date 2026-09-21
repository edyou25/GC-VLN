import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

p1 = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260910-133801/tmp.json"
p2 = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260910-133801/tmp2.json"

with open(p1) as f:
    a = json.load(f)

with open(p2) as f:
    b = json.load(f)

ids = sorted(set(a) & set(b), key=int)
n = len(ids)
side = math.ceil(math.sqrt(n))

# Run1 / Run2 grids
# 0 = empty, 1 = fail, 2 = success
grid1 = np.zeros((side, side), dtype=int)
grid2 = np.zeros((side, side), dtype=int)

# Success relation grid
# 0 = empty
# 1 = both fail
# 2 = both success
# 3 = only run1 success
# 4 = only run2 success
succ_grid = np.zeros((side, side), dtype=int)

# Fail relation grid
# 0 = empty
# 1 = both success
# 2 = both fail
# 3 = only run1 fail
# 4 = only run2 fail
fail_grid = np.zeros((side, side), dtype=int)

for k, ep in enumerate(ids):
    r = k // side
    c = k % side

    s1 = a[ep]["success"] > 0
    s2 = b[ep]["success"] > 0
    f1 = not s1
    f2 = not s2

    grid1[r, c] = 2 if s1 else 1
    grid2[r, c] = 2 if s2 else 1

    # success relation
    if s1 and s2:
        succ_grid[r, c] = 2
    elif s1 and (not s2):
        succ_grid[r, c] = 3
    elif (not s1) and s2:
        succ_grid[r, c] = 4
    else:
        succ_grid[r, c] = 1

    # fail relation
    if f1 and f2:
        fail_grid[r, c] = 2
    elif f1 and (not f2):
        fail_grid[r, c] = 3
    elif (not f1) and f2:
        fail_grid[r, c] = 4
    else:
        fail_grid[r, c] = 1

sr1 = sum(a[i]["success"] > 0 for i in ids) / n
sr2 = sum(b[i]["success"] > 0 for i in ids) / n

both_success = sum((a[i]["success"] > 0) and (b[i]["success"] > 0) for i in ids)
both_fail = sum((a[i]["success"] <= 0) and (b[i]["success"] <= 0) for i in ids)
run1_only_success = sum((a[i]["success"] > 0) and (b[i]["success"] <= 0) for i in ids)
run2_only_success = sum((a[i]["success"] <= 0) and (b[i]["success"] > 0) for i in ids)

# colormaps
run_colors = {
    "empty": "#f2f2f2",
    "fail": "#000000",
    "success": "#6fbf73",
}

succ_colors = {
    "empty": "#f2f2f2",
    "both_fail": "#000000",
    "both_success": "#6fbf73",
    "run1_only": "#ddf007",
    "run2_only": "#a32cb8",
}

fail_colors = {
    "empty": "#f2f2f2",
    "both_success": "#6fbf73",
    "both_fail": "#000000",
    "run1_only": "#a32cb8",
    "run2_only": "#ddf007",
}

run_cmap = ListedColormap(list(run_colors.values()))
succ_cmap = ListedColormap(list(succ_colors.values()))
fail_cmap = ListedColormap(list(fail_colors.values()))

def add_legend(ax, color_map, labels, ncol=2):
    handles = [
        Patch(color=color_map[key], label=label)
        for key, label in labels.items()
    ]

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=ncol,
        frameon=False
    )

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

plots = [
    (axes[0, 0], grid1, run_cmap, f"Run 1   SR = {sr1:.1%}"),
    (axes[0, 1], grid2, run_cmap, f"Run 2   SR = {sr2:.1%}"),
    (
        axes[1, 0],
        succ_grid,
        succ_cmap,
        f"Success Relation\nBoth={both_success}, Run1-only={run1_only_success}, Run2-only={run2_only_success}"
    ),
    (
        axes[1, 1],
        fail_grid,
        fail_cmap,
        f"Fail Relation\nBoth={both_fail}, Run1-only={run2_only_success}, Run2-only={run1_only_success}"
    ),
]

for ax, grid, cmap, title in plots:
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(cmap.colors)-1)

    ax.set_xticks(np.arange(-0.5, side, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, side, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.4)

    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=11)

# legends
add_legend(
    axes[0, 0],
    run_colors,
    {
        "success": "Success",
        "fail": "Fail",
        "empty": "Empty",
    },
    ncol=3
)

add_legend(
    axes[1, 0],
    succ_colors,
    {
        "both_success": "Both success",
        "run1_only": "Run1 success only",
        "run2_only": "Run2 success only",
        "both_fail": "Both fail",
    }
)

add_legend(
    axes[1, 1],
    fail_colors,
    {
        "both_fail": "Both fail",
        "run1_only": "Run1 fail only",
        "run2_only": "Run2 fail only",
        "both_success": "Both success",
    }
)

plt.suptitle(f"GC-VLN Run Comparison | {n} episodes | Grid {side} x {side}", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

out = "/home/yfyou/GC-VLN/outputs/run_compare.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()

print(f"Episodes: {n}")
print(f"Grid: {side} x {side}")
print(f"Run1 SR: {sr1:.4f} ({sr1:.1%})")
print(f"Run2 SR: {sr2:.4f} ({sr2:.1%})")
print(f"Both success: {both_success}")
print(f"Both fail: {both_fail}")
print(f"Run1 success only: {run1_only_success}")
print(f"Run2 success only: {run2_only_success}")
print("Saved:", out)

# ============================================================
# Figure 2: Per-scene Success Relation (11 MP3D buildings)
# ============================================================

import gzip
from pathlib import Path
from collections import defaultdict

# R2R-CE annotations: episode_id -> scene_id
annotation_file = (
    "/home/yfyou/GC-VLN/data/datasets/"
    "R2R_VLNCE_v1-2_preprocessed/"
    "val_unseen/val_unseen.json.gz"
)

with gzip.open(annotation_file, "rt") as f:
    annotations = json.load(f)

ep_to_scene = {}

for ep in annotations["episodes"]:
    ep_id = str(ep["episode_id"])
    scene_id = Path(ep["scene_id"]).stem
    ep_to_scene[ep_id] = scene_id

# Group common episodes by scene
scene_groups = defaultdict(list)

for ep_id in ids:
    if ep_id not in ep_to_scene:
        raise KeyError(f"Scene not found for episode {ep_id}")

    scene_groups[ep_to_scene[ep_id]].append(ep_id)

scenes = sorted(scene_groups)

print(f"\nNumber of buildings: {len(scenes)}")

# Figure layout: 11 scenes + 1 shared legend
ncols = 4
nrows = math.ceil((len(scenes) + 1) / ncols)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(16, 4.2 * nrows),
)

axes = np.asarray(axes).flatten()

# Use the original Success Relation colormap
# 0: empty
# 1: both fail
# 2: both success
# 3: Run1 success only
# 4: Run2 success only

for j, scene in enumerate(scenes):

    scene_ids = scene_groups[scene]

    count = len(scene_ids)
    side_i = math.ceil(math.sqrt(count))

    grid = np.zeros((side_i, side_i), dtype=int)

    both_s = 0
    both_f = 0
    r1_only = 0
    r2_only = 0

    for k, ep in enumerate(scene_ids):

        r = k // side_i
        c = k % side_i

        s1 = a[ep]["success"] > 0
        s2 = b[ep]["success"] > 0

        if s1 and s2:
            grid[r, c] = 2
            both_s += 1

        elif s1 and not s2:
            grid[r, c] = 3
            r1_only += 1

        elif not s1 and s2:
            grid[r, c] = 4
            r2_only += 1

        else:
            grid[r, c] = 1
            both_f += 1

    sr1_scene = (both_s + r1_only) / count
    sr2_scene = (both_s + r2_only) / count

    ax = axes[j]

    ax.imshow(
        grid,
        cmap=succ_cmap,
        vmin=0,
        vmax=4,
        interpolation="nearest",
    )

    # Cell boundaries
    ax.set_xticks(
        np.arange(-0.5, side_i, 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, side_i, 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="black",
        linewidth=0.35,
    )

    ax.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    ax.set_title(
        f"{scene}  |  N={count}\n"
        f"SR: {sr1_scene:.1%} / {sr2_scene:.1%}\n"
        f"Both={both_s}, R1={r1_only}, "
        f"R2={r2_only}, Fail={both_f}",
        fontsize=10,
        pad=8,
    )

    print(
        f"{scene}: "
        f"N={count}, "
        f"SR1={sr1_scene:.3f}, "
        f"SR2={sr2_scene:.3f}, "
        f"Both={both_s}, "
        f"R1only={r1_only}, "
        f"R2only={r2_only}, "
        f"BothFail={both_f}"
    )


# Shared legend in the remaining subplot
legend_ax = axes[len(scenes)]
legend_ax.axis("off")

legend_labels = {
    "both_success": "Both success",
    "run1_only": "Run1 success only",
    "run2_only": "Run2 success only",
    "both_fail": "Both fail",
}

handles = [
    Patch(
        facecolor=succ_colors[key],
        label=label,
    )
    for key, label in legend_labels.items()
]

legend_ax.legend(
    handles=handles,
    loc="center",
    frameon=False,
    fontsize=12,
    ncol=1,
)

# Hide any additional unused axes
for ax in axes[len(scenes) + 1:]:
    ax.axis("off")

fig.suptitle(
    f"GC-VLN Success Relation by Scene | "
    f"{len(scenes)} Buildings | {n} Episodes",
    fontsize=16,
    y=0.995,
)

plt.tight_layout(
    rect=[0, 0, 1, 0.975],
    h_pad=2.0,
    w_pad=1.2,
)

out_scene = (
    "/home/yfyou/GC-VLN/outputs/"
    "run_compare_by_scene.png"
)

plt.savefig(
    out_scene,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("\nSaved:", out_scene)