import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

p1 = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260910-133801/tmp.json"
p2 = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260910-133801/tmp2.json"

BIN_WIDTH = 10

with open(p1) as f:
    a = json.load(f)

with open(p2) as f:
    b = json.load(f)

A = {int(i) for i in a}
B = {int(i) for i in b}
evaluated = A & B

EP_MIN = min(evaluated)
N_EP = max(evaluated)

OUT = f"/home/yfyou/GC-VLN/outputs/r2r_d5_b{BIN_WIDTH}.png"

run1 = {
    int(i) for i, m in a.items()
    if int(i) in evaluated and m["success"] > 0
}
run2 = {
    int(i) for i, m in b.items()
    if int(i) in evaluated and m["success"] > 0
}

both = run1 & run2
run1_only = run1 - run2
run2_only = run2 - run1
run1_or_run2 = run1 | run2

subplot1_sets = {
    "both": both,
    "run1 or run2": run1_or_run2,
}

subplot2_sets = {
    "run1": run1,
    "run2": run2,
}

subplot3_sets = {
    "run1_only": run1_only,
    "run2_only": run2_only,
}

subplot4_sets = {
    "run1": run1,
    "run2_only": run2_only,
}

colors = {
    "both": "#1b7837",
    "run1 or run2": "#6a3d9a",
    "run1": "#4daf4a",
    "run2": "#377eb8",
    "run1_only": "#80b1d3",
    "run2_only": "#fb8072",
    "fig4": "#377eb8",
    "fig5": "#fb8072",
}

print(f"Evaluated: {len(evaluated)}")
print(f"Episode range: {EP_MIN} - {N_EP}")
for name, s in {
    **subplot1_sets,
    **subplot2_sets,
    **subplot3_sets,
    **subplot4_sets,
}.items():
    print(f"{name:12s}: {len(s):4d} ({len(s)/len(evaluated):.2%})")

start = EP_MIN - 0.5
edges = np.arange(start, N_EP + BIN_WIDTH + 0.5, BIN_WIDTH)
lefts = edges[:-1]
widths = np.diff(edges)

eval_counts, _ = np.histogram(
    np.array(sorted(evaluated)),
    bins=edges
)

def build_percentages(sets_dict):
    result = {}
    for name, s in sets_dict.items():
        counts, _ = np.histogram(
            np.array(sorted(s)),
            bins=edges
        )
        pct = np.divide(
            counts * 100.0,
            eval_counts,
            out=np.zeros_like(counts, dtype=float),
            where=eval_counts > 0
        )
        result[name] = pct
    return result

def draw_overlap_bars(ax, sets_dict, pct_map, title):
    names = list(sets_dict.keys())

    for i, left in enumerate(lefts):
        order = sorted(
            names,
            key=lambda n: pct_map[n][i],
            reverse=True
        )

        for name in order:
            h = pct_map[name][i]

            if h > 0:
                ax.bar(
                    left,
                    h,
                    width=widths[i],
                    align="edge",
                    color=colors[name],
                    edgecolor="none"
                )

    ax.set_ylabel("Percentage")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

    handles = [
        Patch(
            color=colors[name],
            label=f"{name} ({len(sets_dict[name])/len(evaluated):.1%})"
        )
        for name in sets_dict
    ]

    ax.legend(
        handles=handles,
        ncol=len(sets_dict),
        frameon=False
    )


def draw_single_bars(ax, values, color, title, label):
    ax.bar(
        lefts,
        values,
        width=widths,
        align="edge",
        color=color,
        edgecolor="none"
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Percentage")
    ax.yaxis.set_major_formatter(PercentFormatter(100))

    vmax = max(1.0, np.max(np.abs(values)))
    ymax = np.ceil(vmax / 5.0) * 5.0
    ax.set_ylim(-ymax, ymax)

    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        handles=[Patch(color=color, label=label)],
        ncol=1,
        frameon=False
    )

pct1 = build_percentages(subplot1_sets)
pct2 = build_percentages(subplot2_sets)
pct3 = build_percentages(subplot3_sets)
pct4 = build_percentages(subplot4_sets)


fig, axes = plt.subplots(
    3,
    1,
    figsize=(16, 14),
    sharex=True
)

draw_overlap_bars(
    axes[0],
    subplot1_sets,
    pct1,
    "Fig1: both / run1 or run2"
)

draw_overlap_bars(
    axes[1],
    subplot2_sets,
    pct2,
    "Fig2: run1 / run2"
)

draw_overlap_bars(
    axes[2],
    subplot3_sets,
    pct3,
    "Fig3: run1_only / run2_only"
)

# draw_overlap_bars(
#     axes[3],
#     subplot4_sets,
#     pct4,
#     "Fig4: run1_only / run2_only"
# )




axes[-1].set_xlim(EP_MIN, N_EP)
tick_step = 100
tick_start = int(np.ceil(EP_MIN / tick_step) * tick_step)
axes[-1].set_xticks(np.arange(tick_start, N_EP + 1, tick_step))
axes[-1].set_xlabel("Episode ID")

plt.tight_layout()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", OUT)