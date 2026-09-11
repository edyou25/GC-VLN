import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

p1 = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260910-133801/tmp.json"
p2 = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260910-133801/tmp2.json"

with open(p1) as f:
    a = json.load(f)

with open(p2) as f:
    b = json.load(f)

ids = sorted(set(a) & set(b), key=int)
n = len(ids)

side = math.ceil(math.sqrt(n))

# 0 = empty, 1 = fail, 2 = success
grid1 = np.zeros((side, side), dtype=int)
grid2 = np.zeros((side, side), dtype=int)

for k, ep in enumerate(ids):
    r = k // side
    c = k % side

    grid1[r, c] = 2 if a[ep]["success"] > 0 else 1
    grid2[r, c] = 2 if b[ep]["success"] > 0 else 1

sr1 = sum(a[i]["success"] > 0 for i in ids) / n
sr2 = sum(b[i]["success"] > 0 for i in ids) / n

cmap = ListedColormap([
    "lightgray",   # empty
    "red",         # fail
    "green"        # success
])

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, grid, title in [
    (axes[0], grid1, f"Run 1   SR = {sr1:.1%}"),
    (axes[1], grid2, f"Run 2   SR = {sr2:.1%}")
]:
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2)

    # grid lines
    ax.set_xticks(np.arange(-0.5, side, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, side, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)

    ax.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False
    )

    ax.set_title(title)

plt.suptitle(
    f"Episode Success Comparison  |  {n} episodes  |  grid {side} x {side}"
)

plt.tight_layout()

out = "/home/yfyou/GC-VLN/outputs/run_compare_grid.png"
os.makedirs(os.path.dirname(out), exist_ok=True)

plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()

print(f"Episodes: {n}")
print(f"Grid: {side} x {side}")
print(f"Run1 SR: {sr1:.4f} ({sr1:.1%})")
print(f"Run2 SR: {sr2:.4f} ({sr2:.1%})")
print("Saved:", out)