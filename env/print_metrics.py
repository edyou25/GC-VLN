import json
import matplotlib.pyplot as plt

p1 = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260910-133801/tmp.json"
p2 = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260910-133801/tmp2.json"

with open(p1) as f:
    d1 = json.load(f)

with open(p2) as f:
    d2 = json.load(f)

fig, axes = plt.subplots(2, 1, figsize=(16, 3), sharex=True)

for ax, data, name in zip(axes, [d1, d2], ["Run 1", "Run 2"]):
    success = [int(i) for i, m in data.items() if m["success"] > 0]
    fail = [int(i) for i, m in data.items() if m["success"] <= 0]

    ax.eventplot(success, colors="green", lineoffsets=0, linelengths=1)
    ax.eventplot(fail, colors="red", lineoffsets=0, linelengths=1)

    ax.set_ylabel(name, rotation=0, ha="right", va="center")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.2)

axes[-1].set_xlabel("Episode ID")

plt.tight_layout()

out = "/home/yfyou/GC-VLN/outputs/run_compare.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", out)