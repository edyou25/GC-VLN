import json

path = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260911-094330/zero_shot_vln_val_unseen_r0_w1_0(1).json"
# path = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260911-094337/zero_shot_vln_val_unseen_r0_w1_0(1).json"
# path = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260911-094345/zero_shot_vln_val_unseen_r0_w1_0(1).json"
# path = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260911-094353/zero_shot_vln_val_unseen_r0_w1_0(1).json"
path = "/home/yfyou/GC-VLN/outputs/logs/r2r/val_unseen/r2r_50_20260911-195450/zero_shot_vln_val_unseen_r0_w1_0(1).json"

with open(path, "r") as f:
    data = json.load(f)

for m in data.values():
    if not 0 <= m["spl"] <= 1:
        m["spl"] = 0
    if m["distance_to_goal"] == float("inf"):
        m["distance_to_goal"] = 10

print("Episodes:", len(data))

for key in next(iter(data.values())):
    value = sum(m[key] for m in data.values()) / len(data)
    print(f"{key:20s}: {value:.6f}")

success_ids = [ep_id for ep_id, m in data.items() if m["success"] > 0]
fail_ids = [ep_id for ep_id, m in data.items() if m["success"] <= 0]

print("\nSuccess IDs:")
print(success_ids)

print("\nFail IDs:")
print(fail_ids)