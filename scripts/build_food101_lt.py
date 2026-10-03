# coding: utf-8
"""Build a documented Food101-LT split into meta/train_lt.txt."""
import csv, hashlib, json, random
from collections import defaultdict
from pathlib import Path

FOOD101 = Path("/workspace/food101/food-101")
OUT     = Path("/workspace/food101_lt"); OUT.mkdir(exist_ok=True)
N_MAX, N_MIN, SEED = 750, 5, 42

rng = random.Random(SEED)
entries = (FOOD101/"meta"/"train.txt").read_text().split()
by_class = defaultdict(list)
for e in entries:
    by_class[e.split("/")[0]].append(e)
classes = sorted(by_class)
assert len(classes) == 101, len(classes)

# geometric decay: hits N_MAX and N_MIN exactly, imbalance ratio = N_MAX/N_MIN
r = (N_MIN/N_MAX) ** (1/(len(classes)-1))
counts = [max(int(round(N_MAX * r**i)), 1) for i in range(len(classes))]

ordered = list(classes)
rng.shuffle(ordered)                      # which classes become head classes

lt, rows = [], []
for rank, (cls, n) in enumerate(zip(ordered, counts)):
    lt.extend(sorted(rng.sample(sorted(by_class[cls]), n)))
    rows.append({"class": cls, "rank": rank, "n_train": n})

total = sum(counts); mean = total/len(classes)
for row in rows:
    row["group"] = "head" if row["n_train"] > mean else "tail"

split_path = FOOD101/"meta"/"train_lt.txt"
split_path.write_text("\n".join(sorted(lt)) + "\n")

with open(OUT/"class_counts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["class","rank","n_train","group"]); w.writeheader(); w.writerows(rows)

manifest = {
    "source": "Food-101 official train split",
    "profile": "geometric", "n_classes": len(classes),
    "n_max": N_MAX, "n_min": min(counts),
    "imbalance_ratio": round(N_MAX/min(counts), 2),
    "total_train_images": total, "mean_per_class": round(mean, 2),
    "head_classes": sum(r_["group"]=="head" for r_ in rows),
    "tail_classes": sum(r_["group"]=="tail" for r_ in rows),
    "head_tail_rule": "head = more than mean per-class count",
    "seed": SEED, "class_order": "shuffled(seed)",
    "test_set": "official Food-101 test, unmodified (250/class)",
    "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
}
(OUT/"manifest.json").write_text(json.dumps(manifest, indent=2))

for k, v in manifest.items(): print(f"{k:<22} {v}")
print("\nlargest:", [(r_['class'], r_['n_train']) for r_ in rows[:3]])
print("smallest:", [(r_['class'], r_['n_train']) for r_ in rows[-3:]])
print(f"\nwrote {split_path}")
