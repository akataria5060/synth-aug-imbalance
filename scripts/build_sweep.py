# coding: utf-8
"""
build_sweep.py

Builds Food101-LT splits at several imbalance ratios from the same Food-101
source. Everything except n_min is held fixed: same 101 classes, same seed,
same class ordering (so the class that is largest at IR=150 is also largest at
IR=5), same geometric decay, same untouched test set.

That fixed ordering is the point. It means the only thing varying across the
sweep is the imbalance ratio, so the synthetic gain measured at each ratio is
attributable to imbalance and not to which classes happen to be starved.

Writes meta/train_lt_ir{IR}.txt for each ratio, plus a manifest and per-class
counts under /workspace/food101_lt/sweep/.

Usage:
    python build_sweep.py
"""

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

FOOD101 = Path("/workspace/food101/food-101")
OUT     = Path("/workspace/food101_lt/sweep"); OUT.mkdir(parents=True, exist_ok=True)

N_MAX = 750
SEED  = 42

# (imbalance_ratio, n_min).  IR = N_MAX / n_min.
RATIOS = [
    (150, 5),
    (50, 15),
    (20, 38),
    (5, 150),
    (1, 750),
]


def build(ir, n_min, ordered, by_class):
    """Geometric decay from N_MAX to n_min across the (fixed) class ordering."""
    n = len(ordered)
    if n_min == N_MAX:                      # IR = 1, fully balanced
        counts = [N_MAX] * n
    else:
        r = (n_min / N_MAX) ** (1.0 / (n - 1))
        counts = [max(int(round(N_MAX * (r ** i))), 1) for i in range(n)]

    rng = random.Random(SEED + ir)          # per-ratio sampling, fixed ordering
    lt, rows = [], []
    for rank, (cls, k) in enumerate(zip(ordered, counts)):
        pool = sorted(by_class[cls])
        lt.extend(sorted(rng.sample(pool, k)))
        rows.append({"class": cls, "rank": rank, "n_train": k})

    total = sum(counts)
    mean  = total / n
    for row in rows:
        row["group"] = "head" if row["n_train"] > mean else "tail"

    split = FOOD101 / "meta" / f"train_lt_ir{ir}.txt"
    split.write_text("\n".join(sorted(lt)) + "\n")

    with open(OUT / f"class_counts_ir{ir}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "rank", "n_train", "group"])
        w.writeheader(); w.writerows(rows)

    return {
        "imbalance_ratio": ir,
        "n_max": N_MAX,
        "n_min": min(counts),
        "total_train_images": total,
        "mean_per_class": round(mean, 2),
        "head_classes": sum(x["group"] == "head" for x in rows),
        "tail_classes": sum(x["group"] == "tail" for x in rows),
        "split_file": str(split),
        "sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
    }


def main():
    entries = (FOOD101 / "meta" / "train.txt").read_text().split()
    by_class = defaultdict(list)
    for e in entries:
        by_class[e.split("/")[0]].append(e)
    classes = sorted(by_class)
    assert len(classes) == 101, len(classes)

    # class ordering fixed once, shared by every ratio
    ordered = list(classes)
    random.Random(SEED).shuffle(ordered)

    manifest = {
        "source": "Food-101 official train split",
        "profile": "geometric",
        "n_classes": len(classes),
        "n_max": N_MAX,
        "seed": SEED,
        "class_order": "shuffled(seed=42), identical across all ratios",
        "test_set": "official Food-101 test, unmodified (250/class)",
        "ratios": [],
    }

    print(f"{'IR':>4} {'n_min':>6} {'total':>7} {'head':>5} {'tail':>5}  sha256")
    for ir, n_min in RATIOS:
        m = build(ir, n_min, ordered, by_class)
        manifest["ratios"].append(m)
        print(f"{ir:>4} {m['n_min']:>6} {m['total_train_images']:>7} "
              f"{m['head_classes']:>5} {m['tail_classes']:>5}  {m['sha256'][:12]}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {OUT/'manifest.json'}")
    print("\nThe class ordering is identical across ratios, so the class that is")
    print("largest at IR=150 is largest at IR=5. Only the steepness changes.")


if __name__ == "__main__":
    main()
