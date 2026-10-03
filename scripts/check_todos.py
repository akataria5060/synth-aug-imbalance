# coding: utf-8
"""Resolve the last three verifiable TODOs in sn-article.tex. CPU only, seconds.

    cd /workspace && python check_todos.py
"""
import json, random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

META = Path("/workspace/uec256_meta")
UEC  = Path("/root/data/UECFood256/UECFOOD256")
SEED = 42

# ---- 1. fidelity extremes: raw mean vs normalised (mean / real_spread)
sc = json.loads((META/"fidelity.json").read_text())
prompts = json.loads((META/"prompts.json").read_text())
name = lambda i: prompts[str(i)]["name"]
n = len(sc)
raw  = np.array([sc[str(i)]["mean"] for i in range(n)])
norm = raw / np.array([sc[str(i)]["real_spread"] for i in range(n)])
print("1. FIDELITY")
for lab, v in (("raw mean", raw), ("normalised", norm)):
    lo, hi = int(v.argmin()), int(v.argmax())
    print(f"   {lab:<11} min {v.min():+.3f} ({name(lo)})   max {v.max():+.3f} ({name(hi)})   "
          f"median {np.median(v):.3f}   below 0.2: {(v < 0.2).sum()}")

# ---- 2. audit pairs: confusions between the two classes, pooled over 3 seeds
cm = np.load(META/"A_align_val_confusion_pooled.npy")
print("\n2. AUDIT PAIRS (pooled validation matrix, 3 seeds; 1-based class ids)")
for a, b in ((1, 133), (39, 66), (75, 76), (138, 139)):
    i, j = a - 1, b - 1
    print(f"   [{a}] {name(i)[:22]:<22} / [{b}] {name(j)[:22]:<22}  "
          f"{int(cm[i, j])} + {int(cm[j, i])} = {int(cm[i, j] + cm[j, i])}")

# ---- 3. UEC per-class instance counts: full dataset and training partition
cats = sorted([d for d in UEC.iterdir() if d.is_dir() and d.name.isdigit()],
              key=lambda d: int(d.name))
c2i = {d.name: i for i, d in enumerate(cats)}
inst = []
for d in cats:
    bb = d/"bb_info.txt"
    if not bb.exists(): continue
    for line in bb.read_text().splitlines()[1:]:
        p = line.split()
        if len(p) != 5: continue
        img_id, x1, y1, x2, y2 = p
        box = (int(x1), int(y1), int(x2), int(y2))
        if (d/f"{img_id}.jpg").exists() and box[2] > box[0] and box[3] > box[1]:
            inst.append((c2i[d.name], img_id))

rng = random.Random(SEED)
by_stem = defaultdict(list)
for x in inst: by_stem[x[1]].append(x)
stems_by_cat = defaultdict(list)
for stem, items in by_stem.items():
    stems_by_cat[min(i[0] for i in items)].append(stem)
tr = []
for cat in sorted(stems_by_cat):
    s = stems_by_cat[cat][:]; rng.shuffle(s)
    for x in s[:int(.70 * len(s))]: tr += by_stem[x]

print("\n3. UEC CLASS SIZES")
for lab, items in (("full dataset", inst), ("train split", tr)):
    c = np.array(list(Counter(i[0] for i in items).values()))
    print(f"   {lab:<13} n={len(items):>6}  min {c.min()}  median {np.median(c):.0f}  "
          f"max {c.max()}  IR {c.max()/c.min():.1f}  classes<120: {(c < 120).sum()}")
