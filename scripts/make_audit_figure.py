# coding: utf-8
"""Rebuild Fig. 6 (near-duplicate UEC-256 class pairs, real training images).

Rows are labelled with the dish name and the official UEC-256 class number.
Images are box crops from the training partition, using the same
photograph-grouped 70/15/15 split (seed 42) as the experiments.

    cd /workspace
    python make_audit_figure.py --candidates   # six small previews, seeds 1-6
    python make_audit_figure.py --seed N       # final figure from sample N
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42      # embed fonts as TrueType
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

UEC  = Path("/root/data/UECFood256/UECFOOD256")
META = Path("/workspace/uec256_meta")
OUT  = Path("/workspace/paper_figs")
SEED = 42
PAIRS = [(1, 133), (39, 66), (75, 76)]   # zero-based indices of the audited pairs
N_IMG = 4
SIZE  = 256

prompts = json.loads((META/"prompts.json").read_text())
cats = sorted([d for d in UEC.iterdir() if d.is_dir() and d.name.isdigit()],
              key=lambda d: int(d.name))
c2i = {d.name: i for i, d in enumerate(cats)}

# ---- every valid box, as in the training code
inst = []
for d in cats:
    bb = d/"bb_info.txt"
    if not bb.exists():
        continue
    for line in bb.read_text().splitlines()[1:]:
        p = line.split()
        if len(p) != 5:
            continue
        img_id, x1, y1, x2, y2 = p
        box = (int(x1), int(y1), int(x2), int(y2))
        if (d/f"{img_id}.jpg").exists() and box[2] > box[0] and box[3] > box[1]:
            inst.append((c2i[d.name], img_id, d, box))

# ---- photograph-grouped split, identical to the experiments
rng = random.Random(SEED)
by_stem = defaultdict(list)
for x in inst:
    by_stem[x[1]].append(x)
stems_by_cat = defaultdict(list)
for stem, items in by_stem.items():
    stems_by_cat[min(i[0] for i in items)].append(stem)
train = []
for cat in sorted(stems_by_cat):
    s = stems_by_cat[cat][:]
    rng.shuffle(s)
    for stem in s[:int(.70 * len(s))]:
        train += by_stem[stem]
assert len(train) == 22041, f"training split has {len(train)} crops, expected 22041"

# ---- figure
rows = [c for pair in PAIRS for c in pair]


def draw(sample_seed, path, dpi):
    pick = random.Random(sample_seed)
    fig, axes = plt.subplots(len(rows), N_IMG, figsize=(6.0, 8.6))
    for r, cls in enumerate(rows):
        pool = sorted([x for x in train if x[0] == cls], key=lambda x: (x[1], x[3]))
        chosen = pick.sample(pool, N_IMG)
        for c, (_, img_id, d, box) in enumerate(chosen):
            im = Image.open(d/f"{img_id}.jpg").convert("RGB").crop(box)
            axes[r, c].imshow(ImageOps.fit(im, (SIZE, SIZE)))
            axes[r, c].axis("off")
        name = prompts[str(cls)]["name"]
        axes[r, 0].text(-0.08, 0.5, f"{name}\n(class {cls + 1})",
                        transform=axes[r, 0].transAxes, ha="right", va="center",
                        fontsize=9)
    fig.subplots_adjust(left=0.27, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.04, hspace=0.10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


ap = argparse.ArgumentParser()
ap.add_argument("--candidates", action="store_true")
ap.add_argument("--seed", type=int, default=None)
args = ap.parse_args()

if args.candidates:
    for sd in range(1, 7):
        out = Path("/workspace")/f"audit_candidate_{sd}.png"
        draw(sd, out, dpi=110)
        print("wrote", out)
elif args.seed is not None:
    for ext in ("pdf", "png"):
        draw(args.seed, OUT/f"fig_prompt_audit.{ext}", dpi=300)
    print(f"wrote {OUT/'fig_prompt_audit.pdf'} from sample {args.seed}")
    for cls in rows:
        print(f"  class {cls + 1:>3}: {prompts[str(cls)]['name']}")
else:
    ap.error("use --candidates or --seed N")
