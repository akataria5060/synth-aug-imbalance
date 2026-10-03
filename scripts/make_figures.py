# coding: utf-8
"""Image-grid figures for the paper, assembled from the generated pools."""

import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

FOOD101   = Path("/workspace/food101/food-101")
POOL_F    = Path("/workspace/food101_lt_synth")
UEC_ROOT  = Path("/root/data/UECFood256/UECFOOD256")
POOL_U    = Path("/workspace/uec256_synth")
META      = Path("/workspace/uec256_meta")
OUT       = Path("/workspace/paper_figs"); OUT.mkdir(exist_ok=True)
PANELS    = OUT/"panels"; PANELS.mkdir(exist_ok=True)
DPI = 300


def uec_classes():
    cats = sorted([d for d in UEC_ROOT.iterdir() if d.is_dir() and d.name.isdigit()],
                  key=lambda d: int(d.name))
    names = {}
    for ln in (UEC_ROOT/"category.txt").read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 2: names[int(p[0])] = p[1].strip()
    return cats, [names.get(int(d.name), f"class_{d.name}") for d in cats]


def uec_real_examples(cat_dir, n=4):
    out = []
    for line in (cat_dir/"bb_info.txt").read_text().splitlines()[1:]:
        p = line.split()
        if len(p) != 5: continue
        img_id, x1, y1, x2, y2 = p
        f = cat_dir/f"{img_id}.jpg"
        box = (int(x1), int(y1), int(x2), int(y2))
        if f.exists() and box[2] > box[0] and box[3] > box[1]:
            out.append(Image.open(f).convert("RGB").crop(box))
        if len(out) >= n: break
    return out


def grid(rows, row_labels, path, title=None, cell=2.0):
    nr = len(rows); nc = max(len(r) for r in rows)
    fig, ax = plt.subplots(nr, nc, figsize=(cell*nc, cell*nr + 0.4))
    ax = np.atleast_2d(ax)
    if nc == 1: ax = ax.reshape(nr, 1)
    for i, row in enumerate(rows):
        for j in range(nc):
            a = ax[i, j]; a.axis("off")
            if j < len(row): a.imshow(row[j].resize((256, 256)))
        ax[i, 0].axis("on")
        ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])
        for s in ax[i, 0].spines.values(): s.set_visible(False)
        ax[i, 0].set_ylabel(row_labels[i], rotation=0, ha="right", va="center",
                            fontsize=8, labelpad=8)
    if title: fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(str(path) + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(str(path) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}.png/.pdf")


def fig_food101_samples():
    rows, labels = [], []
    for c in ["sashimi", "steak", "caesar_salad", "carrot_cake"]:
        fs = sorted((POOL_F/c).glob("*.jpg"))[:4]
        if not fs: continue
        rows.append([Image.open(f) for f in fs]); labels.append(c.replace("_", " "))
    grid(rows, labels, OUT/"fig_food101_samples",
         title="Stable Diffusion v1.5 samples, Food101-LT classes")


def fig_uec_good_bad(n_show=3):
    cats, names = uec_classes()
    scores = json.loads((META/"fidelity.json").read_text())
    m  = np.array([scores[str(i)]["mean"] for i in range(len(cats))])
    rs = np.array([scores[str(i)]["real_spread"] for i in range(len(cats))])
    ratio = m / rs
    order = np.argsort(ratio)
    for tag, idxs in (("worst", order[:n_show]), ("best", order[-n_show:][::-1])):
        rows, labels = [], []
        for ci in idxs:
            real = uec_real_examples(cats[ci], 3)
            syn  = [Image.open(f) for f in sorted((POOL_U/f"{ci:03d}").glob("*.jpg"))[:3]]
            if not real or not syn: continue
            rows.append(real); labels.append(f"{names[ci][:22]}\nREAL")
            rows.append(syn);  labels.append(f"fidelity {ratio[ci]:.2f}\nGENERATED")
        grid(rows, labels, OUT/f"fig_uec_{tag}_fidelity",
             title=f"UEC-256: {tag} generation fidelity")


def fig_prompt_audit():
    """Three pairs. 138/139 dropped — the distinction is not visible."""
    cats, names = uec_classes()
    rows, labels = [], []
    for a, b in [(1, 133), (39, 66), (75, 76)]:
        for ci in (a, b):
            ex = uec_real_examples(cats[ci], 4)
            if not ex: continue
            rows.append(ex); labels.append(f"[{ci}] {names[ci][:24]}")
    grid(rows, labels, OUT/"fig_prompt_audit",
         title="Near-duplicate UEC-256 class pairs (real images)")


def fig_fidelity_hist():
    cats, _ = uec_classes()
    scores = json.loads((META/"fidelity.json").read_text())
    m  = np.array([scores[str(i)]["mean"] for i in range(len(cats))])
    rs = np.array([scores[str(i)]["real_spread"] for i in range(len(cats))])
    ratio = m / rs
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.hist(ratio, bins=40, color="#4a7ba7", edgecolor="white", linewidth=0.4)
    ax.axvline(0.2, color="#c44", ls="--", lw=1)
    ax.text(0.19, ax.get_ylim()[1]*0.9, f"{(ratio<0.2).sum()} classes below 0.2",
            fontsize=8, color="#c44", ha="right")
    ax.set_xlabel("generation fidelity (cosine to real class centroid / real spread)")
    ax.set_ylabel("classes")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/"fig_fidelity_hist.png", dpi=DPI)
    fig.savefig(OUT/"fig_fidelity_hist.pdf")
    plt.close(fig)
    print("wrote fig_fidelity_hist.png/.pdf")


def dump_individual_panels():
    cats, _ = uec_classes()
    for c in ["sashimi", "steak", "caesar_salad", "carrot_cake", "donuts", "pork_chop"]:
        for k, f in enumerate(sorted((POOL_F/c).glob("*.jpg"))[:4]):
            Image.open(f).resize((512,512)).save(PANELS/f"food101_{c}_{k}.png")
    for ci in (1, 39, 66, 75, 76, 133, 138, 139, 34, 53):
        for k, im in enumerate(uec_real_examples(cats[ci], 3)):
            im.resize((512,512)).save(PANELS/f"uec{ci:03d}_real_{k}.png")
        for k, f in enumerate(sorted((POOL_U/f"{ci:03d}").glob("*.jpg"))[:3]):
            Image.open(f).resize((512,512)).save(PANELS/f"uec{ci:03d}_gen_{k}.png")
    print(f"wrote panels to {PANELS}")


if __name__ == "__main__":
    fig_food101_samples()
    fig_uec_good_bad()
    fig_prompt_audit()
    fig_fidelity_hist()
    dump_individual_panels()
    print("\ndone")
