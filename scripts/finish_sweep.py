# coding: utf-8
"""
finish_sweep.py

Finishes the Food101-LT imbalance sweep. Skips any run whose results JSON
already exists, so it is safe to re-run after a crash.

Run from a terminal, detached:

    cd /workspace
    nohup python finish_sweep.py > finish_sweep.log 2>&1 &
    tail -f finish_sweep.log

Current setting: IR=150 arm B with UNIFORM allocation (synth_fixed3800.txt),
seeds 42/1/2, on train_lt.txt -- the same split as the A_align / B_align /
C_align / Cclean allocation arms. This gives:
  * the IR=150 curve point with the same allocation rule as IR 50/20/5
    (arm A for that point = the existing A_align runs, same split), and
  * a uniform arm for the IR=150 allocation comparison on identical data.

Tags written: ir150_B_s42, ir150_B_s1, ir150_B_s2  (no existing file is touched).
Every run verifies the sha256 of its split file before training and records
it in the results JSON.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy, csv, hashlib, json, math, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import swin_b

# ============================================================================
# Config -- identical to sweep.ipynb
# ============================================================================

FOOD101    = Path("/workspace/food101/food-101")
LT_DIR     = Path("/workspace/food101_lt")
ARMS       = Path("/workspace/food101_lt_arms")
POOL       = Path("/workspace/food101_lt_synth")
SWEEP      = Path("/workspace/food101_lt/sweep")
ALIGN_CKPT = "/workspace/align/keep/align_train_20k_seed42.pth"

DEVICE = "cuda"
SEED   = 42
EPOCHS, BATCH, LR = 30, 32, 1e-4
WARMUP, WD, MIXUP, SMOOTH, EMA_DECAY, TTA_VIEWS, WORKERS = 3, 1e-4, 0.2, 0.1, 0.999, 8, 12
FIXED_BUDGET = 3800

# what to run:  (imbalance_ratio, arm, seed)
# IR=150 arm A is NOT rerun: the existing A_align, A_align_s1, A_align_s2
# runs are on the same split (train_lt.txt) and serve as the A arm.
RATIOS = [150]
ARMSET = ["B"]
SEEDS  = [42, 1, 2]

_SWEEP_SPLIT = "train_lt"

SWEEP_M = json.loads((SWEEP/"manifest.json").read_text())
LT_M    = json.loads((LT_DIR/"manifest.json").read_text())


def split_name(ir):
    """IR=150 uses the original allocation-experiment split; others the sweep's."""
    return "train_lt" if ir == 150 else f"train_lt_ir{ir}"


def expected_sha(ir):
    if ir == 150:
        return LT_M["split_sha256"]
    return {m["imbalance_ratio"]: m["sha256"] for m in SWEEP_M["ratios"]}[ir]


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ============================================================================
# Data
# ============================================================================

MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]

train_tf = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(0.3,0.3,0.3,0.05),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD),
    transforms.RandomErasing(p=0.25)])
eval_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
tta_tf = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD)])


class Food101LT(Dataset):
    def __init__(self, split, tf, synth_items=None):
        self.classes = sorted((FOOD101/"meta"/"classes.txt").read_text().split())
        self.c2i = {c: i for i, c in enumerate(self.classes)}
        self.tf = tf
        self.items = [(FOOD101/"images"/f"{l}.jpg", self.c2i[l.split("/")[0]])
                      for l in (FOOD101/"meta"/f"{split}.txt").read_text().split()]
        self.n_real = len(self.items)
        self.n_synth = 0
        if synth_items:
            self.items.extend(synth_items); self.n_synth = len(synth_items)
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def load_synth_list(path, c2i):
    if not path: return None
    items = []
    for ln in Path(path).read_text().splitlines():
        if not ln.strip(): continue
        img, cls = ln.split("\t")
        items.append((Path(img), c2i[cls]))
    return items


seed_all()
CLASSES = sorted((FOOD101/"meta"/"classes.txt").read_text().split())
C2I = {c: i for i, c in enumerate(CLASSES)}

HEAD_IDS = TAIL_IDS = None

def set_head_tail(ir):
    """head/tail membership changes with the imbalance ratio."""
    global HEAD_IDS, TAIL_IDS
    f = (LT_DIR/"class_counts.csv") if ir == 150 else (SWEEP/f"class_counts_ir{ir}.csv")
    g = {r["class"]: r["group"] for r in csv.DictReader(open(f))}
    HEAD_IDS = torch.tensor([i for i, c in enumerate(CLASSES) if g[c] == "head"])
    TAIL_IDS = torch.tensor([i for i, c in enumerate(CLASSES) if g[c] == "tail"])
    return len(HEAD_IDS), len(TAIL_IDS)

set_head_tail(150)

te, te_tta = Food101LT("test", eval_tf), Food101LT("test", tta_tf)
mk = lambda d, s: DataLoader(d, batch_size=BATCH, shuffle=s,
                             num_workers=WORKERS, pin_memory=True)
te_dl, tta_dl = mk(te, False), mk(te_tta, False)


# ============================================================================
# Model
# ============================================================================

def build_swin(n_classes, init):
    m = swin_b(weights="IMAGENET1K_V1" if init == "imagenet" else None)
    dim = m.head.in_features
    m.head = nn.Identity()
    if init == "align":
        sd = torch.load(ALIGN_CKPT, map_location="cpu")
        sd = sd.get("state_dict", sd.get("model", sd))
        ext = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
        missing, _ = m.load_state_dict(ext, strict=False)
        loaded = len(m.state_dict()) - len(missing)
        print(f"alignment init: {loaded}/{len(m.state_dict())} tensors", flush=True)
        assert loaded > 0.8*len(m.state_dict()), "prefix mismatch"
    m.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, n_classes))
    return m


class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(m.detach(), alpha=1-self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)


def mixup(x, y, alpha=MIXUP):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam


# ============================================================================
# Evaluation
# ============================================================================

def split_acc(c1, c5, labels):
    head_m = torch.isin(labels, HEAD_IDS); tail_m = ~head_m
    def pct(mask, c):
        n = int(mask.sum().item())
        return round(100.0 * c[mask].sum().item() / max(n, 1), 2)
    all_m = torch.ones_like(head_m)
    return {"top1": pct(all_m, c1), "top5": pct(all_m, c5),
            "top1_head": pct(head_m, c1), "top1_tail": pct(tail_m, c1),
            "top5_head": pct(head_m, c5), "top5_tail": pct(tail_m, c5)}


@torch.no_grad()
def evaluate(model, loader, want_cm=False):
    model.eval(); c1, c5, ls, pr = [], [], [], []
    for x, y in loader:
        top5 = model(x.to(DEVICE, non_blocking=True)).topk(5, 1, True, True)[1].cpu()
        c = top5.eq(y.view(-1,1))
        c1.append(c[:,0]); c5.append(c.any(1)); ls.append(y); pr.append(top5[:,0])
    labels = torch.cat(ls)
    out = split_acc(torch.cat(c1), torch.cat(c5), labels)
    if not want_cm: return out, None
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(cm, (labels.numpy(), torch.cat(pr).numpy()), 1)
    return out, cm


@torch.no_grad()
def evaluate_tta(model, base, tta, n_views=TTA_VIEWS, want_cm=False):
    model.eval(); probs = labels = None
    for v in range(n_views):
        loader = base if v == 0 else tta
        ps, ls = [], []
        for x, y in loader:
            ps.append(F.softmax(model(x.to(DEVICE, non_blocking=True)), 1).cpu()); ls.append(y)
        vp = torch.cat(ps)
        probs = vp if probs is None else probs + vp
        if labels is None: labels = torch.cat(ls)
    top5 = probs.topk(5, 1, True, True)[1]
    c = top5.eq(labels.view(-1,1))
    out = split_acc(c[:,0], c.any(1), labels)
    if not want_cm: return out, None
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(cm, (labels.numpy(), top5[:,0].numpy()), 1)
    return out, cm


# ============================================================================
# Train
# ============================================================================

def run_arm(tag, init="align", synth=None, alloc="none",
            expect_synth=None, seed=SEED, split_sha=None):
    seed_all(seed)
    tr = Food101LT(_SWEEP_SPLIT, train_tf, load_synth_list(synth, C2I))
    if expect_synth is not None and tr.n_synth != expect_synth:
        raise SystemExit(f"budget mismatch: {tr.n_synth} synthetic, expected {expect_synth}")
    tr_dl = mk(tr, True)
    print(f"{tag}:  split {_SWEEP_SPLIT}   real {tr.n_real}   "
          f"synth {tr.n_synth}   total {len(tr)}   seed {seed}", flush=True)

    model = build_swin(len(CLASSES), init).to(DEVICE)
    ema   = EMA(model)
    crit  = nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda")

    steps = len(tr_dl)
    def lr_at(ep, it):
        t = ep + it/steps
        if t < WARMUP: return LR * t / WARMUP
        return LR * 0.5 * (1 + math.cos(math.pi * (t-WARMUP)/max(EPOCHS-WARMUP, 1)))

    history, best, t0 = [], -1.0, time.time()
    for ep in range(EPOCHS):
        model.train(); run, seen = 0.0, 0
        for it, (x, y) in enumerate(tr_dl):
            for g in opt.param_groups: g["lr"] = lr_at(ep, it)
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            xm, ya, yb, lam = mixup(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                o = model(xm)
                loss = lam*crit(o, ya) + (1-lam)*crit(o, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ema.update(model)
            run += loss.item()*y.size(0); seen += y.size(0)

        m, _ = evaluate(ema.shadow, te_dl)
        history.append({"epoch": ep+1, "loss": round(run/seen, 4), **m})
        print(f"  ep {ep+1:>2}/{EPOCHS}  loss {run/seen:.4f}  top1 {m['top1']:6.2f}  "
              f"head {m['top1_head']:6.2f}  tail {m['top1_tail']:6.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if m["top1"] > best:
            best = m["top1"]; torch.save(ema.shadow.state_dict(), ARMS/f"{tag}.pth")

    model.load_state_dict(torch.load(ARMS/f"{tag}.pth", map_location=DEVICE)); model.to(DEVICE)
    no_tta,  cm  = evaluate(model, te_dl, want_cm=True)
    with_tta, cmt = evaluate_tta(model, te_dl, tta_dl, want_cm=True)
    np.save(ARMS/f"{tag}_confusion.npy", cm)
    np.save(ARMS/f"{tag}_confusion_tta.npy", cmt)

    res = {"tag": tag, "init": init, "alloc": alloc, "synth_list": str(synth),
           "n_real": tr.n_real, "n_synth": tr.n_synth,
           "config": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "warmup": WARMUP,
                      "wd": WD, "mixup": MIXUP, "smooth": SMOOTH, "ema": EMA_DECAY,
                      "tta_views": TTA_VIEWS, "seed": seed, "budget": FIXED_BUDGET,
                      "split": _SWEEP_SPLIT, "split_sha256": split_sha},
           "no_tta": no_tta, "with_tta": with_tta,
           "minutes": round((time.time()-t0)/60, 1), "history": history}
    (ARMS/f"{tag}_results.json").write_text(json.dumps(res, indent=2))

    print(f"\n{tag}", flush=True)
    for name, m in (("no TTA", no_tta), ("8-view TTA", with_tta)):
        print(f"  {name:<12} top1 {m['top1']:6.2f}   head {m['top1_head']:6.2f}   "
              f"tail {m['top1_tail']:6.2f}   top5 {m['top5']:6.2f}", flush=True)
    print(f"  {res['minutes']} min\n", flush=True)
    return res


def run_sweep(ir, arm, seed, synth_path):
    global _SWEEP_SPLIT
    nh, nt = set_head_tail(ir)
    _SWEEP_SPLIT = split_name(ir)
    sha = file_sha(FOOD101/"meta"/f"{_SWEEP_SPLIT}.txt")
    if sha != expected_sha(ir):
        raise SystemExit(f"split sha mismatch for IR {ir}: {sha} != {expected_sha(ir)}")
    print(f"--- IR {ir}  arm {arm}  seed {seed}   split {_SWEEP_SPLIT} "
          f"(sha {sha[:12]})   head {nh} / tail {nt} ---", flush=True)
    try:
        return run_arm(f"ir{ir}_{arm}_s{seed}", init="align",
                       synth=(synth_path if arm == "B" else None),
                       alloc=("none" if arm == "A" else "uniform"),
                       expect_synth=(FIXED_BUDGET if arm == "B" else None),
                       seed=seed, split_sha=sha)
    finally:
        _SWEEP_SPLIT = "train_lt"
        set_head_tail(150)


# ============================================================================
# Main
# ============================================================================

def main():
    synth_path = SWEEP/"synth_fixed3800.txt"
    if not synth_path.exists():
        raise SystemExit(f"missing {synth_path} -- run the list-building cell first")
    n = len(synth_path.read_text().splitlines())
    assert n == FIXED_BUDGET, f"synth list has {n}, expected {FIXED_BUDGET}"

    todo = []
    for ir in RATIOS:
        for arm in ARMSET:
            for s in SEEDS:
                tag = f"ir{ir}_{arm}_s{s}"
                if (ARMS/f"{tag}_results.json").exists():
                    print(f"skip {tag} (already done)")
                    continue
                todo.append((ir, arm, s))

    if not todo:
        print("nothing to do")
        return

    real = {m["imbalance_ratio"]: m["total_train_images"] for m in SWEEP_M["ratios"]}
    est = sum(real[ir] / 15244 * 82 for ir, _, _ in todo)
    print(f"\n{len(todo)} runs queued, est. {est/60:.1f} hours\n", flush=True)

    for ir, arm, s in todo:
        run_sweep(ir, arm, s, synth_path)

    print("sweep complete", flush=True)


if __name__ == "__main__":
    main()
