# coding: utf-8
"""
fix_leakage.py

Reruns Food101-LT arm C with a leakage-free allocation.

THE PROBLEM
-----------
The original arm C derived its per-class budget from arm A's confusion matrix
computed on the TEST set, then was evaluated on that same test set. C was told
where the test errors were. It still did not beat arm B, so the null result
survives -- but the experiment as run is not clean.

THE FIX
-------
Hold out 15% of the LT training split (stratified, so every class keeps at
least one training image), train arm A on the remaining 85%, and take the
confusion matrix from that held-out portion. The test set is never touched
until final evaluation.

Arm C is then rerun on the FULL LT training split using that clean allocation,
so it is directly comparable to the existing arm A and arm B numbers.

Runs:
  Cprobe_s{42,1,2}   arm A on 85% of train, to produce the held-out matrix
  Cclean_s{42,1,2}   arm C on full train, allocation from the pooled matrix

6 runs, ~8 hours. Skips anything already finished.

Run detached:
    cd /workspace
    nohup python fix_leakage.py > fix_leakage.log 2>&1 &
    tail -f fix_leakage.log
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy, csv, json, math, random, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import swin_b

FOOD101    = Path("/workspace/food101/food-101")
LT_DIR     = Path("/workspace/food101_lt")
ARMS       = Path("/workspace/food101_lt_arms")
POOL       = Path("/workspace/food101_lt_synth")
ALIGN_CKPT = "/workspace/align/keep/align_train_20k_seed42.pth"

DEVICE = "cuda"
SEED   = 42
EPOCHS, BATCH, LR = 30, 32, 1e-4
WARMUP, WD, MIXUP, SMOOTH, EMA_DECAY, TTA_VIEWS, WORKERS = 3, 1e-4, 0.2, 0.1, 0.999, 8, 12
BUDGET   = 3800
HOLDOUT  = 0.15
SEEDS    = [42, 1, 2]


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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

CLASSES = sorted((FOOD101/"meta"/"classes.txt").read_text().split())
C2I = {c: i for i, c in enumerate(CLASSES)}
_grp = {r["class"]: r["group"] for r in csv.DictReader(open(LT_DIR/"class_counts.csv"))}
HEAD_IDS = torch.tensor([i for i, c in enumerate(CLASSES) if _grp[c] == "head"])
TAIL_IDS = torch.tensor([i for i, c in enumerate(CLASSES) if _grp[c] == "tail"])


class ItemDS(Dataset):
    """Dataset over an explicit (path, label) list, with optional synthetic items."""
    def __init__(self, items, tf, synth_items=None):
        self.items = list(items)
        self.n_real = len(self.items)
        self.n_synth = 0
        if synth_items:
            self.items += synth_items; self.n_synth = len(synth_items)
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def lt_items(split="train_lt"):
    return [(FOOD101/"images"/f"{l}.jpg", C2I[l.split("/")[0]])
            for l in (FOOD101/"meta"/f"{split}.txt").read_text().split()]


def stratified_holdout(items, frac=HOLDOUT, seed=SEED):
    """Per-class split. Every class keeps at least one training image."""
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for it in items: by_cls[it[1]].append(it)
    tr, ho = [], []
    for ci in sorted(by_cls):
        v = by_cls[ci][:]; rng.shuffle(v)
        k = min(int(round(len(v) * frac)), len(v) - 1)   # never strand a class
        ho += v[:k]; tr += v[k:]
    return tr, ho


def load_synth_list(path):
    items = []
    for ln in Path(path).read_text().splitlines():
        if not ln.strip(): continue
        img, cls = ln.split("\t")
        items.append((Path(img), C2I[cls]))
    return items


def build_swin(n_classes, init="align"):
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
        assert loaded > 0.8*len(m.state_dict())
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
def evaluate_tta(model, base, tta, n_views=TTA_VIEWS):
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
    return split_acc(c[:,0], c.any(1), labels), None


mk = lambda d, s: DataLoader(d, batch_size=BATCH, shuffle=s,
                             num_workers=WORKERS, pin_memory=True)

TE     = ItemDS(lt_items("test"), eval_tf)
TE_TTA = ItemDS(lt_items("test"), tta_tf)
te_dl, tta_dl = mk(TE, False), mk(TE_TTA, False)


def train(tag, train_items, eval_dl, synth=None, seed=SEED, note=""):
    seed_all(seed)
    tr = ItemDS(train_items, train_tf, synth)
    tr_dl = mk(tr, True)
    print(f"{tag}: real {tr.n_real}  synth {tr.n_synth}  total {len(tr)}  "
          f"seed {seed}  {note}", flush=True)

    model = build_swin(len(CLASSES)).to(DEVICE)
    ema   = EMA(model)
    crit  = nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda")
    steps = len(tr_dl)
    def lr_at(ep, it):
        t = ep + it/steps
        if t < WARMUP: return LR * t / WARMUP
        return LR * 0.5 * (1 + math.cos(math.pi * (t-WARMUP)/max(EPOCHS-WARMUP, 1)))

    best, t0 = -1.0, time.time()
    for ep in range(EPOCHS):
        model.train()
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
        m, _ = evaluate(ema.shadow, eval_dl)
        if m["top1"] > best:
            best = m["top1"]; torch.save(ema.shadow.state_dict(), ARMS/f"{tag}.pth")
        if (ep+1) % 5 == 0 or ep == EPOCHS-1:
            print(f"  ep {ep+1:>2}/{EPOCHS}  top1 {m['top1']:6.2f}  "
                  f"tail {m['top1_tail']:6.2f}  ({time.time()-t0:.0f}s)", flush=True)

    model.load_state_dict(torch.load(ARMS/f"{tag}.pth", map_location=DEVICE)); model.to(DEVICE)
    return model, round((time.time()-t0)/60, 1)


def main():
    all_items = lt_items("train_lt")
    tr85, ho15 = stratified_holdout(all_items)
    print(f"LT train {len(all_items)}  ->  {len(tr85)} train / {len(ho15)} held out")
    ho_dl = mk(ItemDS(ho15, eval_tf), False)

    # ---- phase 1: probe models, held-out confusion matrices ---------------
    cms = []
    for s in SEEDS:
        tag = f"Cprobe_s{s}"
        cm_path = ARMS/f"{tag}_holdout_confusion.npy"
        if cm_path.exists():
            print(f"skip {tag} (have matrix)")
            cms.append(np.load(cm_path)); continue
        model, mins = train(tag, tr85, ho_dl, seed=s, note="(probe, 85% of train)")
        _, cm = evaluate(model, ho_dl, want_cm=True)
        np.save(cm_path, cm); cms.append(cm)
        print(f"  {tag}: {int(cm.sum()-np.trace(cm))} held-out errors, {mins} min\n", flush=True)
        del model; torch.cuda.empty_cache()

    cm_pooled = sum(cms)
    np.save(ARMS/"Cclean_pooled_holdout_confusion.npy", cm_pooled)
    row_err = cm_pooled.sum(axis=1) - np.diag(cm_pooled)
    tot = int(cm_pooled.sum() - np.trace(cm_pooled))
    print(f"pooled held-out errors: {tot}  ({tot/len(CLASSES):.1f} per class), "
          f"{(row_err==0).sum()} classes with none\n", flush=True)

    # ---- phase 2: clean arm C allocation ----------------------------------
    pool_have = np.array([len(list((POOL/c).glob("*.jpg"))) for c in CLASSES])
    w = row_err.astype(float)
    if w.sum() == 0: raise SystemExit("no held-out errors")
    w = w / w.sum()
    a = np.floor(w * BUDGET).astype(int)
    for idx in np.argsort(-(w*BUDGET - a))[:BUDGET - int(a.sum())]: a[idx] += 1
    a = np.minimum(a, pool_have)
    deficit = BUDGET - int(a.sum())
    for ci in np.argsort(-(pool_have - a)):
        if deficit <= 0: break
        take = int(min(pool_have[ci] - a[ci], deficit)); a[ci] += take; deficit -= take
    assert a.sum() == BUDGET, a.sum()

    old = np.array([0]*len(CLASSES))
    oldp = LT_DIR/"synth_C.txt"
    if oldp.exists():
        for ln in oldp.read_text().splitlines():
            if ln.strip(): old[C2I[ln.split("\t")[1]]] += 1
        print(f"corr(old leaky alloc, clean alloc) = "
              f"{np.corrcoef(old, a)[0,1]:.3f}", flush=True)

    lines = []
    for ci, c in enumerate(CLASSES):
        for f in sorted((POOL/c).glob("*.jpg"))[:a[ci]]:
            lines.append(f"{f}\t{c}")
    synth_path = LT_DIR/"synth_Cclean.txt"
    synth_path.write_text("\n".join(lines) + "\n")
    print(f"clean arm-C list: {len(lines)} images -> {synth_path}\n", flush=True)
    synth_items = load_synth_list(synth_path)

    # ---- phase 3: clean arm C on the full training split ------------------
    for s in SEEDS:
        tag = f"Cclean_s{s}"
        if (ARMS/f"{tag}_results.json").exists():
            print(f"skip {tag} (already done)"); continue
        model, mins = train(tag, all_items, te_dl, synth=synth_items, seed=s,
                            note="(clean alloc, full train)")
        no_tta, cm = evaluate(model, te_dl, want_cm=True)
        with_tta, _ = evaluate_tta(model, te_dl, tta_dl)
        np.save(ARMS/f"{tag}_confusion.npy", cm)
        res = {"tag": tag, "init": "align", "alloc": "targeted-clean",
               "synth_list": str(synth_path),
               "n_real": len(all_items), "n_synth": len(synth_items),
               "config": {"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                          "warmup": WARMUP, "wd": WD, "mixup": MIXUP,
                          "smooth": SMOOTH, "ema": EMA_DECAY,
                          "tta_views": TTA_VIEWS, "seed": s, "budget": BUDGET,
                          "holdout_frac": HOLDOUT},
               "no_tta": no_tta, "with_tta": with_tta, "minutes": mins}
        (ARMS/f"{tag}_results.json").write_text(json.dumps(res, indent=2))
        print(f"\n{tag}  no TTA top1 {no_tta['top1']:.2f}  tail {no_tta['top1_tail']:.2f}"
              f"   TTA top1 {with_tta['top1']:.2f}   {mins} min\n", flush=True)
        del model; torch.cuda.empty_cache()

    print("leakage fix complete", flush=True)


if __name__ == "__main__":
    main()
