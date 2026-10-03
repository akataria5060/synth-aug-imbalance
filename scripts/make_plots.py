# coding: utf-8
"""All remaining quantitative figures for the paper.

Std and error bars are sample std (ddof=1) over seeds throughout.
"""

import csv, glob, json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

F_ARMS = Path("/workspace/food101_lt_arms")
U_ARMS = Path("/workspace/uec256_lt_arms")
LT_DIR = Path("/workspace/food101_lt")
U_META = Path("/workspace/uec256_meta")
OUT    = Path("/workspace/paper_figs"); OUT.mkdir(exist_ok=True)

BLUE, RED, GREY, GREEN = "#2c5f8a", "#c1502e", "#888888", "#4a8a5f"
N_REAL_150 = 15244


def load(pat, root):
    return [json.load(open(f)) for f in sorted(glob.glob(str(root/pat)))]


def ms(runs, key="top1"):
    v = np.array([r["no_tta"][key] for r in runs], dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def f150(pat):
    """Food101-LT IR-150 runs on train_lt.txt, alignment init."""
    return [r for r in load(pat, F_ARMS)
            if r["n_real"] == N_REAL_150 and r["init"] == "align"]


def curve_runs():
    """{(ir, arm): runs} for the curve on the shared sweep list (matches make_curve.py)."""
    res = {}
    for r in load("ir*_results.json", F_ARMS):
        p = r["tag"].split("_")
        res.setdefault((int(p[0][2:]), p[1]), []).append(r)
    res[(150, "A")] = f150("A_align*_results.json")
    return res


# ---------------------------------------------------------------- fig 1
def fig_allocation_bars():
    """Arms side by side on both datasets. The central null result."""
    f_arms = [("none",               f150("A_align*_results.json")),
              ("pool-capped\nlist",  f150("ir150_B_s*_results.json")),
              ("balanced\nfill",     f150("B_align*_results.json")),
              ("confusion\ntargeted", f150("Cclean_s*_results.json"))]
    u_arms = [("none",                load("A_align_s*_results.json", U_ARMS)),
              ("uniform",             load("B_align_s*_results.json", U_ARMS)),
              ("confusion\ntargeted", load("C_align_s*_results.json", U_ARMS)),
              ("fidelity\nweighted",  load("D_align_s*_results.json", U_ARMS))]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, arms, title in ((axes[0], f_arms, "Food101-LT (IR 150)"),
                            (axes[1], u_arms, "UEC-256 (IR 7.9)")):
        missing = [lab for lab, v in arms if not v]
        if missing:
            print(f"  {title}: no runs for {[m.replace(chr(10), ' ') for m in missing]}")
        arms = [(lab, v) for lab, v in arms if v]
        m  = [ms(v)[0] for _, v in arms]
        sd = [ms(v)[1] for _, v in arms]
        x = np.arange(len(arms))
        cols = [GREY] + [BLUE]*(len(arms)-1)
        ax.bar(x, m, yerr=sd, capsize=4, color=cols, width=0.6,
               edgecolor="white", linewidth=0.5)
        ax.axhline(m[0], color=GREY, ls="--", lw=0.8, zorder=0)
        for xi, (mi, si) in enumerate(zip(m, sd)):
            ax.text(xi, mi + si + 0.25, f"{mi:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([lab for lab, _ in arms], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(min(m) - 3, max(m) + 2.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("top-1 accuracy (%)")
    fig.tight_layout()
    fig.savefig(OUT/"fig_allocation_bars.png", dpi=300)
    fig.savefig(OUT/"fig_allocation_bars.pdf")
    plt.close(fig); print("wrote fig_allocation_bars")


# ---------------------------------------------------------------- fig 2
def fig_allocation_scatter():
    """The rules genuinely disagree -- they are not relabelled versions."""
    def counts(path, classes):
        c = Counter()
        for ln in Path(path).read_text().splitlines():
            if ln.strip(): c[ln.split("\t")[1]] += 1
        return np.array([c.get(k, 0) for k in classes])

    f_cls = sorted(r["class"] for r in csv.DictReader(open(LT_DIR/"class_counts.csv")))
    B = counts(LT_DIR/"synth_B.txt", f_cls)
    C = counts(LT_DIR/"synth_Cclean.txt", f_cls)

    def ucounts(p):
        c = Counter()
        for ln in Path(p).read_text().splitlines():
            if ln.strip(): c[int(ln.split("\t")[1])] += 1
        return np.array([c.get(i, 0) for i in range(256)])
    uC, uD = ucounts(U_META/"synth_C.txt"), ucounts(U_META/"synth_D.txt")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, x, y, xl, yl, ttl in (
        (axes[0], B, C, "balanced fill", "confusion targeted",
         f"Food101-LT   r = {np.corrcoef(B, C)[0,1]:.2f}"),
        (axes[1], uC, uD, "confusion targeted", "fidelity weighted",
         f"UEC-256   r = {np.corrcoef(uC, uD)[0,1]:.2f}")):
        lim = max(x.max(), y.max()) * 1.08
        ax.plot([0, lim], [0, lim], color=GREY, ls="--", lw=0.8, zorder=0)
        ax.scatter(x, y, s=16, color=BLUE, alpha=0.55, edgecolors="none")
        ax.set_xlabel(f"images allocated — {xl}", fontsize=8)
        ax.set_ylabel(f"images allocated — {yl}", fontsize=8)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlim(-lim*0.03, lim); ax.set_ylim(-lim*0.03, lim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT/"fig_allocation_scatter.png", dpi=300)
    fig.savefig(OUT/"fig_allocation_scatter.pdf")
    plt.close(fig); print("wrote fig_allocation_scatter")


# ---------------------------------------------------------------- fig 3
def fig_training_curves():
    """Tail plateaus without synthetic data; synthetic lifts it.
    Real-only = alignment-init arm A (not A_imagenet)."""
    A = f150("A_align*_results.json")
    B = f150("B_align*_results.json")
    if not A or not B: print("skip training curves"); return

    def band(runs, key):
        h = np.array([[e[key] for e in r["history"]] for r in runs], dtype=float)
        return h.mean(0), (h.std(0, ddof=1) if len(runs) > 1 else np.zeros(h.shape[1]))

    ep = np.arange(1, len(A[0]["history"]) + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for runs, col, lab in ((A, GREY, "real only"),
                           (B, BLUE, "+3,800 synthetic (balanced fill)")):
        for key, ls in (("top1_head", "--"), ("top1_tail", "-")):
            m, s = band(runs, key)
            ax.plot(ep, m, ls, color=col, lw=1.5,
                    label=f"{lab} — {'head' if 'head' in key else 'tail'}")
            ax.fill_between(ep, m-s, m+s, color=col, alpha=0.15, linewidth=0)
    ax.set_xlabel("epoch"); ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title("Food101-LT, IR 150", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/"fig_training_curves.png", dpi=300)
    fig.savefig(OUT/"fig_training_curves.pdf")
    plt.close(fig); print("wrote fig_training_curves")


# ---------------------------------------------------------------- fig 4
def fig_fidelity_vs_error():
    """Fidelity and confusion are independent axes."""
    from scipy.stats import spearmanr
    sc = json.loads((U_META/"fidelity.json").read_text())
    n = len(sc)
    m  = np.array([sc[str(i)]["mean"] for i in range(n)])
    rs = np.array([sc[str(i)]["real_spread"] for i in range(n)])
    ratio = m / rs
    cm = np.load(U_META/"A_align_val_confusion_pooled.npy")
    err = cm.sum(1) - np.diag(cm)
    rho = spearmanr(ratio, err).correlation

    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.scatter(ratio, err, s=18, color=BLUE, alpha=0.55, edgecolors="none")
    ax.axvline(0.2, color=RED, ls="--", lw=0.9)
    ax.text(0.19, err.max()*0.95, f"{(ratio<0.2).sum()} classes", ha="right",
            fontsize=8, color=RED)
    ax.set_xlabel("generation fidelity")
    ax.set_ylabel("held-out errors (pooled, 3 seeds)")
    ax.set_title(f"UEC-256:  Spearman $\\rho$ = {rho:.3f}", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/"fig_fidelity_vs_error.png", dpi=300)
    fig.savefig(OUT/"fig_fidelity_vs_error.pdf")
    plt.close(fig); print("wrote fig_fidelity_vs_error")


# ---------------------------------------------------------------- fig 5
def fig_headtail_gap():
    """Where the gain lands: tail, not head. Runs on the shared sweep list."""
    res = curve_runs()
    irs, hg, tg = [], [], []
    for ir in sorted({k[0] for k in res}, reverse=True):
        A, B = res.get((ir, "A")), res.get((ir, "B"))
        if not A or not B:
            print(f"  headtail: IR {ir} incomplete -- skipped"); continue
        irs.append(ir)
        hg.append(ms(B, "top1_head")[0] - ms(A, "top1_head")[0])
        tg.append(ms(B, "top1_tail")[0] - ms(A, "top1_tail")[0])

    x = np.arange(len(irs)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.axhline(0, color=GREY, lw=0.8)
    ax.bar(x - w/2, hg, w, label="head classes", color=GREY)
    ax.bar(x + w/2, tg, w, label="tail classes", color=BLUE)
    for xi, v in zip(x - w/2, hg):
        ax.text(xi, v + (0.15 if v >= 0 else -0.45), f"{v:+.1f}",
                ha="center", fontsize=7.5)
    for xi, v in zip(x + w/2, tg):
        ax.text(xi, v + 0.2, f"{v:+.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([f"IR {i}" for i in irs], fontsize=8)
    ax.set_ylabel("top-1 gain from synthetic (pp)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/"fig_headtail_gap.png", dpi=300)
    fig.savefig(OUT/"fig_headtail_gap.pdf")
    plt.close(fig); print("wrote fig_headtail_gap")


# ---------------------------------------------------------------- summary
def print_allocation_summary():
    """Paste-ready numbers for the allocation tables, plus seed-paired diffs."""
    print("\nallocation arms (top-1 / tail, mean ± sample std):")
    f_arms = [("A none", "A_align*"), ("pool-capped", "ir150_B_s*"),
              ("B balanced", "B_align*"), ("C leaky", "C_align*"), ("C' clean", "Cclean_s*")]
    for lab, pat in f_arms:
        v = f150(pat + "_results.json")
        if not v: print(f"  F101 {lab:<11} (no runs)"); continue
        t, ts = ms(v); tl, tls = ms(v, "top1_tail")
        print(f"  F101 {lab:<11} n={len(v)}  {t:.2f} ± {ts:.2f}   tail {tl:.2f} ± {tls:.2f}")
    for lab in "ABCD":
        v = load(f"{lab}_align_s*_results.json", U_ARMS)
        t, ts = ms(v)
        print(f"  UEC  {lab:<11} n={len(v)}  {t:.2f} ± {ts:.2f}")

    def by_seed(runs):
        return {r["config"]["seed"]: r["no_tta"]["top1"] for r in runs}
    base = by_seed(f150("B_align*_results.json"))
    print("\nseed-paired differences vs balanced fill (Food101-LT):")
    for lab, pat in (("pool-capped", "ir150_B_s*"), ("C leaky", "C_align*"), ("C' clean", "Cclean_s*")):
        other = by_seed(f150(pat + "_results.json"))
        d = [round(other[s] - base[s], 2) for s in sorted(base) if s in other]
        if d: print(f"  {lab:<9} {d}   mean {np.mean(d):+.2f}")


if __name__ == "__main__":
    for fn in (fig_allocation_bars, fig_allocation_scatter, fig_training_curves,
               fig_fidelity_vs_error, fig_headtail_gap, print_allocation_summary):
        try: fn()
        except Exception as e: print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")
    print("\ndone")
