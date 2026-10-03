# coding: utf-8
"""Headline figure: synthetic-augmentation gain vs training-set imbalance ratio.

Every curve point uses UNIFORM allocation of a fixed 3,800-image budget.

  IR 50 / 20 / 5 : ir{IR}_A_s*  vs  ir{IR}_B_s*      (sweep splits)
  IR 150         : A_align*     vs  ir150_B_s*       (train_lt.txt, the
                   allocation-experiment split; same per-class sizes as the
                   sweep's IR-150 file, independent image draw)

The balanced-fill runs (B_align*) are deliberately NOT used here -- they
belong to the allocation comparison, not the curve.

Mean +/- sample std (ddof=1) over seeds. Top-1, no TTA.
"""

import csv, glob, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

F_ARMS = Path("/workspace/food101_lt_arms")
U_ARMS = Path("/workspace/uec256_lt_arms")
SWEEP  = Path("/workspace/food101_lt/sweep")
OUT    = Path("/workspace/paper_figs"); OUT.mkdir(exist_ok=True)
FIXED_BUDGET = 3800
N_REAL_150   = 15244


def load(path):
    return json.load(open(path))


def collect_food101():
    """{(ir, arm): [runs]} -- uniform-allocation runs only."""
    res = {}
    for f in sorted(glob.glob(str(F_ARMS/"ir*_results.json"))):
        r = load(f)
        p = r["tag"].split("_")                     # ir150_B_s42 -> ['ir150','B','s42']
        if p[1] == "B" and r.get("alloc") != "uniform":
            raise SystemExit(f"{r['tag']}: expected uniform allocation, got {r.get('alloc')}")
        res.setdefault((int(p[0][2:]), p[1]), []).append(r)
    # IR 150 arm A: existing alignment-init real-only runs on train_lt.txt
    for f in sorted(glob.glob(str(F_ARMS/"A_align*_results.json"))):
        r = load(f)
        if r["n_real"] == N_REAL_150 and r["init"] == "align":
            res.setdefault((150, "A"), []).append(r)
    return res


def collect_uec():
    A = [load(f) for f in sorted(glob.glob(str(U_ARMS/"A_align_s*_results.json")))]
    B = [load(f) for f in sorted(glob.glob(str(U_ARMS/"B_align_s*_results.json")))]
    return A, B


def stats(runs, key="top1"):
    v = np.array([r["no_tta"][key] for r in runs], dtype=float)
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    return v.mean(), sd, len(v)


def row(dataset, ir, n_real, n_synth, A, B, tail=True):
    am, asd, an = stats(A); bm, bsd, bn = stats(B)
    out = {"dataset": dataset, "ir": ir, "n_real": n_real,
           "synth_per_real": n_synth / n_real,
           "A": am, "A_sd": asd, "n_A": an, "B": bm, "B_sd": bsd, "n_B": bn,
           "gain": bm - am, "gain_sd": float(np.hypot(asd, bsd))}
    if tail:
        out["tail_gain"] = stats(B, "top1_tail")[0] - stats(A, "top1_tail")[0]
        out["head_gain"] = stats(B, "top1_head")[0] - stats(A, "top1_head")[0]
    else:
        out["tail_gain"] = out["head_gain"] = float("nan")
    return out


def n_str(r):
    return f"{r['n_A']}" if r["n_A"] == r["n_B"] else f"{r['n_A']}/{r['n_B']}"


def main():
    res  = collect_food101()
    real = {m["imbalance_ratio"]: m["total_train_images"]
            for m in json.loads((SWEEP/"manifest.json").read_text())["ratios"]}

    rows = []
    for ir in sorted({k[0] for k in res}, reverse=True):
        A, B = res.get((ir, "A")), res.get((ir, "B"))
        if not A or not B:
            print(f"IR {ir}: incomplete (A={len(A or [])}, B={len(B or [])}) -- skipped")
            continue
        assert all(r["n_real"] == real[ir] for r in A + B), f"IR {ir}: n_real mismatch"
        rows.append(row("Food101-LT", ir, real[ir], FIXED_BUDGET, A, B))

    uA, uB = collect_uec()
    uec = None
    if uA and uB:
        uec = row("UEC-256", 7.9, uA[0]["n_real"], uB[0]["n_synth"], uA, uB, tail=False)

    allrows = rows + ([uec] if uec else [])
    print(f"\n{'dataset':<12} {'IR':>5} {'real':>7} {'s/r':>6} {'n':>4} "
          f"{'A':>14} {'B':>14} {'gain':>7}")
    for r in allrows:
        print(f"{r['dataset']:<12} {r['ir']:>5} {r['n_real']:>7} {r['synth_per_real']:>5.1%} "
              f"{n_str(r):>4} {r['A']:>7.2f}±{r['A_sd']:<5.2f} {r['B']:>7.2f}±{r['B_sd']:<5.2f} "
              f"{r['gain']:>+7.2f}")

    with open(OUT/"curve_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader(); w.writerows(allrows)

    def tex_row(r):
        return (f"{r['dataset']} & {r['ir']} & {r['n_real']:,} & "
                f"{100*r['synth_per_real']:.1f}\\% & {n_str(r)} & "
                f"${r['A']:.2f}\\pm{r['A_sd']:.2f}$ & ${r['B']:.2f}\\pm{r['B_sd']:.2f}$ & "
                f"${r['gain']:+.2f}$ \\\\")

    tex = ["% Mean $\\pm$ sample std (ddof=1) over n seeds; top-1, no TTA.",
           "% Food101-LT B: the shared pool-capped sweep list at every ratio. UEC-256 B: uniform, 21 per class.",
           "\\begin{tabular}{lrrrrrrr}", "\\toprule",
           "dataset & IR & real & synth./real & $n$ & $A$ (real only) & $B$ (+synth) & gain \\\\",
           "\\midrule"]
    tex += [tex_row(r) for r in rows]
    if uec:
        tex += ["\\midrule", tex_row(uec)]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT/"curve_table.tex").write_text("\n".join(tex) + "\n")

    irs   = np.array([r["ir"] for r in rows], dtype=float)
    gains = np.array([r["gain"] for r in rows])
    errs  = np.array([r["gain_sd"] for r in rows])

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.axhline(0, color="#999", lw=0.8, zorder=1)
    ax.errorbar(irs, gains, yerr=errs, fmt="o-", color="#2c5f8a", markersize=6,
                capsize=3, lw=1.6, zorder=3, label="Food101-LT (3,800 synthetic)")
    for x, y in zip(irs, gains):
        ax.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="#2c5f8a")
    if uec:
        ax.errorbar([uec["ir"]], [uec["gain"]], yerr=[uec["gain_sd"]], fmt="s",
                    color="#c1502e", markersize=7, capsize=3, zorder=4,
                    label="UEC-256 (5,376 synthetic)")
        ax.annotate(f"{uec['gain']:+.2f}", (uec["ir"], uec["gain"]),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    fontsize=8, color="#c1502e")

    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 150])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("training-set imbalance ratio (max class / min class)")
    ax.set_ylabel("top-1 gain from synthetic data (pp)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(-1.2, max(gains.max() if len(gains) else 1, 1) + 1.4)
    fig.tight_layout()
    fig.savefig(OUT/"fig_imbalance_curve.png", dpi=300)
    fig.savefig(OUT/"fig_imbalance_curve.pdf")
    plt.close(fig)
    print(f"\nwrote {OUT/'fig_imbalance_curve.png'} (+ .pdf, .csv, .tex)")


if __name__ == "__main__":
    main()
