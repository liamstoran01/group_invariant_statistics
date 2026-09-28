"""
symmetry model comparison
-------------------------
Every candidate symmetry is operationalized identically: average M* over the
candidate's invariance classes (its own kernel estimate), take the top-6
eigenmodes of the averaged matrix as its predicted embedding, and score the
prediction against the actual embedding (spectral factorization of raw M*).

Scores:
  * homogeneity R^2 : variance of M* explained by each candidate's pair
                labeling (same nested diagnostic as the old inversion test;
                Z12 -> D12 -> conjugacy drops are printed explicitly)
  * full-fit  : predict the full-data embedding (comparable to the theory
                overlay: per-chord cosine after one global Procrustes, and
                top-6 subspace overlap)
  * held-out  : kernel estimated on half the chorales predicts the OTHER
                half's embedding (extra parameters can hurt here); also
                broken down per plane, since plane 2 is where candidates
                separate most. Panel (a) averages this over 10 random
                half-splits (bars = mean, error bars = sample std).
                Panel (b) learning curve: same held-out setup vs training
                size, but with an independent orthogonal Procrustes per
                plane and the per-chord cosine in the stacked 6D space.

Candidates: D12 (T/I, the theory), Z12-only, Z24 (adapted interleaved-fifths
cycle), PLR (right D12), Z12 x Z12 (decoupled circles), latent-Z12 (one
shared circle, relative pairing), radial D12 (class function), and the RAW
unsymmetrized train-half matrix (memorization baseline, 576 parameters).

Also writes model_comparison_metrics.json (or <out stem>_metrics.json) with
the R^2 / fit numbers for easy reuse.

Usage:  python3 symmetry_model_comparison.py [--sequences chord_sequences_full.json]
                                    [--out model_comparison.pdf]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import chordlib as cl


def candidates():
    return [("T/I", cl.d12_labels()),
            ("Z12", cl.z12_labels()[0]),
            ("Z24", cl.z24_labels()),
            ("PLR", cl.plr_labels()),
            ("Z12 x Z12", cl.z12xz12_labels()),
            ("latent Z12", cl.latent_z12_labels(9)),
            ("T/I conj. class", cl.conj_labels()[0])]


def metrics_path(out):
    stem, _ = os.path.splitext(out)
    return f"{stem}_metrics.json"


def texify(name):
    """Z12 / Z24 -> mathtext subscripts for plot labels."""
    name = name.replace(" x ", "x")
    for plain, tex in (("Z12", r"$Z_{12}$"), ("Z24", r"$Z_{24}$")):
        name = name.replace(plain, tex)
    return name


def axis_label(name):
    """Two-line tick labels where needed to free horizontal space."""
    name = texify(name)
    name = name.replace("latent ", "latent\n")
    if name == "T/I conj. class":
        return "T/I\nconj.\nclass"
    return name


def score(pred_M, targ_M, k=6):
    """(subspace overlap, per-chord cosine after global Procrustes)."""
    lp, Vp = cl.embedding_modes(pred_M, k)
    lt, Vt = cl.embedding_modes(targ_M, k)
    overlap = cl.subspace_overlap(Vp, Vt)
    Ep = Vp * np.sqrt(np.clip(lp, 0, None))
    Et = Vt * np.sqrt(np.clip(lt, 0, None))
    R, _ = cl.procrustes(Et, Ep)
    EpR = Ep @ R
    cos = [Et[i] @ EpR[i] / max(np.linalg.norm(Et[i]) *
                                np.linalg.norm(EpR[i]), 1e-12)
           for i in range(cl.N)]
    return overlap, float(np.mean(cos))


def per_plane(pred_M, targ_M):
    lp, Vp = cl.embedding_modes(pred_M, 6)
    lt, Vt = cl.embedding_modes(targ_M, 6)
    out = []
    for p in range(3):
        A, B = Vp[:, 2 * p:2 * p + 2], Vt[:, 2 * p:2 * p + 2]
        ov = cl.subspace_overlap(A, B)
        Ea = A * np.sqrt(np.clip(lp[2 * p:2 * p + 2], 0, None))
        Eb = B * np.sqrt(np.clip(lt[2 * p:2 * p + 2], 0, None))
        R, _ = cl.procrustes(Eb, Ea)
        EaR = Ea @ R
        cos = np.mean([Eb[i] @ EaR[i] / max(np.linalg.norm(Eb[i]) *
                                            np.linalg.norm(EaR[i]), 1e-12)
                       for i in range(cl.N)])
        out.append((ov, float(cos)))
    return out


def score_plane_wise_6d(pred_M, targ_M):
    """Mean per-chord cosine in 6D after an independent Procrustes per plane.

    Same plane pairing as panel (a) / per_plane: PC1-2, PC3-4, PC5-6 each
    get their own 2x2 R; theory/pred coords are stacked and scored in R^6.
    """
    lp, Vp = cl.embedding_modes(pred_M, 6)
    lt, Vt = cl.embedding_modes(targ_M, 6)
    Ep = Vp * np.sqrt(np.clip(lp, 0, None))
    Et = Vt * np.sqrt(np.clip(lt, 0, None))
    Ep_aligned = np.zeros_like(Ep)
    for p in range(3):
        sl = slice(2 * p, 2 * p + 2)
        R, _ = cl.procrustes(Et[:, sl], Ep[:, sl])
        Ep_aligned[:, sl] = Ep[:, sl] @ R
    cos = [Et[i] @ Ep_aligned[i] /
           max(np.linalg.norm(Et[i]) * np.linalg.norm(Ep_aligned[i]), 1e-12)
           for i in range(cl.N)]
    return float(np.mean(cos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="chord_sequences_full.json")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--out", default="model_comparison.pdf",
                    help="primary output; a .png twin is also written")
    args = ap.parse_args()

    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 17,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
    })

    seqs = json.load(open(args.sequences))
    M = cl.triad_block(*cl.mstar(seqs, window=args.window))
    M1 = cl.triad_block(*cl.mstar(seqs[0::2], window=args.window))
    M2 = cl.triad_block(*cl.mstar(seqs[1::2], window=args.window))

    # Homogeneity R^2 for every candidate labeling on full-data M*.
    # Nested Z12 -> D12 -> conjugacy is the old inversion diagnostic.
    lab_z = cl.z12_labels()[0]
    lab_d = cl.d12_labels()
    lab_c = cl.conj_labels()[0]
    r2_z, r2_d, r2_c = cl.r2(M, lab_z), cl.r2(M, lab_d), cl.r2(M, lab_c)
    print("homogeneity R^2 on full-data M* (triad block):")
    print(f"  Z12 only (48 classes):  {r2_z:.4f}")
    print(f"  D12 T/I  (24 classes):  {r2_d:.4f}   "
          f"(drop from adding inversion: {r2_z - r2_d:.4f})")
    print(f"  conjugacy (9 classes):  {r2_c:.4f}   "
          f"(drop from D12 -> radial: {r2_d - r2_c:.4f})")

    rows = []
    print(f"\n{'candidate':>17} {'params':>6} {'R^2':>7} | "
          f"{'full-fit ov/cos':>16} | {'held-out ov/cos':>16}")
    for name, lab in candidates():
        npar = len(np.unique(lab))
        r2 = cl.r2(M, lab)
        o1, c1 = score(cl.class_average(M, lab), M)
        o2, c2 = score(cl.class_average(M1, lab), M2)
        rows.append(dict(name=name, params=npar, r2=r2,
                         full_overlap=o1, full_cosine=c1,
                         held_overlap=o2, held_cosine=c2))
        print(f"{name:>17} {npar:>6} {r2:>7.4f} | {o1:>7.3f} {c1:>8.3f} | "
              f"{o2:>7.3f} {c2:>8.3f}")
    o, c = score(M1, M2)
    rows.append(dict(name="RAW", params=552, r2=None,
                     full_overlap=1.0, full_cosine=1.0,
                     held_overlap=o, held_cosine=c))
    print(f"{'RAW':>17} {552:>6} {'—':>7} | {'1.000':>7} {'1.000':>8} | "
          f"{o:>7.3f} {c:>8.3f}")

    metrics = {
        "sequences": args.sequences,
        "window": args.window,
        "nested_homogeneity_r2": {
            "Z12": r2_z,
            "D12": r2_d,
            "conjugacy": r2_c,
            "drop_Z12_to_D12": r2_z - r2_d,
            "drop_D12_to_conjugacy": r2_d - r2_c,
        },
        "candidates": rows,
    }
    mpath = metrics_path(args.out)
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved -> {mpath}")

    print("\nheld-out, per plane (mean±std cosine over 10 random half-splits):")
    show = candidates() + [("RAW", None)]
    n_splits = 10
    cos_splits = np.zeros((n_splits, len(show), 3))
    for r in range(n_splits):
        rng = np.random.default_rng(1000 + r)
        idx = rng.permutation(len(seqs))
        half = len(seqs) // 2
        train = [seqs[i] for i in idx[:half]]
        test = [seqs[i] for i in idx[half:]]
        Mtr = cl.triad_block(*cl.mstar(train, window=args.window))
        Mte = cl.triad_block(*cl.mstar(test, window=args.window))
        for i, (_, lab) in enumerate(show):
            P = Mtr if lab is None else cl.class_average(Mtr, lab)
            cos_splits[r, i] = [c for _, c in per_plane(P, Mte)]
    cos_mean = cos_splits.mean(axis=0)
    cos_std = cos_splits.std(axis=0, ddof=1)
    print(f"{'candidate':>17} | {'plane1':>14} | {'plane2':>14} | "
          f"{'plane3':>14}")
    for i, (name, _) in enumerate(show):
        cells = [f"{cos_mean[i, p]:.2f}±{cos_std[i, p]:.2f}" for p in range(3)]
        print(f"{name:>17} | " + " | ".join(f"{c:>14}" for c in cells))

    # ---------------- figure: (a) per-plane bars, (b) learning curve ----
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))

    # (a) held-out per-plane cosine: mean ± sample std over 10 half-splits
    ax = axes[0]
    x = np.arange(len(show))
    width = 0.26
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for p in range(3):
        ax.bar(x + (p - 1) * width, cos_mean[:, p], width * 0.9,
               yerr=cos_std[:, p], color=colors[p], label=f"plane {p + 1}",
               error_kw=dict(ecolor="0.2", capsize=3, lw=1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([axis_label(n) for n, _ in show], fontsize=16)
    # Tiny horizontal nudges for crowded tick labels.
    from matplotlib.transforms import ScaledTranslation
    nudges = {"Z12 x Z12": -4 / 72, "PLR": -4 / 72, "latent Z12": 4 / 72}
    for tick, (name, _) in zip(ax.get_xticklabels(), show):
        dx = nudges.get(name)
        if dx is not None:
            tick.set_transform(
                tick.get_transform()
                + ScaledTranslation(dx, 0, fig.dpi_scale_trans))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("per-chord cosine similarity")
    ax.legend(frameon=False)
    ax.set_title("(a) held-out embedding prediction")

    # (b) learning curve: held-out cosine vs training-set size
    ax = axes[1]
    curve_cands = [("T/I", cl.d12_labels(), "tab:blue"),
                   ("Z12", cl.z12_labels()[0], "tab:cyan"),
                   ("Z24", cl.z24_labels(), "tab:green"),
                   ("latent Z12", cl.latent_z12_labels(9),
                    "tab:purple"),
                   ("RAW", None, "tab:red")]
    pool = seqs[0::2]
    sizes = [12, 25, 50, 100, len(pool)]
    for name, lab, color in curve_cands:
        means, stds = [], []
        for n in sizes:
            reps = 15 if n < len(pool) else 1
            vals = []
            for r in range(reps):
                rng = np.random.default_rng(10 * n + r)
                sub = [pool[i] for i in
                       rng.choice(len(pool), n, replace=False)]
                Mtr = cl.triad_block(*cl.mstar(sub, window=args.window))
                P = Mtr if lab is None else cl.class_average(Mtr, lab)
                vals.append(score_plane_wise_6d(P, M2))
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)
        style = "s--" if name == "Z12" else "o-"
        ax.plot(sizes, means, style, color=color, label=texify(name),
                ms=6, lw=1.5, zorder=3 if name == "T/I" else 2)
        ax.fill_between(sizes, means - stds, means + stds, color=color,
                        alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.minorticks_off()
    ax.set_xlabel("# training chorales")
    ax.set_ylabel("per-chord cosine similarity")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, columnspacing=1.0, handletextpad=0.4,
              fontsize=15)
    ax.set_title("(b) data efficiency")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.03)
    print(f"saved -> {args.out}")
    png_out = (args.out[:-4] + ".png" if args.out.lower().endswith(".pdf")
               else args.out + ".png")
    if png_out != args.out:
        fig.savefig(png_out, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print(f"saved -> {png_out}")


if __name__ == "__main__":
    main()
