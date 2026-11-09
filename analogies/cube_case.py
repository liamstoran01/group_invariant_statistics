"""
cube3_modes_vs_theory.py
------------------------
Mine a corpus for the complete Z_2^3 cube
  man/woman/men/women/king/queen/kings/queens
(attributes: gender g, number n, royalty r) and produce the three-panel
figure:

  (a) Statistical Invariance Holds: PM*P and its Reynolds projection
      Pi_{Z_2^3};
  (b) Boolean irreps follow: empirical eigenvectors of M* (filled
      squares) overlaid with matched Walsh characters chi_S (frames).

Corpus input as in mine_cube_trichotomy.py: --demo (text8),
--wiki dump.xml.bz2, or --text files. Requires mine_cube_trichotomy.py
on the path (reuses its corpus counting and PMI construction).

Usage:
  python3 cube3_modes_vs_theory.py --demo --window 10
"""
import argparse
import itertools
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

import mine_cube_trichotomy as mc

WORDS = {(+1, +1, +1): "king",  (-1, +1, +1): "queen",
         (+1, -1, +1): "kings", (-1, -1, +1): "queens",
         (+1, +1, -1): "man",   (-1, +1, -1): "woman",
         (+1, -1, -1): "men",   (-1, -1, -1): "women"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki")
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--out", default="cube3_modes_vs_theory")
    args = ap.parse_args()

    order = list(WORDS)
    targets = [WORDS[p] for p in order]
    pts = np.array(order)
    cargs = types.SimpleNamespace(demo=args.demo, wiki=args.wiki,
                                  text=args.text, window=args.window)
    pair, uni, T = mc.count_corpus(cargs, targets)
    print(f"corpus: {T:,} tokens; unigrams "
          f"{dict(zip(targets, uni.astype(int)))}")
    PMI, zf = mc.build_pmi(pair, uni, T, args.window)
    N = 8
    H = np.eye(N) - np.ones((N, N)) / N
    Mc = H @ PMI @ H

    # Reynolds projection onto the Z_2^3-invariant family
    idx = {tuple(p): i for i, p in enumerate(order)}
    R = np.zeros((N, N))
    for f in itertools.product([1, -1], repeat=3):
        perm = [idx[tuple(p * np.array(f))] for p in pts]
        R += Mc[np.ix_(perm, perm)]
    R /= 8.0
    r2 = 1 - np.var(Mc - R) / np.var(Mc)
    print(f"invariance R^2 = {r2:.3f}")

    # empirical modes vs Walsh characters
    lam, U = np.linalg.eigh(Mc)
    o = np.argsort(-np.abs(lam))
    lam, U = lam[o], U[:, o]
    keepk = [k for k in range(N) if np.abs(lam[k]) > 1e-8][:7]
    subsets = [S for r_ in range(1, 4)
               for S in itertools.combinations("gnr", r_)]

    def chi(S):
        i = {"g": 0, "n": 1, "r": 2}
        return np.array([np.prod([p[i[c]] for c in S])
                         for p in pts]) / np.sqrt(N)

    E = np.zeros((N, 7))
    Th = np.zeros((N, 7))
    names, ovls = [], []
    for col, k in enumerate(keepk):
        ov = [U[:, k] @ chi(S) for S in subsets]
        j = int(np.argmax(np.abs(ov)))
        s = np.sign(ov[j])          # gauge: eigenvector sign freedom
        E[:, col] = s * U[:, k]
        Th[:, col] = chi(subsets[j])
        names.append("".join(subsets[j]))
        ovls.append(abs(ov[j]))
    print("matched characters:", names)
    print("overlaps:", [round(float(v), 3) for v in ovls])

    # ---------------- figure ----------------
    cmap = plt.cm.RdBu_r
    fig, ax = plt.subplots(1, 3, figsize=(14.6, 5.2))
    for a in ax:
        a.set_box_aspect(1)
    vmax = np.max(np.abs(Mc))
    for A, a, sub in [(Mc, ax[0], r"$PM^{\star}P$"),
                      (R, ax[1], r"$\Pi_{\mathbb{Z}_2^3}$")]:
        im = a.imshow(A, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        a.set_xticks(range(N))
        a.set_yticks(range(N))
        a.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
        a.set_yticklabels(targets, fontsize=8)
        a.axhline(3.5, color="k", lw=1.1)
        a.axvline(3.5, color="k", lw=1.1)
        for p in (1.5, 5.5):
            a.axhline(p, color="k", lw=0.5, ls=":")
            a.axvline(p, color="k", lw=0.5, ls=":")
        a.set_title(sub, fontsize=12, pad=6)
    a = ax[2]
    vm = max(float(np.max(np.abs(E))), float(np.max(np.abs(Th))), 1e-12)
    norm = Normalize(vmin=-vm, vmax=vm)
    gap, frame = 0.06, 0.18
    for i in range(N):
        for j in range(7):
            a.add_patch(Rectangle(
                (j - 0.5 + gap, i - 0.5 + gap), 1 - 2 * gap, 1 - 2 * gap,
                facecolor=cmap(norm(Th[i, j])), edgecolor="none",
                zorder=1))
            a.add_patch(Rectangle(
                (j - 0.5 + frame, i - 0.5 + frame),
                1 - 2 * frame, 1 - 2 * frame,
                facecolor=cmap(norm(E[i, j])), edgecolor="none",
                zorder=2))
    a.set_xlim(-0.5, 6.5)
    a.set_ylim(N - 0.5, -0.5)
    a.set_aspect("auto")
    a.set_yticks(range(N))
    a.set_yticklabels(targets, fontsize=8)
    a.set_xticks(range(7))
    a.set_xticklabels([f"{nm}\n{ov:.2f}" for nm, ov
                       in zip(names, ovls)], fontsize=7.5)
    a.set_xlabel("matched character / overlap", fontsize=8)
    a.legend([Line2D([0], [0], marker="s", color="none",
                     markerfacecolor="0.35", markeredgecolor="none", ms=9),
              Line2D([0], [0], marker="s", color="none",
                     markerfacecolor="none", markeredgecolor="0.35",
                     markeredgewidth=2.2, ms=11)],
             ["empirical", "theory"], fontsize=8, loc="upper center",
             bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.05, right=0.92, top=0.78, bottom=0.22,
                        wspace=0.42)
    p0, p1, p2 = (a.get_position() for a in ax)
    fig.text(0.5 * (p0.x0 + p1.x1), p0.y1 + 0.07,
             "(a) Statistical Invariance Holds",
             ha="center", va="bottom", fontsize=16, transform=fig.transFigure)
    fig.text(0.5 * (p2.x0 + p2.x1), p2.y1 + 0.07,
             "(b) Boolean irreps follow",
             ha="center", va="bottom", fontsize=16, transform=fig.transFigure)
    cax1 = fig.add_axes([p1.x1 + 0.008, p1.y0, 0.014, p1.height])
    fig.colorbar(im, cax=cax1)
    cax2 = fig.add_axes([p2.x1 + 0.008, p2.y0, 0.014, p2.height])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax2)
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=170, bbox_inches="tight")
    print(f"saved -> {args.out}.pdf/.png")


if __name__ == "__main__":
    main()