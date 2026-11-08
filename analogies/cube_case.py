"""
cube3_modes_vs_theory.py
------------------------
Mine a corpus for the complete Z_2^3 cube
  man/woman/men/women/king/queen/kings/queens
(attributes: gender g, number n, royalty r) and produce the four-panel
figure:

  (1) doubly centered PMI M* over the 8 words;
  (2) its Reynolds projection = the nearest exactly Z_2^3-invariant
      matrix (average over the 8 diagonal attribute flips);
  (3) the empirical eigenvectors of M* (|eigenvalue| order, each
      sign-aligned to its best-matching character, overlap printed);
  (4) the theory panel: the seven nontrivial Walsh characters
      chi_S(alpha) = prod_{k in S} alpha_k / sqrt(8) -- pure sign
      patterns fixed by the group before any data is seen.

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
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.8),
                           gridspec_kw={"width_ratios": [1, 1, .95, .95]})
    vmax = np.max(np.abs(Mc))
    panels = [(Mc, ax[0], "doubly centered PMI $M^*$ (mined)"),
              (R, ax[1], "nearest exactly $\\mathbb{Z}_2^3$-invariant\n"
               f"(Reynolds projection, $R^2$={r2:.2f})")]
    for A, a, title in panels:
        im = a.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_xticks(range(N))
        a.set_yticks(range(N))
        a.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
        a.set_yticklabels(targets, fontsize=8)
        a.axhline(3.5, color="k", lw=1.1)
        a.axvline(3.5, color="k", lw=1.1)
        for p in (1.5, 5.5):
            a.axhline(p, color="k", lw=0.5, ls=":")
            a.axvline(p, color="k", lw=0.5, ls=":")
        a.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax[1], shrink=0.75, pad=0.02)
    vm = np.max(np.abs(E))
    for A, a, title in [(E, ax[2], "empirical eigenvectors of $M^*$\n"
                         "($|\\lambda|$ order, sign-aligned)"),
                        (Th, ax[3], "theory: Walsh characters "
                         "$\\chi_S$\n(matched modes)")]:
        a.imshow(A, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
        a.set_yticks(range(N))
        a.set_yticklabels(targets, fontsize=8)
        a.set_xticks(range(7))
        if A is E:
            a.set_xticklabels([f"{nm}\n{ov:.2f}" for nm, ov
                               in zip(names, ovls)], fontsize=7.5)
            a.set_xlabel("matched character / overlap", fontsize=8)
        else:
            a.set_xticklabels(names, fontsize=9)
        a.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=170, bbox_inches="tight")
    print(f"saved -> {args.out}.pdf/.png")


if __name__ == "__main__":
    main()