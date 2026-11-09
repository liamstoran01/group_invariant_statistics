"""
appendix_cube_figures.py
------------------------
Produces the two appendix figures for the Z_2^3 cube
  man/woman/men/women/king/queen/kings/queens
(attributes: gender g, number n, royalty r), mined from a corpus.

Figure 1 (<out>_projections):
  two heatmaps side by side -- the projection of P M* P onto (left)
  the full Z_2^3-invariant family (all irreps; Reynolds projection)
  and (right) the independent-attribute model of Korchinski et al.
  (irreps {g}, {n}, {r} only). The R^2 of the original P M* P
  explained by each model is printed to stdout/JSON for the caption.

Figure 2 (<out>_analogies):
  the per-mode flip test d1[k] d2[k] for all six analogy types of the
  cube, each panel coloring that analogy's enemy irreps
  (S containing both varied attributes) in red. The independence
  model predicts every bar nonnegative; the trichotomy predicts the
  red bars flip.

Corpus input as in mine_cube_trichotomy.py (same directory):
  --demo (text8), --wiki dump.xml.bz2, or --text files.

Usage:
  python3 appendix_cube_figures.py --demo --window 10
"""
import argparse
import itertools
import json
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
ATTR = {"g": 0, "n": 1, "r": 2}


def mine(args):
    order = list(WORDS)
    targets = [WORDS[p] for p in order]
    cargs = types.SimpleNamespace(demo=args.demo, wiki=args.wiki,
                                  text=args.text, window=args.window)
    pair, uni, T = mc.count_corpus(cargs, targets)
    print(f"corpus: {T:,} tokens; unigrams "
          f"{dict(zip(targets, uni.astype(int)))}")
    PMI, zf = mc.build_pmi(pair, uni, T, args.window)
    return order, targets, np.array(order), PMI


def walsh_amplitudes(PMI, pts):
    N = len(pts)
    agree = {}
    for a in range(N):
        for b in range(N):
            agree.setdefault(tuple(pts[a] * pts[b]), []).append(PMI[a, b])
    pooled = {k: np.mean(v) for k, v in agree.items()}
    subsets = [S for r_ in range(4)
               for S in itertools.combinations("gnr", r_)]
    amp = {}
    for S in subsets:
        amp[S] = float(np.mean(
            [pooled[k] * (np.prod([k[ATTR[c]] for c in S]) if S else 1.0)
             for k in pooled]))
    return subsets, amp


def chivec(S, pts):
    return np.array([np.prod([p[ATTR[c]] for c in S]) if S else 1.0
                     for p in pts])


def build_model(amp, keep, pts, H):
    N = len(pts)
    K = np.zeros((N, N))
    for S in keep:
        v = chivec(S, pts)
        K += amp[S] * np.outer(v, v)
    return H @ K @ H


def figure_projections(Mc, pts, targets, amp, subsets, out):
    N = len(pts)
    H = np.eye(N) - np.ones((N, N)) / N
    M_full = build_model(amp, subsets, pts, H)
    M_ind = build_model(amp, [S for S in subsets if len(S) <= 1],
                        pts, H)
    r2 = lambda A: 1 - np.var(Mc - A) / np.var(Mc)
    rf, rk = r2(M_full), r2(M_ind)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.9))
    vmax = np.max(np.abs(Mc))
    panels = [
        (M_full, r"$\Pi_{\mathbb{Z}_2^d}\,\mathrm{P}\mathrm{M}^\star\mathrm{P}$"),
        (M_ind, r"$\Pi_{\mathrm{stats}}$"),
    ]
    for (A, title), a in zip(panels, ax):
        im = a.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_xticks(range(N)); a.set_yticks(range(N))
        a.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
        a.set_yticklabels(targets, fontsize=8)
        a.axhline(3.5, color="k", lw=1.1)
        a.axvline(3.5, color="k", lw=1.1)
        for p in (1.5, 5.5):
            a.axhline(p, color="k", lw=0.5, ls=":")
            a.axvline(p, color="k", lw=0.5, ls=":")
        a.set_title(title, fontsize=14, pad=8)
    fig.colorbar(im, ax=ax, shrink=0.78, pad=0.015)
    fig.savefig(f"{out}_projections.pdf", bbox_inches="tight")
    fig.savefig(f"{out}_projections.png", dpi=180, bbox_inches="tight")
    print(f"R^2: full invariant {rf:.3f}, independent {rk:.3f}, "
          f"gap {rf - rk:.3f}")
    return rf, rk


def figure_analogies(Mc, pts, targets, out):
    N = len(pts)
    lam, U = np.linalg.eigh(Mc)
    o = np.argsort(-np.abs(lam))
    lam, U = lam[o], U[:, o]
    keepk = [k for k in range(N) if abs(lam[k]) > 1e-8][:7]
    subsets = [S for r_ in range(1, 4)
               for S in itertools.combinations("gnr", r_)]
    names = []
    for k in keepk:
        ov = [abs(U[:, k] @ chivec(S, pts)) for S in subsets]
        names.append("".join(subsets[int(np.argmax(ov))]))
    W = U[:, keepk] * np.sqrt(np.abs(lam[keepk]))
    iw = {w: i for i, w in enumerate(targets)}
    wv = lambda g, n, r: W[iw[WORDS[(g, n, r)]]]
    def arr(a, b, c, d):
        return rf"{a} $\rightarrow$ {b} : {c} $\rightarrow$ {d}"

    cases = [
        (arr("king", "queen", "kings", "queens"),
         wv(1, 1, 1) - wv(-1, 1, 1), wv(1, -1, 1) - wv(-1, -1, 1),
         ("gn", "gnr")),
        (arr("king", "queen", "man", "woman"),
         wv(1, 1, 1) - wv(-1, 1, 1), wv(1, 1, -1) - wv(-1, 1, -1),
         ("gr", "gnr")),
        (arr("king", "kings", "queen", "queens"),
         wv(1, 1, 1) - wv(1, -1, 1), wv(-1, 1, 1) - wv(-1, -1, 1),
         ("gn", "gnr")),
        (arr("king", "kings", "man", "men"),
         wv(1, 1, 1) - wv(1, -1, 1), wv(1, 1, -1) - wv(1, -1, -1),
         ("nr", "gnr")),
        (arr("man", "king", "woman", "queen"),
         wv(1, 1, 1) - wv(1, 1, -1), wv(-1, 1, 1) - wv(-1, 1, -1),
         ("gr", "gnr")),
        (arr("man", "king", "men", "kings"),
         wv(1, 1, 1) - wv(1, 1, -1), wv(1, -1, 1) - wv(1, -1, -1),
         ("nr", "gnr"))]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7))
    summary = {}
    for (title, d1, d2, enemies), a in zip(cases, axes.ravel()):
        prods = d1 * d2
        a.bar(range(7), prods,
              color=["crimson" if nm in enemies else "steelblue"
                     for nm in names])
        a.set_xticks(range(7))
        a.set_xticklabels(names, fontsize=8)
        a.axhline(0, color="k", lw=0.6)
        flip = float(sum(p for p in prods if p < 0))
        pos = float(sum(p for p in prods if p > 0))
        a.set_title(title, fontsize=11)
        print(f"  {title}: flip mass {flip:+.2f} vs friendly {pos:+.2f}")
        key = f"{title} ({enemies[0]})"
        summary[key] = {"flip": flip, "friendly": pos}
    fig.tight_layout()
    fig.savefig(f"{out}_analogies.pdf", bbox_inches="tight")
    fig.savefig(f"{out}_analogies.png", dpi=160, bbox_inches="tight")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki")
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--out", default="appendix_cube")
    args = ap.parse_args()

    order, targets, pts, PMI = mine(args)
    N = len(pts)
    H = np.eye(N) - np.ones((N, N)) / N
    Mc = H @ PMI @ H
    subsets, amp = walsh_amplitudes(PMI, pts)
    rf, rk = figure_projections(Mc, pts, targets, amp, subsets,
                                args.out)
    summary = figure_analogies(Mc, pts, targets, args.out)
    with open(f"{args.out}_metrics.json", "w") as fh:
        json.dump({"r2_full_invariant": rf, "r2_independent": rk,
                   "walsh_amplitudes":
                       {"".join(S) or "1": amp[S] for S in subsets},
                   "analogies": summary}, fh, indent=1)
    print(f"saved -> {args.out}_projections.(pdf|png), "
          f"{args.out}_analogies.(pdf|png), {args.out}_metrics.json")


if __name__ == "__main__":
    main()