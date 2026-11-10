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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import mine_cube_trichotomy as mc
from cube3_corpus_llm import (
    draw_matrix, gram_from_vectors, llm_vectors, subset_label,
)

_CACHE = Path(__file__).resolve().parent / "cube_mstar_cache"

WORDS = {(+1, +1, +1): "king",  (-1, +1, +1): "queen",
         (+1, -1, +1): "kings", (-1, -1, +1): "queens",
         (+1, +1, -1): "man",   (-1, +1, -1): "woman",
         (+1, -1, -1): "men",   (-1, -1, -1): "women"}
ATTR = {"g": 0, "n": 1, "r": 2}


def _corpus_stem(args):
    if getattr(args, "text", None):
        text = args.text[0] if isinstance(args.text, (list, tuple)) else args.text
        return Path(text).stem.lower()
    if getattr(args, "demo", False):
        return "demo"
    return "wiki"


def _pmi_cache_path(args):
    return _CACHE / f"pmi_{_corpus_stem(args)}_w{args.window}.npz"


def _llm_cache_path(model, layer):
    tag = str(model).replace("/", "_")
    return _CACHE / f"llm_{tag}_L{layer}_cube8.npy"


def mine(args):
    order = list(WORDS)
    targets = [WORDS[p] for p in order]
    cache = _pmi_cache_path(args)
    refresh = getattr(args, "refresh_cache", False)
    if cache.is_file() and not refresh:
        z = np.load(cache, allow_pickle=False)
        PMI = z["PMI"]
        uni = z["uni"]
        T = int(z["T"])
        print(f"loaded PMI cache {cache.name}: {T:,} tokens; unigrams "
              f"{dict(zip(targets, uni.astype(int)))}")
        return order, targets, np.array(order), PMI
    cargs = types.SimpleNamespace(demo=args.demo, wiki=args.wiki,
                                  text=args.text, window=args.window)
    pair, uni, T = mc.count_corpus(cargs, targets)
    print(f"corpus: {T:,} tokens; unigrams "
          f"{dict(zip(targets, uni.astype(int)))}")
    PMI, zf = mc.build_pmi(pair, uni, T, args.window)
    _CACHE.mkdir(exist_ok=True)
    np.savez(cache, PMI=PMI, pair=pair, uni=uni, T=np.int64(T))
    print(f"cached PMI -> {cache.name}")
    return order, targets, np.array(order), PMI


def load_llm_cube(targets, pts, out, layer=8, model="gpt2", refresh=False):
    cache = _llm_cache_path(model, layer)
    if cache.is_file() and not refresh:
        X = np.load(cache)
        print(f"loaded LLM cache {cache.name}  shape {X.shape}")
        return X, layer
    largs = types.SimpleNamespace(
        mock=False, model=model, layer=layer, n_test=2, n_splits=0,
        ridge=0.0, probe_seed=0)
    X, used_layer, _ = llm_vectors(largs, targets, pts, out)
    _CACHE.mkdir(exist_ok=True)
    np.save(cache, X)
    print(f"cached LLM vectors -> {cache.name}")
    return X, used_layer


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
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "font.size": 20, "axes.titlesize": 24, "axes.labelsize": 22,
        "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
    })
    tick_fs, title_fs = 20, 24
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 6.4),
                           gridspec_kw={"wspace": 0.05})
    vmax = float(np.max(np.abs(Mc)))
    cmap = plt.cm.BrBG
    draw_matrix(ax[0], M_full, targets, vmax, cmap,
                r"$\Pi_{\mathbb{Z}_2^3}\,\mathrm{PM}^{\!\star}\!\mathrm{P}$",
                tick_fs=tick_fs, title_fs=title_fs)
    im = draw_matrix(ax[1], M_ind, targets, vmax, cmap,
                     r"$\Pi_{\deg\,\mathbb{Z}_2^3 < 1}\,\mathrm{PM}^{\!\star}\!\mathrm{P}$",
                     ytick=False, tick_fs=tick_fs, title_fs=title_fs)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    fig.savefig(f"{out}_projections.pdf", bbox_inches="tight")
    fig.savefig(f"{out}_projections.png", dpi=170, bbox_inches="tight")
    print(f"R^2: full invariant {rf:.3f}, independent {rk:.3f}, "
          f"gap {rf - rk:.3f}")
    return rf, rk


def analogy_coords(M, pts, targets):
    N = len(pts)
    lam, U = np.linalg.eigh(M)
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
    return names, wv


def draw_flip(ax, names, prods, enemies, title, tick_fs=16, title_fs=20,
              nulls=()):
    colors = []
    for nm in names:
        if nm in enemies:
            colors.append("crimson")
        elif nm in nulls:
            colors.append("0.72")
        else:
            colors.append("steelblue")
    ax.bar(range(7), prods, color=colors)
    ax.set_xticks(range(7))
    ax.set_xticklabels([subset_label(nm) for nm in names],
                       fontsize=tick_fs, rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(title, fontsize=title_fs, pad=10)


def figure_analogies(Mc, pts, targets, out):
    names, wv = analogy_coords(Mc, pts, targets)
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


def figure_plural_corpus_llm(Mc, pts, targets, out, layer=8,
                            refresh_cache=False, model="gpt2"):
    """king → kings : man → men  on corpus M* vs an LLM layer."""
    enemies = ("nr", "gnr")
    nulls = ("g", "r", "gr")
    title = (r"$d_1 = W_{\mathrm{kings}} - W_{\mathrm{king}},"
             r"\quad d_2 = W_{\mathrm{men}} - W_{\mathrm{man}}$")
    X, used_layer = load_llm_cube(targets, pts, out, layer=layer,
                                  model=model, refresh=refresh_cache)
    Ml = gram_from_vectors(X)
    short = "gpt2" if model == "gpt2" else model.split("/")[-1]
    if short.lower().startswith("qwen"):
        short = "qwen"
    print(f"{model} layer {used_layer} Gram for {title}")
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "font.size": 18, "axes.titlesize": 20, "axes.labelsize": 20,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
    })
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 5.8),
                           gridspec_kw={"wspace": 0.22})
    for M, a, sub in ((Mc, ax[0], "word embedding"),
                      (Ml, ax[1], short)):
        names, wv = analogy_coords(M, pts, targets)
        d1 = wv(1, -1, 1) - wv(1, 1, 1)
        d2 = wv(1, -1, -1) - wv(1, 1, -1)
        prods = d1 * d2
        draw_flip(a, names, prods, enemies, sub, nulls=nulls)
        flip = float(sum(p for p in prods if p < 0))
        pos = float(sum(p for p in prods if p > 0))
        print(f"  {sub}: flip mass {flip:+.2f} vs friendly {pos:+.2f}")
    ax[0].set_ylabel(r"$d_1[k]\,d_2[k]$", fontsize=20)
    fig.suptitle(title, fontsize=22, y=1.04)
    fig.subplots_adjust(bottom=0.28)
    fig.canvas.draw()
    p0, p1 = ax[0].get_position(), ax[1].get_position()
    fig.legend(
        handles=[
            Patch(facecolor="steelblue", edgecolor="none",
                  label=r"theoretically $>0$"),
            Patch(facecolor="crimson", edgecolor="none",
                  label=r"theoretically $<0$"),
            Patch(facecolor="0.72", edgecolor="none",
                  label=r"theoretically $=0$"),
        ],
        loc="upper center", ncol=3, fontsize=16, frameon=False,
        bbox_to_anchor=(0.5 * (p0.x0 + p1.x1), p0.y0 - 0.14),
        bbox_transform=fig.transFigure,
        handlelength=0.9, columnspacing=1.4,
    )
    tag = "" if model == "gpt2" else f"_{short}"
    fig.savefig(f"{out}_king_kings{tag}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}_king_kings{tag}.png", dpi=170, bbox_inches="tight")
    print(f"saved -> {out}_king_kings{tag}.(pdf|png)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki")
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--out", default="appendix_cube")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=8,
                    help="LLM layer for the king→kings : man→men panel")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="recount corpus / reload GPT-2 even if cache exists")
    ap.add_argument("--king-kings-only", action="store_true",
                    help="only redraw the king→kings : man→men two-panel")
    args = ap.parse_args()

    order, targets, pts, PMI = mine(args)
    N = len(pts)
    H = np.eye(N) - np.ones((N, N)) / N
    Mc = H @ PMI @ H
    if args.king_kings_only:
        figure_plural_corpus_llm(Mc, pts, targets, args.out,
                                 layer=args.layer, model=args.model,
                                 refresh_cache=args.refresh_cache)
        return
    subsets, amp = walsh_amplitudes(PMI, pts)
    rf, rk = figure_projections(Mc, pts, targets, amp, subsets,
                                args.out)
    summary = figure_analogies(Mc, pts, targets, args.out)
    figure_plural_corpus_llm(Mc, pts, targets, args.out, layer=args.layer,
                             model=args.model,
                             refresh_cache=args.refresh_cache)
    with open(f"{args.out}_metrics.json", "w") as fh:
        json.dump({"r2_full_invariant": rf, "r2_independent": rk,
                   "walsh_amplitudes":
                       {"".join(S) or "1": amp[S] for S in subsets},
                   "analogies": summary}, fh, indent=1)
    print(f"saved -> {args.out}_projections.(pdf|png), "
          f"{args.out}_analogies.(pdf|png), {args.out}_king_kings.(pdf|png), "
          f"{args.out}_metrics.json")


if __name__ == "__main__":
    main()