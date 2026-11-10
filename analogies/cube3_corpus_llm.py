"""
cube3_corpus_llm.py
-------------------
Four-panel cube figure: corpus invariance (a) plus Boolean-irrep overlays
(b) for the mined word-embedding M* and for the best LLM layer.

  (a) Statistical Invariance Holds: corpus PM*P and Pi_{Z_2^3}
  (b) Boolean Irreps Follow: empirical modes (fill) vs Walsh chi_S
      (frame), once from the corpus M* ("word embedding") and once from
      the LLM Gram ("LLM").

Corpus flags match cube_case.py. LLM layer defaults to the held-out
attribute probe in llm_cube_mode.py (--layer to skip the sweep).

Usage:
  python3 cube3_corpus_llm.py --text ../months/text8.txt --window 10
  python3 cube3_corpus_llm.py --text ../months/text8.txt --model gpt2
  python3 cube3_corpus_llm.py --text ../months/text8.txt --layer 8
"""
import argparse
import itertools
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

import cube_case
import llm_cube_mode as llm
import mine_cube_trichotomy as mc

_DEFAULT_TEXT8 = str(
    Path(__file__).resolve().parent.parent / "months" / "text8.txt")
_CACHE = Path(__file__).resolve().parent / "cube_mstar_cache"


def reynolds(M, pts):
    N = M.shape[0]
    idx = {tuple(p): i for i, p in enumerate(pts)}
    R = np.zeros((N, N))
    for f in itertools.product([1, -1], repeat=3):
        perm = [idx[tuple(np.array(p) * np.array(f))] for p in pts]
        R += M[np.ix_(perm, perm)]
    R /= 8.0
    r2 = 1 - np.var(M - R) / np.var(M)
    return R, float(r2)


_CHAR_INDEX = {"g": 0, "n": 1, "r": 2, "p": 1}


def subset_label(name):
    nums = ",".join(str(_CHAR_INDEX[c]) for c in name)
    return r"$\{" + nums + r"\}$"


def match_walsh(M, pts):
    N = M.shape[0]
    lam, U = np.linalg.eigh(M)
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
        E[:, col] = np.sign(ov[j]) * U[:, k]
        Th[:, col] = chi(subsets[j])
        names.append("".join(subsets[j]))
        ovls.append(abs(ov[j]))
    return E, Th, names, ovls


def gram_from_vectors(X):
    Xc = X - X.mean(0, keepdims=True)
    N = Xc.shape[0]
    H = np.eye(N) - np.ones((N, N)) / N
    M = H @ (Xc @ Xc.T) @ H
    M = 0.5 * (M + M.T)
    M /= max(np.sqrt(np.mean(M ** 2)), 1e-12)
    return M


def draw_matrix(ax, A, labels, vmax, cmap, subtitle, ytick=True,
                tick_fs=8, title_fs=12):
    N = A.shape[0]
    im = ax.imshow(A, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(N))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels if ytick else [], fontsize=tick_fs)
    for p in np.arange(0.5, N - 0.5, 1.0):
        ax.axhline(p, color="k", lw=1.1)
        ax.axvline(p, color="k", lw=1.1)
    ax.set_title(subtitle, fontsize=title_fs, pad=10)
    ax.set_box_aspect(1)
    return im


def draw_overlay(ax, E, Th, names, ovls, labels, norm, cmap, subtitle,
                 ytick=True, xlabel=True, show_match=True,
                 show_overlap=True, tick_fs=8, title_fs=12, xtick_fs=None,
                 xrot=0):
    N = E.shape[0]
    gap, frame = 0.06, 0.18
    for i in range(N):
        for j in range(7):
            ax.add_patch(Rectangle(
                (j - 0.5 + gap, i - 0.5 + gap), 1 - 2 * gap, 1 - 2 * gap,
                facecolor=cmap(norm(Th[i, j])), edgecolor="none", zorder=1))
            ax.add_patch(Rectangle(
                (j - 0.5 + frame, i - 0.5 + frame),
                1 - 2 * frame, 1 - 2 * frame,
                facecolor=cmap(norm(E[i, j])), edgecolor="none", zorder=2))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_aspect("auto")
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels if ytick else [], fontsize=tick_fs)
    if show_match:
        ax.set_xticks(range(7))
        if show_overlap:
            xt = [f"{nm}\n{ov:.2f}" for nm, ov in zip(names, ovls)]
        else:
            xt = list(names)
        kw = dict(fontsize=tick_fs if xtick_fs is None else xtick_fs)
        if xrot:
            kw.update(rotation=xrot, ha="right")
        ax.set_xticklabels(xt, **kw)
        if xlabel:
            ax.set_xlabel("matched character / overlap", fontsize=tick_fs)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.set_title(subtitle, fontsize=title_fs, pad=10)
    ax.set_box_aspect(1)


def corpus_mstar(args, targets):
    N = 8
    H = np.eye(N) - np.ones((N, N)) / N
    texts = args.text if isinstance(getattr(args, "text", None), (list, tuple)) else (
        [args.text] if getattr(args, "text", None) else [])
    stem = Path(texts[0]).stem.lower() if texts else (
        "demo" if getattr(args, "demo", False) else "wiki")
    cache = _CACHE / f"pmi_{stem}_w{args.window}.npz"
    if cache.is_file() and not getattr(args, "refresh_cache", False):
        z = np.load(cache, allow_pickle=False)
        print(f"loaded PMI cache {cache.name}: {int(z['T']):,} tokens; unigrams "
              f"{dict(zip(targets, z['uni'].astype(int)))}")
        return H @ z["PMI"] @ H
    cargs = types.SimpleNamespace(demo=args.demo, wiki=args.wiki,
                                  text=args.text, window=args.window)
    pair, uni, T = mc.count_corpus(cargs, targets)
    print(f"corpus: {T:,} tokens; unigrams "
          f"{dict(zip(targets, uni.astype(int)))}")
    PMI, _ = mc.build_pmi(pair, uni, T, args.window)
    _CACHE.mkdir(exist_ok=True)
    np.savez(cache, PMI=PMI, pair=pair, uni=uni, T=np.int64(T))
    print(f"cached PMI -> {cache.name}")
    return H @ PMI @ H


def llm_vectors(args, targets, pts, out_stem):
    largs = types.SimpleNamespace(
        mock=args.mock, model=args.model, source="layer",
        layer=0 if args.layer is None else args.layer,
        select_layer=(args.layer is None),
        whiten=False)
    if (not args.mock) and (not largs.select_layer):
        tag = str(args.model).replace("/", "_")
        cache = _CACHE / f"llm_{tag}_L{largs.layer}_cube8.npy"
        if cache.is_file() and not getattr(args, "refresh_cache", False):
            X = np.load(cache)
            print(f"loaded LLM cache {cache.name}  shape {X.shape}")
            return X, largs.layer, None
    raw, _ = llm.get_vectors(largs, targets)
    if largs.select_layer:
        Y = np.array(pts, dtype=float)
        layer, probe = llm.select_layer(
            raw, Y, args.n_test, args.n_splits, args.ridge,
            args.probe_seed, out_stem)
        X = raw[layer]
    else:
        layer, probe = largs.layer, None
        X = raw
    if not args.mock:
        tag = str(args.model).replace("/", "_")
        cache = _CACHE / f"llm_{tag}_L{layer}_cube8.npy"
        _CACHE.mkdir(exist_ok=True)
        np.save(cache, X)
        print(f"cached LLM vectors -> {cache.name}")
    return X, layer, probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki")
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=None,
                    help="LLM layer; default = held-out attribute probe")
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--n-splits", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--probe-seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true",
                    help="planted LLM cube (no transformers)")
    ap.add_argument("--out", default="cube3_corpus_llm")
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()
    if not (args.demo or args.wiki or args.text):
        if Path(_DEFAULT_TEXT8).is_file():
            args.text = [_DEFAULT_TEXT8]
        else:
            raise SystemExit("provide --demo, --wiki, or --text")

    order = list(cube_case.WORDS)
    targets = [cube_case.WORDS[p] for p in order]
    pts = np.array(order)

    Mc = corpus_mstar(args, targets)
    Rc, r2c = reynolds(Mc, pts)
    print(f"corpus invariance R^2 = {r2c:.3f}")
    Ec, Thc, names_c, ovls_c = match_walsh(Mc, pts)
    print("word embedding matched character / overlap  (g=0, n=1, r=2):")
    for nm, ov in zip(names_c, ovls_c):
        print(f"  {nm.replace('n', 'p'):>3}  {subset_label(nm):>16}  {ov:.3f}")

    X, layer, probe = llm_vectors(args, targets, pts, args.out)
    Ml = gram_from_vectors(X)
    _, r2l = reynolds(Ml, pts)
    print(f"LLM invariance R^2 = {r2l:.3f}  (layer {layer})")
    El, Thl, names_l, ovls_l = match_walsh(Ml, pts)
    print("LLM matched character / overlap  (g=0, n=1, r=2):")
    for nm, ov in zip(names_l, ovls_l):
        print(f"  {nm.replace('n', 'p'):>3}  {subset_label(nm):>16}  {ov:.3f}")
    if probe is not None:
        print(f"held-out probe R^2 = {probe['r2']:.3f}")

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "font.size": 20, "axes.titlesize": 24, "axes.labelsize": 22,
        "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
    })
    tick_fs, title_fs, section_fs = 20, 24, 24
    cmap_m, cmap_b = plt.cm.BrBG, plt.cm.RdBu_r
    fig = plt.figure(figsize=(21.8, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.14, 1.0], hspace=0.06,
                          wspace=0.18, left=0.06, right=0.98,
                          top=0.90, bottom=0.26)
    ax_ta = fig.add_subplot(gs[0, 0])
    ax_tb = fig.add_subplot(gs[0, 1])
    ax_ta.axis("off")
    ax_tb.axis("off")
    ax_ta.set_title("(a) Statistical Invariance Holds", fontsize=section_fs, pad=0)
    ax_tb.set_title("(b) Boolean Irreps Follow", fontsize=section_fs, pad=0)
    gs_a = gs[1, 0].subgridspec(1, 2, wspace=0.05)
    gs_b = gs[1, 1].subgridspec(1, 2, wspace=0.05)
    ax = [fig.add_subplot(gs_a[0]), fig.add_subplot(gs_a[1]),
          fig.add_subplot(gs_b[0]), fig.add_subplot(gs_b[1])]
    vmax = float(np.max(np.abs(Mc)))
    draw_matrix(ax[0], Mc, targets, vmax, cmap_m,
                r"$\mathrm{PM}^{\!\star}\!\mathrm{P}$",
                tick_fs=tick_fs, title_fs=title_fs)
    im = draw_matrix(ax[1], Rc, targets, vmax, cmap_m,
                     r"$\Pi_{\mathbb{Z}_2^3}\,\mathrm{PM}^{\!\star}\!\mathrm{P}$",
                     ytick=False, tick_fs=tick_fs, title_fs=title_fs)
    vm = max(float(np.max(np.abs(np.stack([Ec, Thc, El, Thl])))), 1e-12)
    norm = Normalize(vmin=-vm, vmax=vm)
    if args.mock:
        llm_sub = "LLM"
    elif "qwen" in args.model.lower():
        llm_sub = "qwen"
    else:
        llm_sub = args.model.split("/")[-1]
    names_c_s = [subset_label(nm) for nm in names_c]
    names_l_s = [subset_label(nm) for nm in names_l]
    draw_overlay(ax[2], Ec, Thc, names_c_s, ovls_c, targets, norm, cmap_b,
                 "word embedding", xlabel=False, show_overlap=False,
                 tick_fs=tick_fs, title_fs=title_fs, xtick_fs=20, xrot=45)
    draw_overlay(ax[3], El, Thl, names_l_s, ovls_l, targets, norm, cmap_b,
                 llm_sub, ytick=False, xlabel=False, show_overlap=False,
                 tick_fs=tick_fs, title_fs=title_fs, xtick_fs=20, xrot=45)
    cbar1 = fig.colorbar(im, ax=ax[:2], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=20)
    sm = ScalarMappable(norm=norm, cmap=cmap_b)
    sm.set_array([])
    cbar2 = fig.colorbar(sm, ax=ax[2:], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=20)
    fig.canvas.draw()
    p = [a.get_position() for a in ax]
    h_emp = Line2D([0], [0], marker="s", color="none",
                   markerfacecolor="0.35", markeredgecolor="none", ms=16)
    h_th = Line2D([0], [0], marker="s", color="none",
                  markerfacecolor="none", markeredgecolor="0.35",
                  markeredgewidth=2.6, ms=18)
    cx_b = 0.5 * (p[2].x0 + p[3].x1)
    fig.legend([h_emp, h_th], ["empirical", "theory"],
               loc="upper center", ncol=2, fontsize=22, frameon=False,
               bbox_to_anchor=(cx_b, p[2].y0 - 0.14),
               bbox_transform=fig.transFigure)
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=170, bbox_inches="tight")
    print(f"saved -> {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
