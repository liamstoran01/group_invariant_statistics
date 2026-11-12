"""
heatmap_wordvecs.py
-------------------
Bach PC1–PC2 theory overlay, but the filled (empirical) points come from
Word2Vec and GloVe trained from scratch on the Bach chorale sequences
(not pretrained Google News / Wikipedia vectors).

Theory (open circles) is unchanged: the leading D12 irrep mode of Bach
M*, same as the Bach panel in heatmap_chordbert.py / heatmap_theory.py.
Empirical (filled) is the leading 2D plane of the 24 triad vectors
(top-2 eigenplane of the centered Gram). Orthogonal Procrustes aligns
theory onto that plane; an isotropic scale is applied only for the
figure (theory already carries sqrt(lam), so the predicted scale is 1).

Also prints the appendix diagnostic: first 6 empirical dimensions vs the
top-3 D12 theory modes, one orthogonal Procrustes per 2D plane, for each
of Word2Vec and GloVe (mean / median cosine overall and per plane).

Training (same --sequences / --window as M*):
  * Word2Vec: skip-gram + negative sampling in PyTorch.
  * GloVe: weighted log-co-occurrence factorization of chordlib's
    windowed count matrix (AdaGrad, Pennington et al.).

Writes one two-panel figure (Word2Vec left, GloVe right):
  wordvec_planes.pdf/.png

Usage:  python3 heatmap_wordvecs.py
            [--sequences chord_sequences_full.json] [--window 3]
            [--dim 50] [--epochs 100] [--glove-max-iter 2000]
            [--seed 0] [--out wordvec_planes]
"""
import argparse
import json
import os

import numpy as np

import chordlib as cl
import heatmap_theory as fht


def chord_cosines(E, T):
    cos = [E[i] @ T[i] /
           max(np.linalg.norm(E[i]) * np.linalg.norm(T[i]), 1e-12)
           for i in range(cl.N)]
    return float(np.mean(cos)), float(np.median(cos)), float(np.min(cos))


def train_word2vec(seqs, window, dim, epochs, seed, n_neg=5, lr=0.01):
    import torch
    import torch.nn.functional as F

    sents = [np.asarray(s, dtype=np.int64) for s in seqs if len(s) > 1]
    V = int(max(s.max() for s in sents)) + 1
    left, right = [], []
    for ids in sents:
        for d in range(1, window + 1):
            a, b = ids[:-d], ids[d:]
            m = a != b
            left.append(a[m])
            right.append(b[m])
            left.append(b[m])
            right.append(a[m])
    ctr = np.concatenate(left)
    ctx = np.concatenate(right)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    print(f"Word2Vec skip-gram (torch {device}): {len(sents)} sequences, "
          f"{len(ctr)} pairs, dim {dim}, window {window}, {epochs} epochs, "
          f"seed {seed}")
    counts = np.bincount(np.concatenate(sents), minlength=V).astype(np.float64)
    noise = np.power(np.maximum(counts, 1.0), 0.75)
    noise /= noise.sum()
    noise_t = torch.tensor(noise, device=device, dtype=torch.float32)
    in_emb = torch.nn.Embedding(V, dim).to(device)
    out_emb = torch.nn.Embedding(V, dim).to(device)
    scale = 0.5 / dim
    torch.nn.init.uniform_(in_emb.weight, -scale, scale)
    torch.nn.init.uniform_(out_emb.weight, -scale, scale)
    opt = torch.optim.Adam(
        list(in_emb.parameters()) + list(out_emb.parameters()), lr=lr)
    ctr_t = torch.tensor(ctr, device=device, dtype=torch.long)
    ctx_t = torch.tensor(ctx, device=device, dtype=torch.long)
    n, batch = ctr_t.shape[0], min(65536, ctr_t.shape[0])
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch):
            ix = perm[start:start + batch]
            c, o = ctr_t[ix], ctx_t[ix]
            B = c.shape[0]
            v_c = in_emb(c)
            pos = (v_c * out_emb(o)).sum(1)
            neg = torch.multinomial(noise_t, B * n_neg, replacement=True)
            v_n = out_emb(neg.view(B, n_neg))
            neg_s = (v_c.unsqueeze(1) * v_n).sum(2)
            loss = (-F.logsigmoid(pos).mean()
                    - F.logsigmoid(-neg_s).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    X = in_emb.weight.detach().float().cpu().numpy()
    missing = [i for i in range(cl.N) if counts[i] == 0]
    if missing:
        raise SystemExit(f"Word2Vec missing triads {missing}")
    return X[:cl.N]


def train_glove(seqs, window, dim, seed, xmax=100.0, alpha=0.75, lr=0.05,
                max_iter=2000, tol=1e-3, patience=5):
    """GloVe on the same symmetric windowed counts used for M*.

    AdaGrad until relative SSE drop < tol for `patience` steps (same
    plateau rule as cube_wordvecs), capped at max_iter.
    """
    V = max(max(s) for s in seqs) + 1
    C = cl.cooccurrence_counts(seqs, V, window)
    ii, jj = np.where(C > 0)
    x = C[ii, jj]
    f = np.minimum(1.0, (x / xmax) ** alpha)
    logx = np.log(x)
    rng = np.random.default_rng(seed)
    scale = 0.5 / dim
    W = rng.normal(0.0, scale, (V, dim))
    Wt = rng.normal(0.0, scale, (V, dim))
    b = np.zeros(V)
    bt = np.zeros(V)
    gW = np.ones_like(W)
    gWt = np.ones_like(Wt)
    gb = np.ones(V)
    gbt = np.ones(V)
    print(f"GloVe: {V} tokens, {len(x)} nonzero pairs, dim {dim}, "
          f"max_iter {max_iter}, tol {tol:g}, seed {seed}")
    sse_prev = None
    flat = 0
    last_it = 0
    for it in range(max_iter):
        pred = np.sum(W[ii] * Wt[jj], 1) + b[ii] + bt[jj]
        diff = pred - logx
        sse = float(np.sum(f * diff ** 2))
        rel = (abs(sse_prev - sse) / max(sse_prev, 1e-12)
               if sse_prev is not None else 1.0)
        last_it = it + 1
        if last_it <= 5 or last_it % 50 == 0:
            print(f"  iter {last_it}/{max_iter}  weighted SSE {sse:.4g}  "
                  f"rel {rel:.3g}")
        coef = 2.0 * f * diff
        dW = np.zeros_like(W)
        dWt = np.zeros_like(Wt)
        db = np.zeros(V)
        dbt = np.zeros(V)
        np.add.at(dW, ii, coef[:, None] * Wt[jj])
        np.add.at(dWt, jj, coef[:, None] * W[ii])
        np.add.at(db, ii, coef)
        np.add.at(dbt, jj, coef)
        gW += dW ** 2
        gWt += dWt ** 2
        gb += db ** 2
        gbt += dbt ** 2
        W -= lr * dW / np.sqrt(gW)
        Wt -= lr * dWt / np.sqrt(gWt)
        b -= lr * db / np.sqrt(gb)
        bt -= lr * dbt / np.sqrt(gbt)
        sse_prev = sse
        if rel < tol:
            flat += 1
            if flat >= patience:
                print(f"  GloVe converged at iter {last_it}  "
                      f"(rel SSE drop < {tol:g} for {patience} steps); "
                      f"SSE {sse:.4g}")
                break
        else:
            flat = 0
    else:
        print(f"  GloVe hit max_iter={max_iter}  (last rel {rel:.3g}); "
              f"SSE {sse:.4g}")
    X = W[:cl.N] + Wt[:cl.N]
    missing = [i for i in range(cl.N) if C[i].sum() == 0]
    if missing:
        raise SystemExit(f"GloVe missing triads {missing}")
    return X


def bach_theory_plane(M):
    """Leading D12 theory mode, scaled by sqrt(lam); same as Bach panel."""
    r, lam, V = cl.theory_modes(M)[0]
    Th = np.column_stack([np.sqrt(max(lam, 0)) * V[:, a]
                          for a in range(V.shape[1])])
    return r, lam, Th


def align_theory(Emp, Th, scale_plot=True):
    """Orthogonal Procrustes. Theory predicts scale 1 (both sides are
    V sqrt(lam)); an extra isotropic s is only for the figure."""
    R, rel = cl.procrustes(Emp, Th)
    ThR = Th @ R
    s = float(np.sum(Emp * ThR) / max(np.sum(ThR * ThR), 1e-12))
    if scale_plot:
        ThR = ThR * s
    return ThR, s, rel


def gram_coords(X, k):
    Xc = X - X.mean(0, keepdims=True)
    G = Xc @ Xc.T
    lam_e, V_e = cl.embedding_modes(G, k=k)
    Emp = V_e * np.sqrt(np.clip(lam_e, 0, None))
    return Emp, lam_e, V_e


def report_per_plane(name, X, M, k_planes=3):
    """Appendix diagnostic: top-2k empirical coords vs top-k theory modes,
    one orthogonal Procrustes per 2D plane (same as heatmap_chordbert)."""
    k = 2 * k_planes
    Emp, lam_e, V_e = gram_coords(X, k)
    modes = cl.theory_modes(M)[:k_planes]
    if len(modes) < k_planes:
        raise SystemExit(
            f"only {len(modes)} theory modes available; need {k_planes}")
    cols, names, lams = [], [], []
    for r, lam, V in modes:
        if V.shape[1] != 2:
            raise SystemExit(
                f"theory mode [{r}] has width {V.shape[1]}, expected 2 "
                f"for plane-wise Procrustes")
        for a in range(V.shape[1]):
            cols.append(np.sqrt(max(lam, 0)) * V[:, a])
        names.append(r)
        lams.append(lam)
    Th = np.column_stack(cols)

    Th_aligned = np.zeros_like(Th)
    print(f"\n{name} top-{k} embeddings: per-plane Procrustes "
          f"(no global {k}D orthogonal):")
    print(f"  empirical eigenvalues: {np.round(lam_e, 4)}")
    print(f"  theory modes: " +
          ", ".join(f"{names[p]}({lams[p]:+.2f})" for p in range(k_planes)))
    plane_means = []
    for p in range(k_planes):
        sl = slice(2 * p, 2 * p + 2)
        Ep, Tp = Emp[:, sl], Th[:, sl]
        Rp, _ = cl.procrustes(Ep, Tp)
        Tp_a = Tp @ Rp
        Th_aligned[:, sl] = Tp_a
        mean, med, mn = chord_cosines(Ep, Tp_a)
        ov = cl.subspace_overlap(V_e[:, sl], modes[p][2])
        plane_means.append(mean)
        print(f"  plane {p + 1} [{names[p]}]  PC{2 * p + 1}-{2 * p + 2}: "
              f"mean cos {mean:.3f}, median {med:.3f}, min {mn:.3f}, "
              f"overlap {ov:.3f}")

    mean, med, mn = chord_cosines(Emp, Th_aligned)
    print(f"  full {k}D after plane-wise R's: "
          f"mean cos {mean:.3f}, median {med:.3f}, min {mn:.3f}")
    print(f"  {name} appendix line: {mean:.2f} (median {med:.2f}) overall; "
          f"planes " + ", ".join(f"{m:.2f}" for m in plane_means))
    return Emp[:, :2], Th_aligned[:, :2]


def fit_plane(name, X, M, scale_plot=True):
    print(f"\n=== {name} ===")
    Emp, lam_e, _ = gram_coords(X, 2)
    r, lam, Th = bach_theory_plane(M)
    Th, s, rel = align_theory(Emp, Th, scale_plot=scale_plot)
    mean, med, mn = chord_cosines(Emp, Th)
    print(f"theory mode [{r}] lam {lam:+.2f}; plot scale {s:.4g} "
          f"(cosine is scale-invariant); rel residual {rel:.3f}")
    print(f"empirical Gram eigs: {np.round(lam_e, 4)}")
    print(f"chord cos: mean {mean:.3f}, median {med:.3f}, min {mn:.3f}")
    return Emp, Th


def save_two_panel(Emp_w, Th_w, Emp_g, Th_g, out_stem, adjust_text, plt,
                   Line2D):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6))
    packs = [
        fht.theory_panel(axes[0], Emp_w, Th_w, 0),
        fht.theory_panel(axes[1], Emp_g, Th_g, 0),
    ]
    axes[0].set_title("Word2Vec", fontsize=20, pad=12)
    axes[1].set_title("GloVe", fontsize=20, pad=12)
    legend_handles = [
        Line2D([0], [0], color="tab:red", lw=2.5, label="major"),
        Line2D([0], [0], color="tab:blue", lw=2.5, label="minor"),
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=9, markerfacecolor="gray", label="empirical"),
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=10, markerfacecolor="none",
               markeredgecolor="gray", markeredgewidth=1.5, label="theory"),
    ]
    fig.canvas.draw()
    b0, b1 = axes[0].get_position(), axes[1].get_position()
    legend_x = 0.5 * (b0.x0 + b1.x1)
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               fontsize=16, frameon=False,
               bbox_to_anchor=(legend_x, b0.y0 - 0.04),
               handletextpad=0.3)
    fig.canvas.draw()
    for texts, E in packs:
        adjust_text(
            texts, x=E[:, 0], y=E[:, 1], ax=texts[0].axes,
            force_text=(0.5, 0.5), force_static=(0.2, 0.2),
            force_explode=(0.0, 0.0), expand=(1.2, 1.2),
            ensure_inside_axes=True,
        )
    pdf = out_stem if out_stem.lower().endswith(".pdf") else out_stem + ".pdf"
    png = pdf[:-4] + ".png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved -> {pdf}")
    print(f"saved -> {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="chord_sequences_full.json")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--dim", type=int, default=50,
                    help="Word2Vec / GloVe embedding dimension")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Word2Vec skip-gram epochs")
    ap.add_argument("--glove-max-iter", type=int, default=2000,
                    help="GloVe AdaGrad cap; stops early on SSE plateau")
    ap.add_argument("--glove-tol", type=float, default=1e-3)
    ap.add_argument("--glove-patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-planes", type=int, default=3,
                    help="theory / empirical planes for the cosine report "
                         "(default 3 = first 6 dimensions)")
    ap.add_argument("--out", default="wordvec_planes")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 14,
    })

    if not os.path.exists(args.sequences):
        raise SystemExit(
            f"missing {args.sequences}; run extract_chords.py --vocab full")

    seqs = json.load(open(args.sequences))
    M = cl.triad_block(*cl.mstar(seqs, window=args.window))
    print(f"sequences: {args.sequences}  ({len(seqs)} pieces)  "
          f"D12 R^2: {cl.r2(M, cl.d12_labels()):.3f}")

    need = 2 * args.k_planes
    if args.dim < need:
        print(f"warning: --dim {args.dim} < {need}; higher planes of the "
              f"24x24 Gram will be weak")

    X_w = train_word2vec(seqs, args.window, args.dim, args.epochs, args.seed)
    X_g = train_glove(seqs, args.window, args.dim, args.seed,
                      max_iter=args.glove_max_iter, tol=args.glove_tol,
                      patience=args.glove_patience)
    Emp_w, Th_w = fit_plane("Word2Vec", X_w, M)
    Emp_g, Th_g = fit_plane("GloVe", X_g, M)
    report_per_plane("Word2Vec", X_w, M, k_planes=args.k_planes)
    report_per_plane("GloVe", X_g, M, k_planes=args.k_planes)
    save_two_panel(Emp_w, Th_w, Emp_g, Th_g, args.out, adjust_text, plt,
                   Line2D)


if __name__ == "__main__":
    main()
