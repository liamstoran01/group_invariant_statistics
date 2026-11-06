"""
sky_wordvecs.py
---------------
Train Word2Vec and GloVe from scratch on the four star-guide books, then
make the same 3D overlay as panel (c) of celestial_main_figure.py (Books
side): empirical dipole coords vs spherical harmonics at true positions.

Word2Vec: skip-gram + negative sampling in PyTorch (GPU if CUDA).
GloVe: AdaGrad factorization of windowed co-occurrence in PyTorch.
Both use the tokenized book streams (sky objects already folded to
canonical tokens by skylib).

Dipole: leading 3 signed eigenmodes of the density-weighted embedding
Gram vs true (x,y,z); rotation only (no global scale).
Quadrupole: modes 4-8 vs the ℓ=2 harmonic block. Prints mean/median
per-object cosine and a label-shuffle null, matching the Books overlay.

Writes wordvec_sky_overlay.pdf/.png (Word2Vec | GloVe).

Usage:  python3 sky_wordvecs.py
            [--dim 50] [--epochs 100] [--seed 0] [--nperm 300]
            [--out wordvec_sky_overlay.pdf]
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import skylib as sk
from celestial_main_figure import (
    draw_overlay, fill_overlay_frame, save_both, style_axes,
)
from sky_overlay_harmonics import (
    VIEWS, harmonic_bases, procrustes, fixed_slice, null_stats,
    weighted_modes_signed,
)


def torch_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"torch device: cuda ({name})")
        return torch.device("cuda")
    print("torch device: cpu")
    return torch.device("cpu")


def encode_streams(streams):
    vocab = {}
    encoded = []
    for s in streams:
        ids = np.empty(len(s), dtype=np.int32)
        for i, t in enumerate(s):
            j = vocab.get(t)
            if j is None:
                j = len(vocab)
                vocab[t] = j
            ids[i] = j
        encoded.append(ids)
    return vocab, encoded


def window_pairs(encoded, window, both_dirs=True):
    """All (left, right) index pairs at distance 1..window, skipping equals."""
    left, right = [], []
    for ids in encoded:
        if ids.size < 2:
            continue
        for d in range(1, window + 1):
            a, b = ids[:-d], ids[d:]
            m = a != b
            a, b = a[m], b[m]
            left.append(a)
            right.append(b)
            if both_dirs:
                left.append(b)
                right.append(a)
    if not left:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    return np.concatenate(left), np.concatenate(right)


def train_word2vec(vocab, encoded, window, dim, epochs, seed, device,
                   n_neg=5, batch=65536, lr=0.01):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    V = len(vocab)
    ntok = int(sum(s.size for s in encoded))
    centers, contexts = window_pairs(encoded, window, both_dirs=True)
    print(f"Word2Vec skip-gram (torch): {len(encoded)} books, {ntok} tokens, "
          f"{len(centers)} pairs, dim {dim}, window {window}, {epochs} "
          f"epochs, {n_neg} neg, seed {seed}")
    counts = np.bincount(np.concatenate(encoded), minlength=V).astype(np.float64)
    noise = np.power(np.maximum(counts, 1.0), 0.75)
    noise /= noise.sum()
    noise_t = torch.tensor(noise, device=device, dtype=torch.float32)
    in_emb = torch.nn.Embedding(V, dim).to(device)
    out_emb = torch.nn.Embedding(V, dim).to(device)
    scale = 0.5 / dim
    torch.nn.init.uniform_(in_emb.weight, -scale, scale)
    torch.nn.init.uniform_(out_emb.weight, -scale, scale)
    opt = torch.optim.Adam(list(in_emb.parameters()) + list(out_emb.parameters()),
                           lr=lr)
    ctr = torch.tensor(centers, device=device, dtype=torch.long)
    ctx = torch.tensor(contexts, device=device, dtype=torch.long)
    n = ctr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        nb = 0
        for start in range(0, n, batch):
            ix = perm[start:start + batch]
            c, o = ctr[ix], ctx[ix]
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
            total += float(loss.detach())
            nb += 1
        if ep == 0 or (ep + 1) % max(epochs // 5, 1) == 0:
            print(f"  epoch {ep + 1}/{epochs}  loss {total / max(nb, 1):.4g}")
    return in_emb.weight.detach().float().cpu().numpy()


def train_glove(vocab, encoded, window, dim, n_iter, seed, device,
                xmax=100.0, alpha=0.75, lr=0.05):
    torch.manual_seed(seed)
    V = len(vocab)
    a, b = window_pairs(encoded, window, both_dirs=False)
    key = a.astype(np.int64) * V + b.astype(np.int64)
    uniq, cnt = np.unique(key, return_counts=True)
    ii_u = (uniq // V).astype(np.int64)
    jj_u = (uniq % V).astype(np.int64)
    ii_np = np.concatenate([ii_u, jj_u])
    jj_np = np.concatenate([jj_u, ii_u])
    x_np = np.concatenate([cnt, cnt]).astype(np.float32)
    ii = torch.tensor(ii_np, device=device, dtype=torch.long)
    jj = torch.tensor(jj_np, device=device, dtype=torch.long)
    x = torch.tensor(x_np, device=device)
    f = torch.clamp((x / xmax) ** alpha, max=1.0)
    logx = torch.log(x)
    scale = 0.5 / dim
    W = torch.empty(V, dim, device=device).uniform_(-scale, scale)
    Wt = torch.empty(V, dim, device=device).uniform_(-scale, scale)
    b = torch.zeros(V, device=device)
    bt = torch.zeros(V, device=device)
    gW = torch.ones_like(W)
    gWt = torch.ones_like(Wt)
    gb = torch.ones_like(b)
    gbt = torch.ones_like(bt)
    print(f"GloVe (torch): {V} tokens, {ii.shape[0]} directed pairs, dim {dim}, "
          f"{n_iter} iters, seed {seed}")
    pair_batch = 1 << 20
    P = ii.shape[0]
    for it in range(n_iter):
        dW = torch.zeros_like(W)
        dWt = torch.zeros_like(Wt)
        db = torch.zeros_like(b)
        dbt = torch.zeros_like(bt)
        sse = 0.0
        for start in range(0, P, pair_batch):
            sl = slice(start, min(start + pair_batch, P))
            iib, jjb = ii[sl], jj[sl]
            pred = (W[iib] * Wt[jjb]).sum(1) + b[iib] + bt[jjb]
            diff = pred - logx[sl]
            coef = 2.0 * f[sl] * diff
            dW.index_add_(0, iib, coef[:, None] * Wt[jjb])
            dWt.index_add_(0, jjb, coef[:, None] * W[iib])
            db.index_add_(0, iib, coef)
            dbt.index_add_(0, jjb, coef)
            sse += float((f[sl] * diff.square()).sum())
        gW += dW.square()
        gWt += dWt.square()
        gb += db.square()
        gbt += dbt.square()
        W -= lr * dW / gW.sqrt()
        Wt -= lr * dWt / gWt.sqrt()
        b -= lr * db / gb.sqrt()
        bt -= lr * dbt / gbt.sqrt()
        if it == 0 or (it + 1) % max(n_iter // 5, 1) == 0:
            print(f"  iter {it + 1}/{n_iter}  weighted SSE {sse:.4g}")
    return (W + Wt).detach().float().cpu().numpy()


def object_matrix(vocab, X, names, keep, canon):
    """n_keep x D matrix in catalogue order of keep."""
    rows = []
    missing = []
    for i in keep:
        tok = canon[names[i]]
        j = vocab.get(tok)
        if j is None:
            missing.append(names[i])
        else:
            rows.append(X[j])
    if missing:
        raise SystemExit(f"missing embeddings for {missing[:8]}"
                         f"{'...' if len(missing) > 8 else ''}")
    return np.stack(rows, 0).astype(np.float64)


def gram_modes(X, sw, k=12):
    Xc = X - X.mean(0, keepdims=True)
    G = 0.5 * ((Xc @ Xc.T) + (Xc @ Xc.T).T)
    return weighted_modes_signed(G, sw, k)


def report_dipole_quad(name, U, lam, sw, xyz, Bb, rng, nperm):
    A, Ba, cm, cmed = fixed_slice(U, lam, sw, slice(0, 3), xyz,
                                  scale=False)
    nmu, n95 = null_stats(
        lambda Us: fixed_slice(Us, lam, sw, slice(0, 3), xyz,
                               scale=False)[2],
        U, sw, rng, nperm)
    print(f"{name} dipole (modes 1-3 vs true positions, "
          f"{U.shape[0]} objects; rotation only):")
    print(f"    per-object cos {cm:.2f} (median {cmed:.2f}); "
          f"null {nmu:.2f} (95% {n95:.2f})")
    A2 = (U / sw)[:, 3:8] * np.sqrt(np.abs(lam[3:8]))
    _, cq, _ = procrustes(A2, Bb[2] / sw, scale=False)
    nmu_q, n95_q = null_stats(
        lambda Us: float(np.mean(procrustes(
            (Us / sw)[:, 3:8] * np.sqrt(np.abs(lam[3:8])),
            Bb[2] / sw, scale=False)[1])),
        U, sw, rng, nperm)
    print(f"{name} quadrupole (modes 4-8 vs ℓ=2): "
          f"cos {np.mean(cq):.2f} (median {np.median(cq):.2f}); "
          f"null {nmu_q:.2f} (95% {n95_q:.2f})")
    return A, Ba, cm, cmed, float(np.mean(cq)), float(np.median(cq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=100,
                    help="Word2Vec epochs and GloVe AdaGrad iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nperm", type=int, default=300)
    ap.add_argument("--window", type=int, default=sk.WINDOW,
                    help="skip-gram / GloVe window (default: skylib WINDOW)")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="catalogue names to omit from the overlay/fit")
    ap.add_argument("--out", default="wordvec_sky_overlay.pdf")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(0)
    names, lat, lon, streams, canon = sk.corpus_stream()
    keep, Mb, p, ntok = sk.corpus_mstar(names, streams, canon)
    if args.drop:
        drop = {n.lower() for n in args.drop}
        keep = [i for i in keep if names[i].lower() not in drop]
        print(f"dropped {sorted(drop)}; {len(keep)} objects remain")
    latb, lonb = lat[keep], lon[keep]
    xyzb = sk.unit_vectors(latb, lonb)
    wb, _ = sk.density_weights(xyzb)
    swb = np.sqrt(wb)[:, None]
    Bb = harmonic_bases(latb, lonb, wb)
    print(f"books: {sk.BOOKS}  tokens {ntok}  kept objects {len(keep)}")
    device = torch_device()
    vocab, encoded = encode_streams(streams)

    Xw_all = train_word2vec(vocab, encoded, args.window, args.dim,
                            args.epochs, args.seed, device)
    Xw = object_matrix(vocab, Xw_all, names, keep, canon)
    Xg_all = train_glove(vocab, encoded, args.window, args.dim,
                         args.epochs, args.seed, device)
    Xg = object_matrix(vocab, Xg_all, names, keep, canon)

    lam_w, U_w = gram_modes(Xw, swb, 12)
    lam_g, U_g = gram_modes(Xg, swb, 12)
    Aw, Baw, *_ = report_dipole_quad(
        "Word2Vec", U_w, lam_w, swb, xyzb, Bb, rng, args.nperm)
    Ag, Bag, *_ = report_dipole_quad(
        "GloVe", U_g, lam_g, swb, xyzb, Bb, rng, args.nperm)

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    norm = Normalize(vmin=float(lonb.min()), vmax=float(lonb.max()))
    fig = plt.figure(figsize=(13, 5.8))
    gs = fig.add_gridspec(1, 2, wspace=0.02)
    ax_w = fig.add_subplot(gs[0, 0], projection="3d")
    ax_g = fig.add_subplot(gs[0, 1], projection="3d")
    elev, azim = VIEWS[0]
    for ax, A, Ba, title in (
        (ax_w, Aw, Baw, "Word2Vec"),
        (ax_g, Ag, Bag, "GloVe"),
    ):
        ax.computed_zorder = False
        draw_overlay(ax, A, Ba, lonb, norm)
        ax.view_init(elev=elev, azim=azim)
        fill_overlay_frame(ax, A, Ba)
        style_axes(ax)
        ax.set_title(title, fontsize=21, pad=2)
        ax.title.set_x(0.46)

    sm = ScalarMappable(norm=norm, cmap="hsv")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_w, ax_g], shrink=0.75)
    cbar.set_label("true RA (deg)", fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="gray", markeredgecolor="gray",
               markersize=9, label="empirical"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="none", markeredgecolor="gray",
               markersize=9, markeredgewidth=1.4, label="theory"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=14,
               frameon=False, bbox_to_anchor=(0.45, 0.08))
    save_both(fig, args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
