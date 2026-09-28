"""
sky_overlay_harmonics.py
------------------------
The 1x2 overlay figure (books | LLM, one camera view each, shared RA
colorbar), with the comparison target now the DIPOLE SPHERICAL HARMONICS
EVALUATED AT THE OBJECTS' TRUE POSITIONS --- no kernel eigendecomposition
anywhere. The corpus kernel plays no role in either panel's target; each
is "embeddings vs harmonics at truth." Books fit only a rotation; the LLM
also fits one overall scale.

Instruments (matching the paper caption):
  * Books (left):  strict slice --- leading 3 eigenmodes of the books'
                   raw M* taken as-is.
  * LLM (right):   located subspace --- best-aligned 3-dim dipole
                   subspace within modes 1-4 (leading mode is partially
                   hybridized with a non-geometric direction); the
                   label-shuffle null repeats the search.

Pinning: by default the books' M* diagonal (a self-co-occurrence
statistic the kernel does not constrain) is pinned to the fitted
kernel's c(1) before factorization. --no-pin keeps the measured
diagonal, to quantify the effect of pinning.

Usage: python3 sky_overlay_harmonics.py [--model mistrallarge123b]
           [--layer 72] [--no-pin] [--nperm 300]
           [--out sky_overlay_harmonics.pdf]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

import skylib as sk

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

VIEWS = [(18, -60), (18, 120)]  # default camera: first entry
POOL_DIPOLE = list(range(0, 4))      # LLM: modes 1-4 (0-indexed 0-3)
POOL_QUAD = list(range(4, 12))       # LLM: modes 5-12, dipole-deflated


def harmonic_bases(lat, lon, w):
    Y = sk.real_harmonics(lat, lon, lmax=2)
    return sk.weighted_degree_bases(Y, w)


def procrustes(A, cfg, scale=True):
    """One rotation of cfg onto A (and optionally one overall scale);
    returns aligned target, per-object cosines, and the rotation."""
    if scale:
        Bs = cfg * (np.linalg.norm(A) / max(np.linalg.norm(cfg), 1e-12))
    else:
        Bs = np.asarray(cfg, dtype=float)
    U2, _, V2 = np.linalg.svd(Bs.T @ A)
    R = U2 @ V2
    Ba = Bs @ R
    c = np.sum(A * Ba, 1) / (np.linalg.norm(A, axis=1) *
                             np.linalg.norm(Ba, axis=1) + 1e-12)
    return Ba, c, R


def fixed_slice(U, lam, sw, sl, cfg, scale=True):
    A = (U / sw)[:, sl] * np.sqrt(np.abs(lam[sl]))
    Ba, c, _ = procrustes(A, cfg, scale=scale)
    return A, Ba, float(np.mean(c)), float(np.median(c))


def located(U, sw, Bl, d, pool, cfg, deflate=None):
    """Canonical subspace search in U[:, pool], then Procrustes cfg."""
    Up = U[:, pool]
    if deflate is not None:
        Up = Up - deflate @ (deflate.T @ Up)
        Q, R = np.linalg.qr(Up)
        Up = Q[:, np.abs(np.diag(R)) > 1e-8]
    _, s, Vt = np.linalg.svd(Bl.T @ Up)
    canon2 = float(np.mean(s[:d] ** 2))
    loc = Up @ Vt.T[:, :d]
    Q, _ = np.linalg.qr(loc)
    Qd = Q[:, :d]
    A = Qd / sw
    Ba, c, _ = procrustes(A, cfg)
    return A, Ba, float(np.mean(c)), float(np.median(c)), canon2, Qd


def null_stats(fn, U, sw, rng, nperm):
    """Label-shuffle null: permute object rows of U/sw (the embedding),
    rerun the FULL statistic (search included where fn searches)."""
    vals = []
    for _ in range(nperm):
        pi = rng.permutation(U.shape[0])
        vals.append(fn(sw * (U / sw)[pi]))
    return float(np.mean(vals)), float(np.percentile(vals, 95))


def zonal_r2(M, cosT, wb):
    """Corpus analog of Gram zonality: density-weighted off-diagonal
    variance of M* explained by the fitted zonal kernel."""
    c_hat = sk.fit_zonal(M, cosT, wb)
    pred = np.interp(np.clip(cosT, sk.XQ[0], sk.XQ[-1]), sk.XQ, c_hat)
    iu = np.triu_indices(M.shape[0], 1)
    pw = (wb[:, None] * wb[None, :])[iu]
    y, yh = M[iu], pred[iu]
    ybar = np.sum(pw * y) / pw.sum()
    return 1.0 - np.sum(pw * (y - yh) ** 2) / np.sum(pw * (y - ybar) ** 2)


def weighted_modes_signed(M, sw, k=None):
    """Like sk.weighted_modes, but sorted by decreasing signed λ
    (not |λ|). Needed for indefinite books M* so the quadrupole
    strict slice prefers positive eigenvalues."""
    lam, U = sk.weighted_modes(M, sw, k=None)
    o = np.argsort(-lam)
    if k is not None:
        o = o[:k]
    return lam[o], U[:, o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistrallarge123b")
    ap.add_argument("--layer", type=int, default=72)
    ap.add_argument("--no-pin", action="store_true",
                    help="keep measured M* diagonals instead of pinning "
                         "to the fitted kernel's c(1)")
    ap.add_argument("--nperm", type=int, default=300,
                    help="label-shuffle null draws (default: 300)")
    ap.add_argument("--out", default="sky_overlay_harmonics.pdf")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    # ---------------- (a) books: strict slice vs harmonics ----------------
    names, lat, lon, acc, canon = sk.corpus_stream()
    keep, Mb, p, ntok = sk.corpus_mstar(names, acc, canon)
    latb = lat[keep]
    lonb = lon[keep]
    xyzb = sk.unit_vectors(latb, lonb)
    wb, cosb = sk.density_weights(xyzb)
    swb = np.sqrt(wb)[:, None]
    Bb = harmonic_bases(latb, lonb, wb)
    c_fit = sk.fit_zonal(Mb, cosb, wb)
    if args.no_pin:
        Mb_f = 0.5 * (Mb + Mb.T)
        pin_note = "UNPINNED (measured diagonal)"
    else:
        Mb_f = sk.pin_diagonal(Mb, c_fit)
        pin_note = "pinned to c(1)"
    Mb_s = 0.5 * (Mb + Mb.T)              # zonality on the UNPINNED M*
    r2 = zonal_r2(Mb_s, cosb, wb)
    null = [zonal_r2(Mb_s, cosb[np.ix_(pi, pi)], wb)
            for pi in (rng.permutation(Mb_s.shape[0])
                       for _ in range(args.nperm))]
    null = np.array(null)
    print(f"BOOKS corpus zonality: R^2 = {r2:.3f} "
          f"(null mean {null.mean():.3f}, 95% {np.percentile(null, 95):.3f}, "
          f"p = {(np.sum(null >= r2) + 1) / (args.nperm + 1):.4f})   [model: 0.49]")
    lam_be, U_be = weighted_modes_signed(Mb_f, swb, 12)
    A_b, Ba_b, cm, cmed = fixed_slice(U_be, lam_be, swb, slice(0, 3),
                                      xyzb, scale=False)
    nmu, n95 = null_stats(
        lambda Us: fixed_slice(Us, lam_be, swb, slice(0, 3), xyzb,
                               scale=False)[2],
        U_be, swb, rng, args.nperm)
    print(f"(a) BOOKS strict slice (modes 1-3 by decreasing signed λ, "
          f"{pin_note}) vs dipole harmonics at true positions "
          f"({len(keep)} objects; rotation only, no global scale):")
    print(f"    per-object cos {cm:.2f} (median {cmed:.2f}); "
          f"null {nmu:.2f} (95% {n95:.2f})")
    A2 = (U_be / swb)[:, 3:8] * np.sqrt(np.abs(lam_be[3:8]))
    _, cq, _ = procrustes(A2, Bb[2] / swb, scale=False)
    nmu_q, n95_q = null_stats(
        lambda Us: float(np.mean(procrustes(
            (Us / swb)[:, 3:8] * np.sqrt(np.abs(lam_be[3:8])),
            Bb[2] / swb, scale=False)[1])),
        U_be, swb, rng, args.nperm)
    print(f"    quadrupole strict slice (modes 4-8 by decreasing signed "
          f"λ): cos {np.mean(cq):.2f} (median {np.median(cq):.2f}); "
          f"null {nmu_q:.2f} (95% {n95_q:.2f})")

    # ---------------- (b) LLM: located dipole vs harmonics ----------
    latL, lonL, xyzL, lamL, UL, _ = sk.llm_modes(args.model, args.layer)
    wL, cosL = sk.density_weights(xyzL)
    swL = np.sqrt(wL)[:, None]
    BL = harmonic_bases(latL, lonL, wL)
    A_L, Ba_L, cmL, cmedL, canon2, Qd = located(
        UL, swL, BL[1], 3, POOL_DIPOLE, xyzL)
    nmuL, n95L = null_stats(
        lambda Us: located(Us, swL, BL[1], 3, POOL_DIPOLE, xyzL)[2],
        UL, swL, rng, args.nperm)
    print(f"\n(b) LLM located dipole (within modes "
          f"{POOL_DIPOLE[0]+1}-{POOL_DIPOLE[-1]+1}) vs dipole harmonics "
          f"at true positions ({len(lonL)} objects, {args.model} layer "
          f"{args.layer}):")
    print(f"    canon^2 {canon2:.2f}; per-object cos {cmL:.2f} "
          f"(median {cmedL:.2f}); null {nmuL:.2f} (95% {n95L:.2f}) "
          f"[null repeats the search]")
    _, _, cmq, cmedq, c2q, _ = located(UL, swL, BL[2], 5, POOL_QUAD,
                                       BL[2] / swL, deflate=Qd)

    def llm_quad_mean_cos(Us):
        _, _, _, _, _, Qd_p = located(Us, swL, BL[1], 3, POOL_DIPOLE,
                                      xyzL)
        return located(Us, swL, BL[2], 5, POOL_QUAD, BL[2] / swL,
                       deflate=Qd_p)[2]

    nmu_qL, n95_qL = null_stats(llm_quad_mean_cos, UL, swL, rng,
                                args.nperm)
    print(f"    quadrupole located (modes {POOL_QUAD[0]+1}-"
          f"{POOL_QUAD[-1]+1}, dipole-deflated): canon^2 {c2q:.2f}; "
          f"cos {cmq:.2f} (median {cmedq:.2f}); null {nmu_qL:.2f} "
          f"(95% {n95_qL:.2f}) [null repeats the search]")

    # ---------------- combined 1x2 figure (books | LLM) ----------------
    lon_all = np.concatenate([lonb, lonL])
    norm = Normalize(vmin=float(lon_all.min()), vmax=float(lon_all.max()))
    fig = plt.figure(figsize=(13, 5.8))
    gs = fig.add_gridspec(1, 2, wspace=0.02)
    ax_b = fig.add_subplot(gs[0, 0], projection="3d")
    ax_L = fig.add_subplot(gs[0, 1], projection="3d")

    def style_axes(ax):
        """No grid/ticks; 3-axis triad at the left-back corner, under the data."""
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.computed_zorder = False
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("none")
            axis.pane.set_alpha(0)
            axis.line.set_color((0, 0, 0, 0))
        el, az = np.deg2rad(ax.elev), np.deg2rad(ax.azim)
        cx = np.cos(el) * np.sin(az)
        cy = -np.cos(el) * np.cos(az)
        cz = np.sin(el)
        xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        xb = xl[0] if cx > 0 else xl[1]
        yb = yl[0] if cy > 0 else yl[1]
        zb = zl[0] if cz > 0 else zl[1]
        left_xy = np.array([cy, -cx])
        flip_x = (xl[0] + xl[1] - xb, yb)
        flip_y = (xb, yl[0] + yl[1] - yb)
        if np.dot([flip_x[0] - xb, flip_x[1] - yb], left_xy) >= \
           np.dot([flip_y[0] - xb, flip_y[1] - yb], left_xy):
            xb, yb = flip_x
        else:
            xb, yb = flip_y
        kw = dict(color="0.35", lw=0.9, solid_capstyle="round", zorder=0)
        ax.plot(xl, [yb, yb], [zb, zb], **kw)
        ax.plot([xb, xb], yl, [zb, zb], **kw)
        ax.plot([xb, xb], [yb, yb], zl, **kw)

    def draw(ax, A, Ba, lo):
        for q in range(A.shape[0]):
            ax.plot([Ba[q, 0], A[q, 0]], [Ba[q, 1], A[q, 1]],
                    [Ba[q, 2], A[q, 2]], color="gray", lw=1.0,
                    alpha=0.55, zorder=1)
        sc = ax.scatter(A[:, 0], A[:, 1], A[:, 2], c=lo, cmap="hsv",
                        norm=norm, s=28, zorder=3)
        edge_cols = plt.cm.hsv(norm(lo))
        ax.scatter(Ba[:, 0], Ba[:, 1], Ba[:, 2], facecolors="none",
                   edgecolors=edge_cols, s=40, linewidths=0.9, zorder=3)
        return sc

    elev, azim = VIEWS[0]
    for ax in (ax_b, ax_L):
        ax.computed_zorder = False
    draw(ax_b, A_b, Ba_b, lonb)
    ax_b.view_init(elev=elev, azim=azim)
    style_axes(ax_b)
    ax_b.set_title("Books", fontsize=16, pad=2)
    ax_b.title.set_x(0.46)
    rng_plot = np.random.default_rng(0)
    n_show = min(75, A_L.shape[0])
    idx = rng_plot.choice(A_L.shape[0], size=n_show, replace=False)
    draw(ax_L, A_L[idx], Ba_L[idx], lonL[idx])
    ax_L.view_init(elev=elev, azim=azim)
    style_axes(ax_L)
    ax_L.set_title("Mistral Large 2", fontsize=16, pad=2)
    ax_L.title.set_x(0.46)

    sm = ScalarMappable(norm=norm, cmap="hsv")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_b, ax_L], shrink=0.75)
    cbar.set_label("true RA (deg)", fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="gray", markeredgecolor="gray",
               markersize=9, label="empirical"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="none", markeredgecolor="gray",
               markersize=9, markeredgewidth=1.4,
               label="theory"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=14,
               frameon=False, bbox_to_anchor=(0.45, 0.08))
    base, ext = os.path.splitext(args.out)
    pdf_path = args.out if ext.lower() == ".pdf" else base + ".pdf"
    png_path = base + ".png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"\nsaved -> {pdf_path} + {png_path}")


if __name__ == "__main__":
    main()
