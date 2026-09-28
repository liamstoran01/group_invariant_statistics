"""
celestial_main_figure.py
--------------------
Split / recombine panels from sky_battery_compact.py and
sky_overlay_harmonics.py, with identical numerics and styling.

Default: four-panel combo --- (a) zonal kernel, (b) degree purity
(heatmap; optional --with-lambda adds empirical-mode vs λ bars),
(c) Mistral Large 2 + Books overlays.

Optional (--sweep): also write battery panel (a) as its own figure.

Usage:
  python celestial_main_figure.py [--model mistrallarge123b] [--lmax 4]
      [--no-dipole-overlap] [--layer 72] [--no-pin] [--nperm 300]
      [--with-lambda] [--sweep] [--out-sweep sky_layer_sweep.pdf]
      [--out-combo sky_kernel_overlay.pdf]
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
from sky_battery import layer_battery
from sky_overlay_harmonics import (
    VIEWS, POOL_DIPOLE, POOL_QUAD,
    harmonic_bases, procrustes, fixed_slice, located, null_stats,
    zonal_r2, weighted_modes_signed,
)


def save_both(fig, path):
    base, ext = os.path.splitext(path)
    pdf_path = path if ext.lower() == ".pdf" else base + ".pdf"
    png_path = base + ".png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"saved -> {pdf_path} + {png_path}")
    return pdf_path, png_path


def compute_battery(model, lmax):
    """Exact sweep + best-layer battery from sky_battery_compact.py."""
    X, lat, lon, xyz, names = sk.load_activations(model)
    Lyr = X.shape[0]
    sweep_layers = list(range(0, Lyr, max(1, Lyr // 30)))
    sweep = []
    for l in sweep_layers:
        r = layer_battery(X[l], lat, lon, xyz, lmax, nperm=20)
        sweep.append((l, r["r2_iso"], r["r2_dec"], r["c3"]))
    best = max(sweep, key=lambda t: t[2])[0]
    r = layer_battery(X[best], lat, lon, xyz, lmax, nperm=100)
    print(f"{model} layer {best}: zonality R^2 = {r['r2_iso']:.3f}  "
          f"(perm null95 {r['null95']:.3f})")
    print(f"  lambda_l: {np.round(r['lam_fh'][1:], 3)}")
    print(f"  {'mode':>5} {'lam':>7}  degree energies (l=1..{lmax})")
    for k, m in enumerate(r["modes"][:4]):
        best_l = max(m, key=m.get)
        print(f"  {k+1:>5} {r['lam'][k]:>7.2f}  " +
              " ".join(f"{m[l]:.2f}" for l in range(1, lmax + 1)) +
              f"   -> l={best_l} ({m[best_l]:.2f})")
    return sweep, best, r


def compute_overlay(model, layer, no_pin, nperm):
    """Exact books + LLM overlays from sky_overlay_harmonics.py."""
    rng = np.random.default_rng(0)

    names, lat, lon, acc, canon = sk.corpus_stream()
    keep, Mb, p, ntok = sk.corpus_mstar(names, acc, canon)
    latb, lonb = lat[keep], lon[keep]
    xyzb = sk.unit_vectors(latb, lonb)
    wb, cosb = sk.density_weights(xyzb)
    swb = np.sqrt(wb)[:, None]
    Bb = harmonic_bases(latb, lonb, wb)
    c_fit = sk.fit_zonal(Mb, cosb, wb)
    if no_pin:
        Mb_f = 0.5 * (Mb + Mb.T)
        pin_note = "UNPINNED (measured diagonal)"
    else:
        Mb_f = sk.pin_diagonal(Mb, c_fit)
        pin_note = "pinned to c(1)"
    Mb_s = 0.5 * (Mb + Mb.T)
    r2 = zonal_r2(Mb_s, cosb, wb)
    null = [zonal_r2(Mb_s, cosb[np.ix_(pi, pi)], wb)
            for pi in (rng.permutation(Mb_s.shape[0])
                       for _ in range(nperm))]
    null = np.array(null)
    print(f"BOOKS corpus zonality: R^2 = {r2:.3f} "
          f"(null mean {null.mean():.3f}, 95% {np.percentile(null, 95):.3f}, "
          f"p = {(np.sum(null >= r2) + 1) / (nperm + 1):.4f})   [model: 0.49]")
    lam_be, U_be = weighted_modes_signed(Mb_f, swb, 12)
    A_b, Ba_b, cm, cmed = fixed_slice(U_be, lam_be, swb, slice(0, 3),
                                      xyzb, scale=False)
    nmu, n95 = null_stats(
        lambda Us: fixed_slice(Us, lam_be, swb, slice(0, 3), xyzb,
                               scale=False)[2],
        U_be, swb, rng, nperm)
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
        U_be, swb, rng, nperm)
    print(f"    quadrupole strict slice (modes 4-8 by decreasing signed "
          f"λ): cos {np.mean(cq):.2f} (median {np.median(cq):.2f}); "
          f"null {nmu_q:.2f} (95% {n95_q:.2f})")

    latL, lonL, xyzL, lamL, UL, _ = sk.llm_modes(model, layer)
    wL, cosL = sk.density_weights(xyzL)
    swL = np.sqrt(wL)[:, None]
    BL = harmonic_bases(latL, lonL, wL)
    A_L, Ba_L, cmL, cmedL, canon2, Qd = located(
        UL, swL, BL[1], 3, POOL_DIPOLE, xyzL)
    nmuL, n95L = null_stats(
        lambda Us: located(Us, swL, BL[1], 3, POOL_DIPOLE, xyzL)[2],
        UL, swL, rng, nperm)
    print(f"\n(b) LLM located dipole (within modes "
          f"{POOL_DIPOLE[0]+1}-{POOL_DIPOLE[-1]+1}) vs dipole harmonics "
          f"at true positions ({len(lonL)} objects, {model} layer "
          f"{layer}):")
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

    nmu_qL, n95_qL = null_stats(llm_quad_mean_cos, UL, swL, rng, nperm)
    print(f"    quadrupole located (modes {POOL_QUAD[0]+1}-"
          f"{POOL_QUAD[-1]+1}, dipole-deflated): canon^2 {c2q:.2f}; "
          f"cos {cmq:.2f} (median {cmedq:.2f}); null {nmu_qL:.2f} "
          f"(95% {n95_qL:.2f}) [null repeats the search]")

    return dict(lonb=lonb, lonL=lonL, A_b=A_b, Ba_b=Ba_b,
                A_L=A_L, Ba_L=Ba_L)


def style_axes(ax):
    """Exact axis framing from sky_overlay_harmonics.py."""
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


def draw_overlay(ax, A, Ba, lo, norm):
    """Exact scatter/links from sky_overlay_harmonics.py."""
    for q in range(A.shape[0]):
        ax.plot([Ba[q, 0], A[q, 0]], [Ba[q, 1], A[q, 1]],
                [Ba[q, 2], A[q, 2]], color="0.45", lw=1.65,
                alpha=0.65, zorder=1)
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], c=lo, cmap="hsv",
               norm=norm, s=42, zorder=3)
    edge_cols = plt.cm.hsv(norm(lo))
    ax.scatter(Ba[:, 0], Ba[:, 1], Ba[:, 2], facecolors="none",
               edgecolors=edge_cols, s=58, linewidths=1.65, zorder=3)


def fill_overlay_frame(ax, A, Ba, zoom=1.35):
    """Tighten a cubic frame around the cloud and zoom so the material
    fills the 3D axes (mplot3d otherwise leaves large empty margins)."""
    pts = np.vstack([A, Ba])
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * float(np.max(hi - lo)) * 1.05  # small pad
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistrallarge123b")
    ap.add_argument("--lmax", type=int, default=4)
    ap.add_argument("--no-dipole-overlap", action="store_true",
                    help="omit the dipole-overlap curve from panel (a)")
    ap.add_argument("--layer", type=int, default=72)
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--nperm", type=int, default=300)
    ap.add_argument("--with-lambda", action="store_true",
                    help="include empirical-mode vs λ bars in panel (b)")
    ap.add_argument("--sweep", action="store_true",
                    help="also write battery panel (a) as its own figure")
    ap.add_argument("--out-sweep", default="sky_layer_sweep.pdf")
    ap.add_argument("--out-combo", default="sky_kernel_overlay.pdf")
    args = ap.parse_args()

    plt.rcParams.update({
        "axes.titlesize": 21,
        "axes.labelsize": 19,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ---------------- battery numerics (compact) ----------------
    sweep, best, r = compute_battery(args.model, args.lmax)
    ls, isos, decs, c3s = zip(*sweep)

    if args.sweep:
        fig1 = plt.figure(figsize=(6.5, 4.1))
        ax = fig1.add_subplot(1, 1, 1)
        ax.plot(ls, decs, "o-", ms=5, lw=1.5, label="decode $R^2$")
        ax.plot(ls, isos, "s-", ms=5, lw=1.5, label="Gram zonality $R^2$")
        if not args.no_dipole_overlap:
            ax.plot(ls, c3s, "^-", ms=5, lw=1.5, label="dipole overlap")
        ax.axvline(best, color="gray", lw=0.8)
        ib = ls.index(best)
        y_lab = (0.5 * (c3s[ib] + isos[ib]) if not args.no_dipole_overlap
                 else 0.5 * (decs[ib] + isos[ib]))
        ax.text(best, y_lab, f" layer {best}",
                color="gray", fontsize=17, va="center", ha="left")
        ax.set_xlabel("layer")
        ax.legend(fontsize=16, loc="upper left", frameon=False)
        ax.set_title("(a) Mistral Large 2: sky structure across layers")
        fig1.tight_layout()
        save_both(fig1, args.out_sweep)
        plt.close(fig1)

    # ---------------- overlay numerics (harmonics) ----------------
    ov = compute_overlay(args.model, args.layer, args.no_pin, args.nperm)
    lon_all = np.concatenate([ov["lonb"], ov["lonL"]])
    norm = Normalize(vmin=float(lon_all.min()), vmax=float(lon_all.max()))

    # Figure 2: (a) kernel | gap | (b) purity [| λ] | (c) LLM | Books | RA
    fig2 = plt.figure(figsize=(20, 5.5))
    w_b = 1.5 if args.with_lambda else 1.05
    gs = fig2.add_gridspec(
        1, 4, width_ratios=[0.9, 0.12, w_b, 2.85], wspace=0.08)
    # Overlay pair tight; RA colorbar kept slim like the purity colorbar.
    gs_right = gs[0, 3].subgridspec(
        1, 2, width_ratios=[1.0, 0.035], wspace=0.05)
    gs_ov = gs_right[0, 0].subgridspec(1, 2, wspace=-0.08)

    # (a) zonal kernel --- fitted C(θ) with ±1 SD band (NW local std)
    ax_k = fig2.add_subplot(gs[0, 0])
    thd = np.degrees(np.arccos(sk.XQ))
    oq = np.argsort(thd)
    th, cf, cs = thd[oq], r["c_fit"][oq], r["c_std"][oq]
    ax_k.fill_between(th, cf - cs, cf + cs, color="tab:orange",
                      alpha=0.25, lw=0, zorder=1, label="±1 std. dev.")
    ax_k.plot(th, cf, "-", lw=2.2, color="darkorange", zorder=3,
              label=r"fitted kernel C(θ)")
    ax_k.set_xlabel(r"true θ (deg)")
    ax_k.set_ylabel(r"C(θ)", labelpad=-2)
    title_fs = 23
    ax_k.set_title("(a) Structural Invariance",
                   fontsize=title_fs)
    ax_k.title.set_x(0.46)
    ax_k.set_xlim(0, 180)
    h_k, l_k = ax_k.get_legend_handles_labels()
    # fitted kernel above ±1 std. dev.
    order = [1, 0]
    ax_k.legend([h_k[i] for i in order], [l_k[i] for i in order],
                loc="lower left", fontsize=17, frameon=True,
                framealpha=0.45, handlelength=1.4)

    # (b) degree-purity heatmap (+ optional λ bars)
    nshow = len(r["modes"])
    colors = plt.cm.tab10(np.arange(args.lmax + 1))
    Pm = np.array([[m[l] for l in range(1, args.lmax + 1)]
                   for m in r["modes"]])

    axl = None
    if args.with_lambda:
        gs_c = gs[0, 2].subgridspec(1, 2, width_ratios=[0.30, 1.05],
                                    wspace=0.08)
        axl = fig2.add_subplot(gs_c[0, 0])
        lam = r["lam"][:nshow]
        win = [max(m, key=m.get) for m in r["modes"]]
        axl.barh(np.arange(nshow), lam,
                 color=[colors[w] for w in win],
                 alpha=0.9, height=0.75)
        axl.invert_yaxis()
        axl.set_ylim(nshow - 0.5, -0.5)
        axl.set_xlabel(r"$\lambda$", fontsize=23, labelpad=4)
        axl.set_yticks(range(0, nshow, 2))
        axl.set_yticklabels([str(k + 1) for k in range(0, nshow, 2)],
                            fontsize=16)
        axl.set_ylabel("empirical mode")
        axh = fig2.add_subplot(gs_c[0, 1], sharey=axl)
    else:
        axh = fig2.add_subplot(gs[0, 2])

    im = axh.imshow(Pm, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                    interpolation="nearest",
                    extent=[0.5, args.lmax + 0.5, nshow - 0.5, -0.5])
    axh.set_xticks(range(1, args.lmax + 1))
    axh.set_xticklabels([f"$\\ell$={l}" for l in range(1, args.lmax + 1)],
                        fontsize=21)
    if args.with_lambda:
        plt.setp(axh.get_yticklabels(), visible=False)
    else:
        axh.set_ylabel("empirical mode")
        axh.set_yticks(range(0, nshow, 2))
        axh.set_yticklabels([str(k + 1) for k in range(0, nshow, 2)],
                            fontsize=16)
    cbar_c = plt.colorbar(im, ax=axh, fraction=0.05, pad=0.02)
    cbar_c.set_label("degree purity", fontsize=19)
    cbar_c.ax.tick_params(labelsize=18)

    # LLM | Books overlays (tight pair) + RA colorbar on the far right
    elev, azim = VIEWS[0]
    ax_llm = fig2.add_subplot(gs_ov[0, 0], projection="3d")
    ax_books = fig2.add_subplot(gs_ov[0, 1], projection="3d")
    for ax in (ax_books, ax_llm):
        ax.computed_zorder = False

    rng_plot = np.random.default_rng(0)
    n_show = min(75, ov["A_L"].shape[0])
    idx = rng_plot.choice(ov["A_L"].shape[0], size=n_show, replace=False)
    A_show, Ba_show = ov["A_L"][idx], ov["Ba_L"][idx]
    draw_overlay(ax_llm, A_show, Ba_show, ov["lonL"][idx], norm)
    ax_llm.view_init(elev=elev, azim=azim)
    fill_overlay_frame(ax_llm, A_show, Ba_show)
    style_axes(ax_llm)
    ax_llm.set_title("Mistral Large 2", fontsize=21, pad=2)
    ax_llm.title.set_x(0.46)

    draw_overlay(ax_books, ov["A_b"], ov["Ba_b"], ov["lonb"], norm)
    ax_books.view_init(elev=elev, azim=azim)
    fill_overlay_frame(ax_books, ov["A_b"], ov["Ba_b"])
    style_axes(ax_books)
    ax_books.set_title("Books", fontsize=21, pad=2)
    ax_books.title.set_x(0.46)

    sm = ScalarMappable(norm=norm, cmap="hsv")
    sm.set_array([])
    cax = fig2.add_subplot(gs_right[0, 1])
    cbar = fig2.colorbar(sm, cax=cax)
    cbar.set_label("true RA (deg)", fontsize=19)
    cbar.ax.tick_params(labelsize=18)
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="gray", markeredgecolor="gray",
               markersize=9, label="empirical"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="none", markeredgecolor="gray",
               markersize=9, markeredgewidth=1.4,
               label="theory"),
    ]
    fig2.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.16)
    fig2.canvas.draw()
    # Align (b)/(c) titles to the vertical center of (a)'s title.
    renderer = fig2.canvas.get_renderer()
    bb_ta = (ax_k.title.get_window_extent(renderer)
             .transformed(fig2.transFigure.inverted()))
    title_y = 0.5 * (bb_ta.y0 + bb_ta.y1)

    # Lower overlay panels so (c) can sit at the same level as (a)/(b),
    # then nudge the 3D spheres + axes further down (legend stays put).
    bb_l = ax_llm.get_position()
    bb_b = ax_books.get_position()
    needed_top = title_y - 0.055
    delta = max(0.0, max(bb_l.y1, bb_b.y1) - needed_top)
    shift_down = 0.05  # extra drop of spheres/axes; legend stays at y=0.12
    for ax in (ax_llm, ax_books):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - shift_down,
                         pos.width, pos.height - delta])
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 - shift_down,
                      pos.width, pos.height - delta])
    # Pull LLM | Books closer; frees space on the left for panel (b).
    ov_close = 0.02
    bb_l = ax_llm.get_position()
    bb_b = ax_books.get_position()
    ax_llm.set_position([bb_l.x0 + ov_close, bb_l.y0,
                         bb_l.width, bb_l.height])
    ax_books.set_position([bb_b.x0 - ov_close, bb_b.y0,
                           bb_b.width, bb_b.height])
    fig2.canvas.draw()

    # Lower panel (b) a bit and scoot it right (title follows).
    b_shift = 0.04
    b_right = 0.018  # was 0.03; smidge left from prior placement
    axes_b = [axh, cbar_c.ax]
    if axl is not None:
        axes_b.append(axl)
    for ax in axes_b:
        pos = ax.get_position()
        ax.set_position([pos.x0 + b_right, pos.y0 - b_shift,
                         pos.width, pos.height])

    bb_heat = axh.get_position()
    if axl is not None:
        bb_spec = axl.get_position()
        x_mid_b = 0.5 * (bb_spec.x0 + bb_heat.x1)
    else:
        x_mid_b = 0.5 * (bb_heat.x0 + bb_heat.x1)
    fig2.text(x_mid_b, title_y,
              "(b) Spherical Harmonics Follow",
              ha="center", va="center", fontsize=title_fs,
              transform=fig2.transFigure)
    bb_l = ax_llm.get_position()
    bb_b = ax_books.get_position()
    x_mid = 0.5 * (bb_l.x0 + bb_b.x1)
    fig2.text(x_mid, title_y,
              "(c) Spherical Structure Across Sources",
              ha="center", va="center", fontsize=title_fs,
              transform=fig2.transFigure)
    fig2.legend(handles=handles, loc="upper center", ncol=2, fontsize=20,
                frameon=False, bbox_to_anchor=(x_mid, 0.145))
    # Match true-RA colorbar length to the spectrum purity colorbar;
    # nudge slightly left toward the Books panel.
    pos = cax.get_position()
    h_ref = cbar_c.ax.get_position().height
    y0 = pos.y0 + 0.5 * (pos.height - h_ref)
    cax.set_position([pos.x0 - 0.028, y0, pos.width * 0.75, h_ref])
    save_both(fig2, args.out_combo)
    plt.close(fig2)


if __name__ == "__main__":
    main()