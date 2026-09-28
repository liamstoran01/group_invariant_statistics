"""
heatmap_theory.py
-----------------
Combined one-row figure: the first two chord-heatmap panels (raw M* and
D12-symmetrized M_sym) on the left, and the first two theory-overlay planes
on the right. Computations match make_heatmap.py and fig_theory_overlay.py
exactly. Console still reports all three theory planes.

Usage:  python heatmap_theory.py
            [--sequences chord_sequences_full.json]
            [--out heatmap_theory.pdf] [--window 3]
            [--place] [--positions label_positions_4d.json]

  --place  Open a 1:1 interactive editor with only the two theory panels
           (true equal aspect — not the full paper figure). Drag labels,
           close the window to write label_positions_{d}d.json. Headless
           runs reuse that sidecar after adjustText. Override with
           --positions.
"""
import argparse
import json
import os

import numpy as np

import chordlib as cl

# Standard shorthand: uppercase = major, lowercase = minor (no M/m suffix).
SHORT_TRIAD_NAMES = cl.NOTE + [n.lower() for n in cl.NOTE]


def positions_path(dim):
    """Sidecar JSON for a given Procrustes ambient dimension."""
    return f"label_positions_{int(dim)}d.json"


def heatmap_panel(ax, M, title, vmax, show_ylabels=True):
    """Heatmap with major/minor block labels instead of per-chord ticks."""
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    # Label centers have no tick marks; a single thick tick marks the split.
    ax.set_xticks([5.5, 17.5])
    ax.set_xticklabels(["majors", "minors"], fontsize=20)
    ax.set_xticks([11.5], minor=True)
    if show_ylabels:
        ax.set_yticks([5.5, 17.5])
        ax.set_yticklabels(["majors", "minors"], fontsize=20, rotation=90,
                           va="center")
    else:
        ax.set_yticks([5.5, 17.5])
        ax.set_yticklabels([])
    ax.set_yticks([11.5], minor=True)
    ax.tick_params(which="major", length=0, width=0)
    ax.tick_params(which="minor", length=8, width=2.4)
    ax.axhline(11.5, color="k", lw=1.8)
    ax.axvline(11.5, color="k", lw=1.8)
    ax.set_title(title, fontsize=29)
    return im


def theory_panel(ax, Emp, Th, p):
    """Same styling as one fig_theory_overlay plane."""
    E, T = Emp[:, 2 * p:2 * p + 2], Th[:, 2 * p:2 * p + 2]
    for i in range(cl.N):
        ax.plot([T[i, 0], E[i, 0]], [T[i, 1], E[i, 1]],
                color="0.35", lw=1.4)
    ax.scatter(T[:12, 0], T[:12, 1], facecolors="none",
               edgecolors="tab:red", s=120, linewidths=1.5)
    ax.scatter(T[12:, 0], T[12:, 1], facecolors="none",
               edgecolors="tab:blue", s=120, linewidths=1.5)
    ax.scatter(E[:12, 0], E[:12, 1], c="tab:red", s=56)
    ax.scatter(E[12:, 0], E[12:, 1], c="tab:blue", s=56)
    # Plain Text (not Annotation): draggable get/set_position round-trips
    # in data coords. Annotation.draggable was writing the point anchor.
    texts = [ax.text(E[i, 0], E[i, 1], SHORT_TRIAD_NAMES[i], fontsize=22,
                     ha="left", va="bottom")
             for i in range(cl.N)]
    ax.set_xlabel(f"PC {2 * p + 1}", fontsize=20, labelpad=6)
    ax.set_ylabel(f"PC {2 * p + 2}", fontsize=20, labelpad=2)
    ax.tick_params(bottom=False, left=False, labelbottom=False,
                   labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Square axes box (not a centered letterbox inside a wide cell) so
    # PC2/PC4 sit next to the spine like PC1/PC3 sit under theirs.
    ax.set_box_aspect(1)
    ax.set_anchor("W")
    pts = np.vstack([E, T])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = 0.15 * span
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    return texts, E


class DragText:
    """Minimal data-coord drag for matplotlib Text (no Text.draggable)."""

    def __init__(self, text):
        self.text = text
        self.press = None
        canvas = text.figure.canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("button_release_event", self._on_release),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
        ]

    def _on_press(self, event):
        if event.inaxes is not self.text.axes or event.button != 1:
            return
        contains, _ = self.text.contains(event)
        if not contains or event.xdata is None or event.ydata is None:
            return
        x0, y0 = self.text.get_position()
        self.press = (x0, y0, event.xdata, event.ydata)

    def _on_motion(self, event):
        if self.press is None or event.inaxes is not self.text.axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        x0, y0, xp, yp = self.press
        self.text.set_position((x0 + event.xdata - xp,
                                y0 + event.ydata - yp))
        self.text.figure.canvas.draw_idle()

    def _on_release(self, event):
        self.press = None


def apply_saved_positions(theory_packs, path):
    """Overwrite label positions from a JSON sidecar (data coords)."""
    saved = json.load(open(path))
    n_applied = 0
    for p, (texts, _) in enumerate(theory_packs):
        for i, t in enumerate(texts):
            key = f"{p}:{i}"
            if key in saved:
                xy = saved[key]
                t.set_position((float(xy[0]), float(xy[1])))
                n_applied += 1
    print(f"applied {n_applied} saved label positions from {path}")


def dump_positions(theory_packs, path, fig=None):
    """Write label data-coords; draw first so drag state is flushed."""
    if fig is not None:
        fig.canvas.draw()
    payload = {}
    for p, (texts, E) in enumerate(theory_packs):
        for i, t in enumerate(texts):
            x, y = t.get_position()
            payload[f"{p}:{i}"] = [float(x), float(y)]
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"saved -> {path}")


def size_place_window(fig, max_width_px=1400):
    """Size the GUI window to the figure's aspect ratio."""
    w_in, h_in = fig.get_size_inches()
    aspect = h_in / max(w_in, 1e-9)
    width = int(max_width_px)
    height = max(int(width * aspect), 200)
    fig.set_size_inches(w_in, h_in, forward=True)
    mng = getattr(fig.canvas, "manager", None)
    if mng is None:
        return
    for call in (
        lambda: mng.resize(width, height),
        lambda: mng.window.resize(width, height),
        lambda: mng.window.wm_geometry(f"{width}x{height}+40+40"),
    ):
        try:
            call()
            print(f"place window {width}x{height}px")
            return
        except Exception:
            continue


def run_place_editor(Emp, Th, pos_path, adjust_text):
    """Interactive 1:1 editor: only the two theory panels, equal aspect.

    Editing the full paper figure over X11 anamorphically stretches axes, so
    visual placements disagree with the saved PNG. This UI matches the
    theory-panel geometry used in the paper figure.
    """
    import matplotlib.pyplot as plt

    place_fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    packs = [
        theory_panel(axes[0], Emp, Th, 0),
        theory_panel(axes[1], Emp, Th, 1),
    ]
    axes[0].set_title("PC 1–2  (drag labels)", fontsize=14)
    axes[1].set_title("PC 3–4  (drag labels)", fontsize=14)
    place_fig.suptitle(
        "Label editor (1:1 theory panels) — close window to save",
        fontsize=13)
    place_fig.tight_layout(rect=[0, 0, 1, 0.95])
    place_fig.canvas.draw()

    # Prefer a previous good sidecar; ignore one that looks like "on points"
    # (corrupted Annotation dumps). Fall back to adjustText.
    use_saved = False
    if os.path.exists(pos_path):
        saved = json.load(open(pos_path))
        # Heuristic: if most labels sit on their dots, treat as corrupt.
        n_on = 0
        for p, (_, E) in enumerate(packs):
            for i in range(cl.N):
                key = f"{p}:{i}"
                if key not in saved:
                    continue
                dx = saved[key][0] - float(E[i, 0])
                dy = saved[key][1] - float(E[i, 1])
                if dx * dx + dy * dy < 1e-6:
                    n_on += 1
        if n_on < cl.N:  # at least some offsets present
            apply_saved_positions(packs, pos_path)
            use_saved = True
            if n_on:
                print(f"note: {n_on} labels in {pos_path} were on-dot "
                      f"(likely from a bad prior dump)")
        else:
            print(f"ignoring {pos_path}: labels sit on points; "
                  f"re-seeding with adjustText")

    if not use_saved:
        for texts, E in packs:
            adjust_text(
                texts, x=E[:, 0], y=E[:, 1], ax=texts[0].axes,
                force_text=(0.5, 0.5), force_static=(0.2, 0.2),
                force_explode=(0.0, 0.0), expand=(1.2, 1.2),
                ensure_inside_axes=True,
            )

    # Keep refs so drag handlers are not garbage-collected during plt.show().
    drag_handles = [DragText(t) for texts, _ in packs for t in texts]
    size_place_window(place_fig)
    print(f"Drag labels on the true-aspect panels, then close -> {pos_path}")
    plt.show()
    dump_positions(packs, pos_path, fig=place_fig)
    del drag_handles
    plt.close(place_fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="chord_sequences_full.json")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--out", default="heatmap_theory.pdf",
                    help="primary output; a .png twin is also written")
    ap.add_argument("--place", action="store_true",
                    help="interactive drag-to-place labels; write JSON on close")
    ap.add_argument("--positions", default=None,
                    help="JSON sidecar for labels (default: "
                         "label_positions_{d}d.json for Procrustes dim d)")
    args = ap.parse_args()

    import matplotlib
    if not args.place:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 14,
    })

    seqs = json.load(open(args.sequences))

    # ---------- heatmap computation (same as make_heatmap.py) ----------
    M_full, kept = cl.mstar(seqs, window=args.window)
    M = cl.triad_block(M_full, kept)
    M_sym = cl.reynolds(M)
    resid = M - M_sym
    rel = np.linalg.norm(resid) / np.linalg.norm(M)
    print(f"sequences: {args.sequences}  "
          f"({len(kept)} token types in statistics)")
    print(f"D12 homogeneity R^2: {cl.r2(M, cl.d12_labels()):.3f}")
    print(f"symmetry-breaking residual ||M - M_sym|| / ||M||: {rel:.3f}")

    # ---------- theory overlay: first two planes, one 4D Procrustes -----
    # Align only in the plotted 4-dim subspace (not full 6D), so each
    # theory plane stays closer to equal-radius majors/minors after R.
    lam_e, V_e = cl.embedding_modes(M, k=4)
    Emp = V_e * np.sqrt(np.clip(lam_e, 0, None))
    modes_all = cl.theory_modes(M)
    modes = modes_all[:2]
    cols, names, lam_t = [], [], []
    for r, lam, V in modes:
        for a in range(V.shape[1]):
            cols.append(np.sqrt(max(lam, 0)) * V[:, a])
        names.append(r)
        lam_t.append(lam)
    Th = np.column_stack(cols)
    R, rel_th = cl.procrustes(Emp, Th)
    Th = Th @ R
    align_dim = Emp.shape[1]
    pos_path = args.positions or positions_path(align_dim)
    print(f"label positions file: {pos_path}")
    cos_4 = [Emp[i] @ Th[i] / max(np.linalg.norm(Emp[i]) *
                                   np.linalg.norm(Th[i]), 1e-12)
             for i in range(cl.N)]
    print(f"theory eigenvalues (x2 each, top-2 planes): {np.round(lam_t, 2)}   "
          f"empirical: {np.round(lam_e, 2)}")
    if len(modes_all) >= 3:
        print(f"(unused 3rd theory mode for this figure: {modes_all[2][0]} "
              f"lam {modes_all[2][1]:+.2f})")
    print(f"global 4-dim per-chord cosine: mean {np.mean(cos_4):.3f}, "
          f"min {np.min(cos_4):.3f}   (relative residual {rel_th:.3f})")
    for p in range(2):
        E, T = Emp[:, 2 * p:2 * p + 2], Th[:, 2 * p:2 * p + 2]
        ov = cl.subspace_overlap(V_e[:, 2 * p:2 * p + 2], modes[p][2])
        cos = [E[i] @ T[i] / max(np.linalg.norm(E[i]) *
                                 np.linalg.norm(T[i]), 1e-12)
               for i in range(cl.N)]
        r = names[p]
        print(f"plane {p + 1} [{r}]: theory lam {lam_t[p]:+.2f} vs empirical "
              f"({lam_e[2 * p]:+.2f}, {lam_e[2 * p + 1]:+.2f});  "
              f"cos {np.mean(cos):.3f}, overlap {ov:.3f}")
        print(f"  theory radii maj/min "
              f"{np.linalg.norm(T[:12], axis=1).mean():.3f}/"
              f"{np.linalg.norm(T[12:], axis=1).mean():.3f}")

    # ---------- figure: heatmaps | theory planes ----------------------
    # In --place mode, edit labels on a separate 1:1 theory-only window
    # first (true equal aspect), then build the paper figure for export.
    if args.place:
        run_place_editor(Emp, Th, pos_path, adjust_text)

    # Heatmaps a bit wider than theory planes; extra gap after the colorbar
    # so the first PC ylabel does not collide with the heatbar.
    fig = plt.figure(figsize=(22, 6.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.03, 1.0],
                          width_ratios=[1.15, 1.15, 0.90, 0.90],
                          hspace=0.01, wspace=0.26,
                          left=0.04, right=0.98, top=0.94, bottom=0.10)
    ax_ta = fig.add_subplot(gs[0, 0:2])
    ax_tb = fig.add_subplot(gs[0, 2:4])
    ax_ta.axis("off")
    ax_tb.axis("off")
    # Center over the two heatmaps, nudged a hair left of dead center.
    ax_ta.set_title("(a)  Statistical Invariance Holds", fontsize=26,
                    pad=0, x=0.46)
    ax_tb.set_title("(b)  Dihedral Irreps Follow", fontsize=26, pad=0)

    ax0 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[1, 1])
    ax2 = fig.add_subplot(gs[1, 2])
    ax3 = fig.add_subplot(gs[1, 3])

    vmax = np.percentile(np.abs(M), 98)
    im = heatmap_panel(ax0, M, r"$\mathrm{M}^\star$", vmax, show_ylabels=True)
    heatmap_panel(ax1, M_sym, r"$\Pi_{\mathrm{T/I}}\,\mathrm{M}^\star$",
                  vmax, show_ylabels=False)
    cbar = fig.colorbar(im, ax=[ax0, ax1], fraction=0.042, pad=0.055)
    cbar.ax.tick_params(labelsize=16)

    theory_packs = [
        theory_panel(ax2, Emp, Th, 0),
        theory_panel(ax3, Emp, Th, 1),
    ]

    legend_handles = [
        Line2D([0], [0], color="tab:red", lw=2.5, label="major"),
        Line2D([0], [0], color="tab:blue", lw=2.5, label="minor"),
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=9, markerfacecolor="gray", label="empirical"),
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=10, markerfacecolor="none",
               markeredgecolor="gray", markeredgewidth=1.5, label="theory"),
    ]
    # Center the legend under the two theory panels.
    fig.canvas.draw()
    b2, b3 = ax2.get_position(), ax3.get_position()
    legend_x = 0.5 * (b2.x0 + b3.x1)
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               fontsize=20, frameon=False,
               bbox_to_anchor=(legend_x, b2.y0 - 0.045),
               handletextpad=0.3)

    # If we have saved placements, skip adjustText so it cannot fight them.
    fig.canvas.draw()
    if os.path.exists(pos_path):
        apply_saved_positions(theory_packs, pos_path)
    else:
        for texts, E in theory_packs:
            adjust_text(
                texts, x=E[:, 0], y=E[:, 1], ax=texts[0].axes,
                force_text=(0.5, 0.5), force_static=(0.2, 0.2),
                force_explode=(0.0, 0.0), expand=(1.2, 1.2),
                ensure_inside_axes=True,
            )

    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.04)
    print(f"saved -> {args.out}")
    png_out = (args.out[:-4] + ".png" if args.out.lower().endswith(".pdf")
               else args.out + ".png")
    if png_out != args.out:
        fig.savefig(png_out, dpi=300, bbox_inches="tight", pad_inches=0.04)
        print(f"saved -> {png_out}")


if __name__ == "__main__":
    main()
