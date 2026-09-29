"""
heatmap_chordbert.py
--------------------
Same one-row layout as heatmap_theory.py (raw M* | averaged M_sym |
first Bach theory plane), but the fourth panel is ChordBERT layer 3's
empirical E5 plane vs the theoretical circle of fifths — same major/minor
colors and marker/label sizes as the theory panels in heatmap_theory.

Fitting:
  * Bach panel: Procrustes in the first 2D plane only.
  * ChordBERT panel: empirical PC1-PC2 directly (modes 1,2), then in-plane
    Procrustes to the E5 theory ring (no top-5 subspace search).

ChordBERT geometry is loaded from a saved gram cache by default
(chordbert_grams.npz from chordbert_dihedral.py).

Usage:  python heatmap_chordbert.py
            [--sequences chord_sequences_full.json]
            [--grams chordbert_grams.npz] [--layer 3]
            [--out heatmap_chordbert.pdf]
            [--irrep_out fig_chordbert_l3_irreps.pdf] [--k_modes 5]
            [--place] [--positions label_positions_heatmap_chordbert.json]
            [--null_rounds 500] [--null_seed 0]

  --place  Open a 1:1 interactive editor with only the Bach + ChordBERT
           panels (true equal aspect). Drag labels, close the window to
           write the positions JSON. Headless runs reuse that sidecar
           after adjustText. Override with --positions.

  --null_rounds  Chord-permutation null for each subplot (default 500):
           permute the 24 triad labels on the Gram, recompute that
           panel's statistic, print observed vs null mean and 95% CI.

Also writes a companion heatmap of that layer's top empirical modes
vs all nine D12 irreps (cell = isotypic energy ||P_rho v||^2).
"""
import argparse
import json
import os

import numpy as np

import chordlib as cl
import chordbert_dihedral as cbd
import heatmap_theory as fht

DEFAULT_POS = "label_positions_heatmap_chordbert.json"


def chord_cosines(E, T):
    """Per-chord cosines after alignment; return (mean, median, min)."""
    cos = [E[i] @ T[i] /
           max(np.linalg.norm(E[i]) * np.linalg.norm(T[i]), 1e-12)
           for i in range(cl.N)]
    return float(np.mean(cos)), float(np.median(cos)), float(np.min(cos))


def report_bach_per_plane_procrustes(M, k_planes=3):
    """Bach M*: top-2k empirical coords vs top-k theory modes, with an
    independent orthogonal Procrustes in each 2D plane (no global 2k×2k R).

    Prints per-plane chord cosines and the full 2k-dim cosine after stacking
    the plane-wise-aligned theory coordinates.
    """
    k = 2 * k_planes
    lam_e, V_e = cl.embedding_modes(M, k=k)
    Emp = V_e * np.sqrt(np.clip(lam_e, 0, None))
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
    print(f"\nBach top-{k} embeddings: per-plane Procrustes "
          f"(no global {k}D orthogonal):")
    print(f"  empirical eigenvalues: {np.round(lam_e, 2)}")
    print(f"  theory modes: " +
          ", ".join(f"{names[p]}({lams[p]:+.2f})" for p in range(k_planes)))
    for p in range(k_planes):
        sl = slice(2 * p, 2 * p + 2)
        Ep, Tp = Emp[:, sl], Th[:, sl]
        Rp, _ = cl.procrustes(Ep, Tp)
        Tp_a = Tp @ Rp
        Th_aligned[:, sl] = Tp_a
        mean, med, mn = chord_cosines(Ep, Tp_a)
        ov = cl.subspace_overlap(V_e[:, sl], modes[p][2])
        print(f"  plane {p + 1} [{names[p]}]  PC{2 * p + 1}-{2 * p + 2}: "
              f"mean cos {mean:.3f}, median {med:.3f}, min {mn:.3f}, "
              f"overlap {ov:.3f}")

    mean, med, mn = chord_cosines(Emp, Th_aligned)
    print(f"  full {k}D after plane-wise R's: "
          f"mean cos {mean:.3f}, median {med:.3f}, min {mn:.3f}")

    # Reference: one global Procrustes (what fig_theory_overlay reports).
    Rg, _ = cl.procrustes(Emp, Th)
    g_mean, g_med, g_mn = chord_cosines(Emp, Th @ Rg)
    print(f"  full {k}D with one global R (reference): "
          f"mean cos {g_mean:.3f}, median {g_med:.3f}, min {g_mn:.3f}")
    return Emp, Th_aligned


def reynolds_residual(M):
    """||M - group_average(M)||_F / ||M||_F."""
    return float(np.linalg.norm(M - cl.reynolds(M)) /
                 max(np.linalg.norm(M), 1e-12))


def bach_plane1_mean_cos(M):
    """Same 2D Procrustes cosine as the Bach panel (leading 2D theory mode)."""
    lam_e, V_e = cl.embedding_modes(M, k=2)
    Emp = V_e * np.sqrt(np.clip(lam_e, 0, None))
    modes = [m for m in cl.theory_modes(M) if m[2].shape[1] == 2]
    if not modes:
        return float("nan")
    _, lam, V = modes[0]
    Th = np.column_stack([np.sqrt(max(lam, 0)) * V[:, a]
                          for a in range(V.shape[1])])
    R, _ = cl.procrustes(Emp, Th)
    mean, _, _ = chord_cosines(Emp, Th @ R)
    return mean


def chordbert_e5_mean_cos(M, e5_modes):
    """Same E5 Procrustes mean cosine as the ChordBERT panel."""
    sel = [m - 1 for m in e5_modes]
    M_sym = cl.reynolds(M)
    lam_a, V_a = cl.embedding_modes(M, k=max(sel) + 1)
    W_sel = V_a[:, sel] * np.sqrt(np.clip(lam_a[sel], 0, None))
    e5 = next((m for m in cl.theory_modes(M_sym) if m[0] == "E5"), None)
    if e5 is None:
        e5 = cl.theory_modes(M_sym)[0]
    _, lam_th, V_th = e5
    if len(sel) == 2:
        E_emp = W_sel
    else:
        U, _, _ = np.linalg.svd(V_a[:, sel].T @ V_th)
        E_emp = W_sel @ U[:, :2]
    Th5 = np.sqrt(max(lam_th, 0)) * V_th
    R2, _ = cl.procrustes(E_emp, Th5)
    mean, _, _ = chord_cosines(E_emp, Th5 @ R2)
    return mean


def _null_mean_ci(vals, alpha=0.05):
    vals = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def run_chord_permutation_nulls(M, M_cb, e5_modes, n_rounds=500, seed=0):
    """Chord-label permutation nulls for each of the four figure panels.

    Each round draws a random permutation p of the 24 triads and replaces
    a Gram by M'[i,j] = M[p[i], p[j]], then recomputes that panel's
    statistic. Reports observed value vs null mean and 95% interval
    (2.5 / 97.5 percentiles).
    """
    rng = np.random.default_rng(seed)
    labs = cl.d12_labels()

    obs = {
        "M* D12 R^2": cl.r2(M, labs),
        "M_sym residual": reynolds_residual(M),
        "Bach mean cos": bach_plane1_mean_cos(M),
        "ChordBERT mean cos": chordbert_e5_mean_cos(M_cb, e5_modes),
    }
    null = {k: np.empty(n_rounds) for k in obs}

    print(f"\nchord-permutation null ({n_rounds} rounds, seed={seed}):")
    for t in range(n_rounds):
        p = rng.permutation(cl.N)
        Mp = M[np.ix_(p, p)]
        Mcp = M_cb[np.ix_(p, p)]
        null["M* D12 R^2"][t] = cl.r2(Mp, labs)
        null["M_sym residual"][t] = reynolds_residual(Mp)
        null["Bach mean cos"][t] = bach_plane1_mean_cos(Mp)
        null["ChordBERT mean cos"][t] = chordbert_e5_mean_cos(Mcp, e5_modes)
        if (t + 1) % 100 == 0 or t == 0:
            print(f"  … {t + 1}/{n_rounds}")

    # Panel order matches the figure left→right.
    panel_names = [
        ("(a) raw M*", "M* D12 R^2"),
        ("(a) averaged M_sym", "M_sym residual"),
        ("(b) Bach", "Bach mean cos"),
        ("(b) ChordBERT", "ChordBERT mean cos"),
    ]
    print("\nnull summary (observed | null mean | 95% [lo, hi]):")
    for panel, key in panel_names:
        mu, lo, hi = _null_mean_ci(null[key])
        print(f"  {panel:22s}  {key:20s}  "
              f"obs {obs[key]:.3f}  |  null {mu:.3f}  "
              f"95% [{lo:.3f}, {hi:.3f}]")
    return obs, null


def chordbert_e5_panel(ax, E, T, title="ChordBERT"):
    """Theory-panel styling (heatmap_theory colors/sizes) for E5."""
    for i in range(cl.N):
        ax.plot([T[i, 0], E[i, 0]], [T[i, 1], E[i, 1]],
                color="0.35", lw=1.4)
    ax.scatter(T[:12, 0], T[:12, 1], facecolors="none",
               edgecolors="tab:red", s=120, linewidths=1.5)
    ax.scatter(T[12:, 0], T[12:, 1], facecolors="none",
               edgecolors="tab:blue", s=120, linewidths=1.5)
    ax.scatter(E[:12, 0], E[:12, 1], c="tab:red", s=56)
    ax.scatter(E[12:, 0], E[12:, 1], c="tab:blue", s=56)
    texts = [ax.text(E[i, 0], E[i, 1], fht.SHORT_TRIAD_NAMES[i],
                     fontsize=22, ha="left", va="bottom")
             for i in range(cl.N)]
    ax.set_xlabel("PC 1", fontsize=20, labelpad=6)
    ax.set_ylabel("PC 2", fontsize=20, labelpad=2)
    ax.set_title(title, fontsize=20, pad=12, x=0.55)
    ax.tick_params(bottom=False, left=False, labelbottom=False,
                   labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)
    ax.set_anchor("W")
    pts = np.vstack([E, T])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = 0.15 * span
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    return texts, E


def _panel_saved_usable(saved, panel, E):
    """True if this panel has sidecar keys that are not all sitting on dots."""
    n_keys = 0
    n_on = 0
    for i in range(cl.N):
        key = f"{panel}:{i}"
        if key not in saved:
            continue
        n_keys += 1
        dx = saved[key][0] - float(E[i, 0])
        dy = saved[key][1] - float(E[i, 1])
        if dx * dx + dy * dy < 1e-6:
            n_on += 1
    return n_keys > 0 and n_on < n_keys


def _apply_panel_positions(texts, saved, panel):
    """Apply JSON keys '{panel}:{i}' (panel is 0=Bach, 1=ChordBERT)."""
    n = 0
    for i, t in enumerate(texts):
        key = f"{panel}:{i}"
        if key in saved:
            xy = saved[key]
            t.set_position((float(xy[0]), float(xy[1])))
            n += 1
    return n


def _seed_on_dots(texts, E):
    for i, t in enumerate(texts):
        t.set_position((float(E[i, 0]), float(E[i, 1])))


def _seed_place_editor(packs, pos_path, adjust_text):
    """Reuse saved placements per panel; otherwise Bach=adjustText, CB=on-dot."""
    bach, cb = packs[0], packs[1]
    texts_b, E_b = bach
    texts_c, E_c = cb
    saved = json.load(open(pos_path)) if os.path.exists(pos_path) else {}

    if _panel_saved_usable(saved, 0, E_b):
        n = _apply_panel_positions(texts_b, saved, 0)
        print(f"Bach: loaded {n} saved labels from {pos_path}")
    else:
        if any(f"0:{i}" in saved for i in range(cl.N)):
            print("Bach: saved labels were on-dot / incomplete; "
                  "seeding with adjustText")
        adjust_text(
            texts_b, x=E_b[:, 0], y=E_b[:, 1], ax=texts_b[0].axes,
            force_text=(0.5, 0.5), force_static=(0.2, 0.2),
            force_explode=(0.0, 0.0), expand=(1.2, 1.2),
            ensure_inside_axes=True,
        )

    if _panel_saved_usable(saved, 1, E_c):
        n = _apply_panel_positions(texts_c, saved, 1)
        print(f"ChordBERT: loaded {n} saved labels from {pos_path}")
    else:
        _seed_on_dots(texts_c, E_c)
        print("ChordBERT: no usable saved labels; seeded on empirical points")


def run_place_editor(Emp, Th, E_cb, T_cb, pos_path, adjust_text):
    """Interactive 1:1 editor: Bach + ChordBERT panels, equal aspect."""
    import matplotlib
    import matplotlib.pyplot as plt

    if not matplotlib.is_interactive() and \
            matplotlib.get_backend().lower() == "agg":
        raise SystemExit(
            f"backend {matplotlib.get_backend()!r} is non-interactive; "
            f"re-run with a GUI display (TkAgg/QtAgg)")

    place_fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    packs = [
        fht.theory_panel(axes[0], Emp, Th, 0),
        chordbert_e5_panel(axes[1], E_cb, T_cb,
                           title="ChordBERT  (drag labels)"),
    ]
    axes[0].set_title("Bach  (drag labels)", fontsize=14)
    place_fig.suptitle(
        "Label editor (1:1 panels) — close window to save",
        fontsize=13)
    place_fig.tight_layout(rect=[0, 0, 1, 0.95])
    place_fig.canvas.draw()

    _seed_place_editor(packs, pos_path, adjust_text)
    place_fig.canvas.draw_idle()

    drag_handles = [fht.DragText(t) for texts, _ in packs for t in texts]
    fht.size_place_window(place_fig)
    print(f"Drag labels on the true-aspect panels, then close -> {pos_path}")
    print(f"(backend={matplotlib.get_backend()})")
    plt.show(block=True)
    fht.dump_positions(packs, pos_path, fig=place_fig)
    del drag_handles
    plt.close(place_fig)


def load_chordbert_row(grams_path, layer, e5_modes=None):
    data = np.load(grams_path)
    grams = data["grams"]
    if layer < 0 or layer >= len(grams):
        raise SystemExit(f"layer {layer} out of range "
                         f"(have 0..{len(grams) - 1} in {grams_path})")
    row = cbd.analyze_layer(grams[layer], e5_modes=e5_modes)
    print(f"ChordBERT layer {layer} from {grams_path}: "
          f"E5 median cos {row['E5_cos_median']:.3f}, "
          f"mean {row['E5_cos_mean']:.3f}, "
          f"modes {row['e5_modes']}")
    return row, grams[layer]


def layer_mode_irrep_figure(M, layer, out_path, k_modes=5):
    """Heatmap: top-k empirical modes x D12 irreps, cell = isotypic
    energy ||P_rho v||^2 for that mode. A1 (trivial) omitted."""
    import matplotlib.pyplot as plt

    proj = cl.isotypic_projectors()
    irreps = [r for r in cl.IRREPS if r != "A1"][::-1]
    lam, V = np.linalg.eigh(M)
    idx = np.argsort(-lam)[:k_modes]
    Pmat = np.zeros((k_modes, len(irreps)))
    print(f"\nChordBERT L{layer} mode irrep content (top {k_modes}):")
    for j, i in enumerate(idx):
        v = V[:, i]
        v = v / max(np.linalg.norm(v), 1e-12)
        energies = {r: float(np.linalg.norm(P @ v) ** 2)
                    for r, P in proj.items()}
        for c, r in enumerate(irreps):
            Pmat[j, c] = energies[r]
        best = max(energies, key=energies.get)
        bits = "  ".join(f"{r}:{energies[r]:.2f}" for r in irreps)
        print(f"  mode {j + 1}  lam {lam[i]:+.3f}  win {best} "
              f"({energies[best]:.3f})  |  {bits}")

    fig, ax = plt.subplots(figsize=(10.5, 0.7 * k_modes + 2.2))
    im = ax.imshow(Pmat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for j in range(k_modes):
        for c in range(len(irreps)):
            val = Pmat[j, c]
            ax.text(c, j, f"{val:.2f}", ha="center", va="center",
                    fontsize=16,
                    color="w" if val < 0.55 else "k")
    ax.set_xticks(range(len(irreps)))
    ax.set_xticklabels(irreps, fontsize=19)
    ax.set_yticks(range(k_modes))
    ax.set_yticklabels([f"{j + 1}" for j in range(k_modes)], fontsize=19)
    ax.set_ylabel("empirical mode", fontsize=21)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("isotypic energy", fontsize=19)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"saved -> {out_path}")
    if out_path.lower().endswith(".pdf"):
        png = out_path[:-4] + ".png"
        fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print(f"saved -> {png}")
    elif not out_path.lower().endswith(".png"):
        fig.savefig(out_path + ".png", dpi=300, bbox_inches="tight",
                    pad_inches=0.03)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="chord_sequences_full.json")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--grams", default="chordbert_grams.npz")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--e5_modes", default="1,2",
                    help="1-indexed empirical modes for ChordBERT panel "
                         "(default: 1,2 = first plane directly)")
    ap.add_argument("--out", default="heatmap_chordbert.pdf")
    ap.add_argument("--irrep_out", default="fig_chordbert_l3_irreps.pdf",
                    help="layer-wise mode bar chart colored by winning irrep")
    ap.add_argument("--k_modes", type=int, default=5,
                    help="top-k empirical modes for the irrep bar chart")
    ap.add_argument("--place", action="store_true",
                    help="interactive drag-to-place labels; write JSON on close")
    ap.add_argument("--positions", default=None,
                    help="JSON for Bach + ChordBERT labels "
                         f"(default: {DEFAULT_POS})")
    ap.add_argument("--null_rounds", type=int, default=500,
                    help="chord-permutation null rounds per subplot "
                         "(0 to skip)")
    ap.add_argument("--null_seed", type=int, default=0,
                    help="RNG seed for the chord-permutation null")
    args = ap.parse_args()

    import matplotlib
    if args.place:
        # chordbert_dihedral used to force Agg at import; pick a GUI
        # backend explicitly so plt.show() blocks for drag-editing.
        for backend in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
        else:
            raise SystemExit(
                "--place needs a GUI matplotlib backend (TkAgg/QtAgg); "
                "none could be loaded")
        print(f"interactive backend: {matplotlib.get_backend()}")
        if matplotlib.get_backend().lower() == "agg":
            raise SystemExit(
                "still on non-interactive Agg; cannot open label editor")
    else:
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
    M_full, kept = cl.mstar(seqs, window=args.window)
    M = cl.triad_block(M_full, kept)
    M_sym = cl.reynolds(M)
    print(f"sequences: {args.sequences}  "
          f"({len(kept)} token types in statistics)")
    print(f"D12 homogeneity R^2: {cl.r2(M, cl.d12_labels()):.3f}")
    report_bach_per_plane_procrustes(M, k_planes=3)

    # Bach theory panel: first plane only, Procrustes in that 2D plane.
    lam_e, V_e = cl.embedding_modes(M, k=2)
    Emp = V_e * np.sqrt(np.clip(lam_e, 0, None))
    r, lam, V = cl.theory_modes(M)[0]
    Th = np.column_stack([np.sqrt(max(lam, 0)) * V[:, a]
                          for a in range(V.shape[1])])
    R, _ = cl.procrustes(Emp, Th)
    Th = Th @ R
    bach_mean, bach_med, bach_min = chord_cosines(Emp, Th)
    print(f"Bach plane-1 [{r}] 2D Procrustes; theory lam {lam:+.2f}; "
          f"empirical {np.round(lam_e, 2)}")
    print(f"Bach chord cos: mean {bach_mean:.3f}, median {bach_med:.3f}, "
          f"min {bach_min:.3f}")
    pos_path = args.positions or DEFAULT_POS
    print(f"label positions file: {pos_path}")

    # ChordBERT L3: use empirical PC1-PC2 directly (no top-5 subspace search).
    e5_modes = [int(x) for x in args.e5_modes.split(",")]
    if not os.path.exists(args.grams):
        raise SystemExit(
            f"missing {args.grams}; run chordbert_dihedral.py first "
            f"(or pass an existing --grams path)")
    cb, M_cb = load_chordbert_row(args.grams, args.layer, e5_modes=e5_modes)
    cb_mean, cb_med, cb_min = chord_cosines(cb["E5_emp"], cb["E5_th"])
    print(f"ChordBERT L{args.layer} chord cos: mean {cb_mean:.3f}, "
          f"median {cb_med:.3f}, min {cb_min:.3f}")
    print(f"group-average residual ||M-M_sym||/||M||: {reynolds_residual(M):.3f}")

    if args.null_rounds > 0:
        run_chord_permutation_nulls(
            M, M_cb, e5_modes,
            n_rounds=args.null_rounds, seed=args.null_seed)

    if args.place:
        run_place_editor(Emp, Th, cb["E5_emp"], cb["E5_th"],
                         pos_path, adjust_text)

    # Extra figure: layer modes colored by winning D12 irrep.
    irrep_out = args.irrep_out
    if "{L}" in irrep_out:
        irrep_out = irrep_out.replace("{L}", str(args.layer))
    elif args.layer != 3 and "l3" in irrep_out.lower():
        irrep_out = irrep_out.replace("l3", f"l{args.layer}").replace(
            "L3", f"L{args.layer}")
    layer_mode_irrep_figure(M_cb, args.layer, irrep_out, k_modes=args.k_modes)

    fig = plt.figure(figsize=(22, 6.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.03, 1.0],
                          width_ratios=[1.15, 1.15, 0.90, 0.90],
                          hspace=0.01, wspace=0.26,
                          left=0.04, right=0.98, top=0.94, bottom=0.10)
    ax_ta = fig.add_subplot(gs[0, 0:2])
    ax_tb = fig.add_subplot(gs[0, 2:4])
    ax_ta.axis("off")
    ax_tb.axis("off")
    ax_ta.set_title("(a)  Statistical Invariance Holds", fontsize=26,
                    pad=0, x=0.46)
    ax_tb.set_title("(b)  Dihedral Irreps Follow", fontsize=26, pad=0)

    ax0 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[1, 1])
    ax2 = fig.add_subplot(gs[1, 2])
    ax3 = fig.add_subplot(gs[1, 3])

    vmax = np.percentile(np.abs(M), 98)
    im = fht.heatmap_panel(ax0, M, r"$\mathrm{M}^\star$", vmax,
                           show_ylabels=True)
    fht.heatmap_panel(ax1, M_sym, r"$\Pi_{\mathrm{T/I}}\,\mathrm{M}^\star$",
                      vmax, show_ylabels=False)
    cbar = fig.colorbar(im, ax=[ax0, ax1], fraction=0.042, pad=0.055)
    cbar.ax.tick_params(labelsize=16)

    packs = [
        fht.theory_panel(ax2, Emp, Th, 0),
        chordbert_e5_panel(ax3, cb["E5_emp"], cb["E5_th"]),
    ]
    ax2.set_title("Bach", fontsize=20, pad=12)

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
    b2, b3 = ax2.get_position(), ax3.get_position()
    legend_x = 0.5 * (b2.x0 + b3.x1)
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               fontsize=20, frameon=False,
               bbox_to_anchor=(legend_x, b2.y0 - 0.045),
               handletextpad=0.3)

    fig.canvas.draw()
    if os.path.exists(pos_path):
        fht.apply_saved_positions(packs, pos_path)
    else:
        for texts, E in packs:
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
