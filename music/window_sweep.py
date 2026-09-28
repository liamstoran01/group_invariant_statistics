"""
window_sweep.py
---------------
Robustness of the dihedral results to the co-occurrence window size.

The symmetry argument predicts a sharp split: window choice deforms the
kernel C(g) -- hence eigenVALUES, variance shares, and which plane ranks
where -- but cannot move the eigenVECTORS off the irrep structure, because
windowing is itself a G-equivariant operation on G-invariant statistics.
So symmetry-protected quantities (irrep identities/purities of the top
planes, rotation angles under T_1, reflection parity, the major/minor
phase offset) should be flat in the window, while amplitude-flavored
quantities (eigenvalues, variance shares) may drift.

For each window w we report, on the triad block of the full-vocabulary M*:
  * D12 homogeneity R^2 and block-diagonal (isotypic) energy fraction
  * per plane (top three): dominant irrep + purity, angle under T_1,
    det of the fitted inversion, equivariance residual
  * the relative-minor phase offset on plane 1 (corpus fingerprint)
  * top-6 eigenvalues (the quantities ALLOWED to move)

Usage:  python3 window_sweep.py
            [--sequences chord_sequences_full.json]
            [--windows 1 2 3 5 8 16] [--out window_sweep.pdf]
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import chordlib as cl


def block_energy_fraction(M24, proj):
    Mc = cl.center(M24)
    Mc = 0.5 * (Mc + Mc.T)
    tot = np.linalg.norm(Mc, "fro") ** 2
    diag = sum(np.linalg.norm(P @ Mc @ P, "fro") ** 2 for P in proj.values())
    return diag / tot


def minor_offset_deg(E):
    """Magnitude of the angular offset of relative minors vs their majors
    on a plane, in degrees (circular mean; sign is orientation gauge)."""
    a = np.arctan2(E[:, 1], E[:, 0])
    off = np.angle(np.mean(np.exp(
        1j * (a[[12 + (r + 9) % 12 for r in range(12)]] - a[:12]))))
    return abs(np.degrees(off))   # sign is plane-orientation gauge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="chord_sequences_full.json")
    ap.add_argument("--windows", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 16])
    ap.add_argument("--out", default="window_sweep.pdf",
                    help="primary output; a .png twin is also written")
    args = ap.parse_args()

    plt.rcParams.update({
        "pdf.fonttype": 42,          # TrueType text (sharp in PDF viewers)
        "ps.fonttype": 42,
        "font.size": 18,
        "axes.titlesize": 19,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    })

    seqs = json.load(open(args.sequences))
    proj = cl.isotypic_projectors()
    labD = cl.d12_labels()

    rows = []
    for w in args.windows:
        M = cl.triad_block(*cl.mstar(seqs, window=w))
        lam, V = cl.embedding_modes(M, k=6)
        E = V * np.sqrt(np.clip(lam, 0, None))
        row = dict(w=w, r2=cl.r2(M, labD),
                   frac=block_energy_fraction(M, proj), lam=lam,
                   offset=minor_offset_deg(E[:, :2]), planes=[])
        for p in range(3):
            pur = cl.plane_purity(V[:, 2 * p:2 * p + 2], proj)
            best = max(pur, key=pur.get)
            nums = cl.plane_numbers(E[:, 2 * p:2 * p + 2])
            row["planes"].append(dict(irrep=best, purity=pur[best],
                                      angle=nums["angle"],
                                      det_i=nums["det_i"],
                                      res=nums["res_t"]))
        rows.append(row)

    # ------------------------------ report ------------------------------
    print(f"{'w':>3} {'R^2':>6} {'blockE':>7} | "
          f"{'plane1 irrep/pur/angle/det':>28} | "
          f"{'plane2':>24} | {'plane3':>24} | {'offset':>7}")
    for r in rows:
        cells = []
        for p in r["planes"]:
            cells.append(f"{p['irrep']}/{p['purity']:.2f}/"
                         f"{p['angle']:5.1f}/{p['det_i']:+.0f}")
        print(f"{r['w']:>3} {r['r2']:>6.3f} {r['frac']:>7.3f} | "
              f"{cells[0]:>28} | {cells[1]:>24} | {cells[2]:>24} | "
              f"{r['offset']:>6.1f}")
    print("\ntop-6 eigenvalues per window (allowed to move):")
    for r in rows:
        print(f"  w={r['w']:>2}: {np.round(r['lam'], 2)}")

    # ------------------------------ figure ------------------------------
    ws = [r["w"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax = axes[0]
    for p, color in enumerate(["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(ws, [r["planes"][p]["angle"] for r in rows], "o-",
                color=color, label=f"plane {p + 1}", ms=6, lw=1.5)
    for h in range(1, 6):
        ax.axhline(h * 30, color="k", ls=":", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks(ws)
    ax.set_xticklabels(ws)
    ax.set_xlabel("window size")
    ax.set_ylim(0, 185)
    ax.set_ylabel("rotation angle (deg)")
    ax.legend(frameon=False, ncol=3, loc="lower center",
              columnspacing=0.45, handletextpad=0.28, handlelength=1.5)
    ax.set_title("(a)")

    ax = axes[1]
    irrep_tags = [r"$E_5$", r"$E_2$", r"$E_3$"]
    for p, color in enumerate(["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(ws, [r["planes"][p]["purity"] for r in rows], "o-",
                color=color, label=f"plane {p + 1} ({irrep_tags[p]})",
                ms=6, lw=1.5)
    ax.axhline(4 / 24, color="r", ls=":", lw=0.8)
    ax.text(ws[-1] * 1.08, 4 / 24 + 0.02, r"chance $= 1/6$",
            color="r", fontsize=15, ha="right")
    ax.set_xscale("log")
    ax.set_xticks(ws)
    ax.set_xticklabels(ws)
    ax.set_xlabel("window size")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("isotypic energy")
    ax.legend(frameon=False, loc="lower left",
              bbox_to_anchor=(-0.03, 0.12))
    ax.set_title("(b)")

    ax = axes[2]
    lam_mat = np.array([r["lam"] for r in rows])
    for k in range(6):
        ax.plot(ws, lam_mat[:, k], "o-",
                color=plt.cm.viridis(k / 6), label=f"mode {k + 1}",
                ms=6, lw=1.5)
    ax.set_xscale("log")
    ax.set_xticks(ws)
    ax.set_xticklabels(ws)
    ax.set_xlabel("window size")
    ax.set_ylabel("eigenvalues")
    ax.legend(frameon=False, ncol=2)
    ax.set_title("(c)")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print(f"\nsaved -> {args.out}")
    png_out = (args.out[:-4] + ".png" if args.out.lower().endswith(".pdf")
               else args.out + ".png")
    if png_out != args.out:
        fig.savefig(png_out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"saved -> {png_out}")


if __name__ == "__main__":
    main()
