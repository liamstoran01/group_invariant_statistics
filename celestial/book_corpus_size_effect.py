"""
book_corpus_size_effect.py
--------------------------
Self-fit size-effect curve. Observed matrices keep measured M* diagonals
by default (--no-pin); pass --pin to pin them to the fitted kernel.

Poisson simulations generate counts from the fitted zonal kernel at
multipliers of the books' event level; each sim refits its own kernel
and scores against that self-fit theory (always pinned, since sims have
no measured diagonal). Observed matrices honor --no-pin / --pin. Modes
ordered by decreasing signed λ.

Outputs: printed numbers + book_corpus_size_effect.pdf/.png.
Usage: python3 book_corpus_size_effect.py [--pin] [--n-sims 1000]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import skylib as sk

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def weighted_modes_signed(M, sw, k=None):
    """Like sk.weighted_modes, but sorted by decreasing signed λ
    (not |λ|). Matches sky_overlay_harmonics for indefinite books M*."""
    lam, U = sk.weighted_modes(M, sw, k=None)
    o = np.argsort(-lam)
    if k is not None:
        o = o[:k]
    return lam[o], U[:, o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pin", action="store_true", default=True,
                    help="keep measured M* diagonals (default)")
    ap.add_argument("--pin", dest="no_pin", action="store_false",
                    help="pin observed M* diagonals to the fitted "
                         "kernel's c(1)")
    ap.add_argument("--n-sims", type=int, default=1000,
                    help="Poisson simulations per multiplier "
                         "(default: 1000)")
    args = ap.parse_args()

    names, lat, lon, streams, canon = sk.corpus_stream()
    keep, Mfull, pfull, Ntok = sk.corpus_mstar(names, streams, canon)
    xyzb = sk.unit_vectors(lat[keep], lon[keep])
    wb, cosb = sk.density_weights(xyzb)
    swb = np.sqrt(wb)[:, None]
    c_fit = sk.fit_zonal(Mfull, cosb, wb)
    Cth = sk.theory_matrix(c_fit, cosb)
    lam_t, U_t = weighted_modes_signed(Cth, swb)

    def prep(M, c_ref):
        """Pin to the reference kernel's c(1), or keep the measured
        diagonal (symmetrized only) under --no-pin."""
        if args.no_pin:
            return 0.5 * (M + M.T)
        return sk.pin_diagonal(M, c_ref)

    def dip(M):
        lam_e, U_e = weighted_modes_signed(prep(M, c_fit), swb, 12)
        E = U_e * np.sqrt(np.abs(lam_e))
        _, cm, cmed, _, _ = sk.block_compare(U_t, lam_t, slice(0, 3),
                                             E, U_e, swb, match="fixed")
        return cm, cmed

    def dip_selffit(M):
        c_s = sk.fit_zonal(M, cosb, wb)
        lam_s, U_s = weighted_modes_signed(sk.theory_matrix(c_s, cosb), swb)
        # simulated corpora have NO measured diagonal (draws are
        # off-diagonal only); the alpha-floor there is an artifact, not
        # data, so simulations are ALWAYS pinned -- --no-pin concerns
        # measured diagonals of the observed matrices only
        lam_e, U_e = weighted_modes_signed(sk.pin_diagonal(M, c_s), swb, 12)
        E = U_e * np.sqrt(np.abs(lam_e))
        _, cm, cmed, _, _ = sk.block_compare(U_s, lam_s, slice(0, 3),
                                             E, U_e, swb, match="fixed")
        return cm, cmed

    tF = dip(Mfull)

    Pexp = np.outer(pfull, pfull) * (2 + Cth) / (2 - Cth)
    lam_counts = np.clip(Pexp * (2.0 * sk.WINDOW * Ntok), 1e-8, None)
    rng = np.random.default_rng(0)
    mults = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    means, los, his = [], [], []
    print("POISSON SIMULATION under zonal truth "
          f"(dipole cos, {args.n_sims} draws, self-fit matched):")
    for mult in mults:
        ms = []
        for _ in range(args.n_sims):
            Ws = rng.poisson(lam_counts * mult).astype(float)
            Ws = np.triu(Ws, 1)
            Ws = Ws + Ws.T + sk.ALPHA
            P = Ws / (2.0 * sk.WINDOW * Ntok * mult)
            M = (P - np.outer(pfull, pfull)) \
                / (0.5 * (P + np.outer(pfull, pfull)))
            ms.append(dip_selffit(0.5 * (M + M.T))[0])
        means.append(np.mean(ms))
        los.append(np.percentile(ms, 5))
        his.append(np.percentile(ms, 95))
        print(f"    counts x{mult:>5}: mean cos {np.mean(ms):.2f}  "
              f"[5-95%: {np.percentile(ms, 5):.2f},"
              f"{np.percentile(ms, 95):.2f}]")
        if mult == 1:
            ms1 = np.asarray(ms)
    print(f"    OBSERVED (x1 real corpus): {tF[0]:.2f}/{tF[1]:.2f}")
    pct = 100.0 * float(np.mean(ms1 <= tF[0]))
    print(f"    OBSERVED vs x1 sims: percentile {pct:.1f}% "
          f"({int(np.sum(ms1 <= tF[0]))}/{len(ms1)} sims <= observed)")

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.plot(mults, means, "o-", color="tab:blue",
            label="simulated: zonal truth + Poisson counts")
    ax.fill_between(mults, los, his, color="tab:blue", alpha=0.15,
                    label="5-95% of simulations")
    ax.scatter([1], [tF[0]], marker="*", s=240, color="tab:orange",
               zorder=5, label=f"book corpus ({tF[0]:.2f})")
    ax.axhline(0.92, color="tab:red", ls="--", lw=1.2)
    ax.text(0.02, 0.86,
            "Mistral Large 2\nper-object cosine\nsimilarity (0.92)",
            color="tab:red", fontsize=17, va="top", ha="left",
            linespacing=1.15, transform=ax.transAxes)
    ax.set_xscale("log")
    ax.set_xticks(mults)
    ax.set_xticklabels(["1/4", "1/2", "1", "2", "4", "8", "16", "32", "64", "128"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("pair-count multiplier (x book corpus)", fontsize=18)
    ax.set_ylabel("per-object cosine similarity", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=16)
    fig.tight_layout()
    fig.savefig("book_corpus_size_effect.pdf", bbox_inches="tight")
    fig.savefig("book_corpus_size_effect.png", dpi=200,
                bbox_inches="tight")
    print("saved -> book_corpus_size_effect.pdf + "
          "book_corpus_size_effect.png")


if __name__ == "__main__":
    main()
