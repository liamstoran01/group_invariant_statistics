"""
months.py -- four panels (a)/(b)/(c1)/(c2) in a single horizontal row.
Usage: python3 months.py [--corpus text8.txt] [--out months.pdf]
Writes months.pdf and months.png by default.
"""
import argparse
import warnings
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType (crisp in LaTeX)
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"  # less jagged than default
import matplotlib.pyplot as plt
import numpy as np

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
L = 16
NBLOCKS = 40
NBOOT = 300
DIGITS = set("zero one two three four five six seven eight nine".split())


def is_month_sense(toks, p):
    nxt1 = toks[p + 1] if p + 1 < len(toks) else ""
    nxt2 = toks[p + 2] if p + 2 < len(toks) else ""
    prv1 = toks[p - 1] if p >= 1 else ""
    return ((nxt1 in DIGITS and nxt2 in DIGITS) or prv1 in DIGITS
            or nxt1 in MONTHS or prv1 in MONTHS)


def build_blocks(path):
    """Per-corpus-block pair-weight and count accumulators (for the
    block bootstrap) plus the full-corpus totals."""
    toks = open(path).read().split()
    N = len(toks)
    idx = {m: i for i, m in enumerate(MONTHS)}
    pos = [(p, idx[t]) for p, t in enumerate(toks)
           if t in idx and (t != "may" or is_month_sense(toks, p))]
    kept_may = sum(1 for _, i in pos if i == 4)
    print(f"'may' kept as month-sense: {kept_may} "
          f"(of {sum(1 for t in toks if t == 'may')})")
    bsz = N // NBLOCKS + 1
    Wb = np.zeros((NBLOCKS, 12, 12))
    cb = np.zeros((NBLOCKS, 12))
    nb = np.zeros(NBLOCKS)
    for p, i in pos:
        cb[p // bsz, i] += 1
    for b in range(NBLOCKS):
        nb[b] = min(bsz, N - b * bsz)
    for a in range(len(pos)):
        b = a + 1
        while b < len(pos) and pos[b][0] - pos[a][0] <= L:
            d = pos[b][0] - pos[a][0]
            if d >= 1:
                i, j = pos[a][1], pos[b][1]
                w = L + 1 - d
                blk = pos[a][0] // bsz
                Wb[blk, i, j] += w
                Wb[blk, j, i] += w
            b += 1
    return Wb, cb, nb


def mstar_from(W, cnt, N):
    fmass = sum(L + 1 - d for d in range(1, L + 1))
    P = W / (2.0 * fmass * N)
    p = cnt / N
    M = (P - np.outer(p, p)) / (0.5 * (P + np.outer(p, p)))
    return 0.5 * (M + M.T)


def class_kernel(M, dist):
    return np.array([M[dist == d].mean() for d in range(7)])


def fourier_basis():
    t = np.arange(12)
    blocks = {0: np.ones((12, 1)) / np.sqrt(12)}
    for k in range(1, 6):
        cth = np.cos(2 * np.pi * k * t / 12)
        sth = np.sin(2 * np.pi * k * t / 12)
        blocks[k] = np.stack([cth / np.linalg.norm(cth),
                              sth / np.linalg.norm(sth)], 1)
    a = np.cos(np.pi * t)
    blocks[6] = (a / np.linalg.norm(a))[:, None]
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="text8.txt")
    ap.add_argument("--out", default="months.pdf")
    args = ap.parse_args()

    Wb, cb, nb = build_blocks(args.corpus)
    W, cnt, N = Wb.sum(0), cb.sum(0), nb.sum()
    M = mstar_from(W, cnt, N)
    print(f"months M* from {args.corpus} "
          f"(counts {int(cnt.min())}-{int(cnt.max())})")

    dist = np.abs(np.arange(12)[:, None] - np.arange(12)[None, :])
    dist = np.minimum(dist, 12 - dist)
    c = class_kernel(M, dist)
    blocks = fourier_basis()
    C = c[dist]
    lam_k = {k: float(blocks[k][:, 0] @ C @ blocks[k][:, 0]) for k in blocks}
    print("kernel C(d):", np.round(c, 3))
    print("Fourier eigenvalues lambda_k:",
          {k: round(v, 3) for k, v in lam_k.items()})

    # -------- block bootstrap: kernel band + eigenvalue error bars ------
    rng = np.random.default_rng(0)
    Pc = np.eye(12) - np.ones((12, 12)) / 12.0    # centering projector
    cboot, lamboot = [], []
    for _ in range(NBOOT):
        sel = rng.integers(0, NBLOCKS, NBLOCKS)
        Mb = mstar_from(Wb[sel].sum(0), cb[sel].sum(0), nb[sel].sum())
        cboot.append(class_kernel(Mb, dist))
        lb = np.linalg.eigvalsh(Pc @ Mb @ Pc)
        lamboot.append(np.sort(np.abs(lb))[::-1][:7])
    cboot = np.array(cboot)
    clo, chi = np.quantile(cboot, [0.025, 0.975], axis=0)
    lamboot = np.array(lamboot)
    lo = np.quantile(lamboot, 0.025, axis=0)
    hi = np.quantile(lamboot, 0.975, axis=0)
    print("C(d) 95% CI half-widths:", np.round((chi - clo) / 2, 3))
    # within-class scatter: does the data follow the kernel?
    def class_r2(include_diag):
        k = 0 if include_diag else 1
        iu, ju = np.triu_indices(12, k=k)
        allv = np.array([M[i, j] for i, j in zip(iu, ju)])
        gm = allv.mean()
        sds, sq_w, sq_t = [], 0.0, 0.0
        for d in range(0 if include_diag else 1, 7):
            vals = np.array([M[i, j] for i, j in zip(iu, ju)
                             if dist[i, j] == d])
            sds.append(vals.std())
            sq_w += ((vals - c[d]) ** 2).sum()
            sq_t += ((vals - gm) ** 2).sum()
        return 1 - sq_w / sq_t, sds

    r2_off, sds = class_r2(False)
    r2_all, _ = class_r2(True)
    print("within-class SD by d=1..6:", np.round(sds, 3))
    print(f"class-averaging R^2 (off-diag): {r2_off:.3f}")
    print(f"class-averaging R^2 (with diag): {r2_all:.3f}")
    dup = cboot[:, 6] - cboot[:, 4]
    print(f"half-year uptick C(6)-C(4): {c[6]-c[4]:.3f} "
          f"(95% CI [{np.quantile(dup, .025):.3f}, "
          f"{np.quantile(dup, .975):.3f}])")

    # -------- empirical factorization of the CENTERED statistics --------
    Mc = Pc @ M @ Pc
    lam_e, U_e = np.linalg.eigh(Mc)
    oe = np.argsort(-np.abs(lam_e))
    lam_e, U_e = lam_e[oe], U_e[:, oe]

    def wavenumber(v):
        en = {k: float(np.sum((blocks[k].T @ v) ** 2)) for k in blocks}
        best = max(en, key=en.get)
        return best, en[best]

    print(f"\n{'mode':>5} {'lam':>8}  wavenumber (purity)")
    labels = []
    for m in range(7):
        k, pur = wavenumber(U_e[:, m])
        labels.append(k)
        print(f"{m+1:>5} {lam_e[m]:>8.3f}  k={k} ({pur:.2f})")

    E = U_e * np.sqrt(np.abs(lam_e))
    theory4 = np.column_stack([blocks[1] * np.sqrt(max(lam_k[1], 0)),
                               blocks[2] * np.sqrt(max(lam_k[2], 0))])
    emp4 = E[:, 0:4]                        # centered: k=1 -> PCs 1-2
    A = np.zeros((12, 4))
    for sl in [slice(0, 2), slice(2, 4)]:
        Tb, Eb = theory4[:, sl], emp4[:, sl]
        U2, _, V2 = np.linalg.svd(Tb.T @ Eb)
        A[:, sl] = Tb @ (U2 @ V2)
    cos = [emp4[i] @ A[i] / max(np.linalg.norm(emp4[i]) *
                                np.linalg.norm(A[i]), 1e-12)
           for i in range(12)]
    null = []
    for _ in range(500):
        pm = rng.permutation(12)
        An = np.zeros((12, 4))
        for sl in [slice(0, 2), slice(2, 4)]:
            Tb, Eb = theory4[pm][:, sl], emp4[:, sl]
            U2, _, V2 = np.linalg.svd(Tb.T @ Eb)
            An[:, sl] = Tb @ (U2 @ V2)
        null.append(np.mean([emp4[i] @ An[i] /
                             max(np.linalg.norm(emp4[i]) *
                                 np.linalg.norm(An[i]), 1e-12)
                             for i in range(12)]))
    print(f"\nper-month cos: {np.mean(cos):.2f}/{np.median(cos):.2f} "
          f"(label-shuffle null {np.mean(null):.2f}, "
          f"95% {np.quantile(null, 0.95):.2f})")
    worst = np.argsort(cos)[:2]
    print("worst-predicted months:",
          [(MONTHS[i], round(cos[i], 2)) for i in worst])
    ang = np.degrees(np.arctan2(emp4[:, 1], emp4[:, 0]))
    steps = np.abs(((np.diff(np.concatenate([ang, ang[:1]])) + 180)
                    % 360) - 180)
    ang2 = np.degrees(np.arctan2(emp4[:, 3], emp4[:, 2]))
    steps2 = np.abs(((np.diff(np.concatenate([ang2, ang2[:1]])) + 180)
                     % 360) - 180)
    print(f"median steps: k=1 {np.median(steps):.1f} deg (theory 30), "
          f"k=2 {np.median(steps2):.1f} deg (theory 60)")

    # ------------------------------ figure ------------------------------
    plt.rcParams.update({"font.size": 18, "axes.titlesize": 21,
                         "axes.labelsize": 20, "xtick.labelsize": 16,
                         "ytick.labelsize": 16, "legend.fontsize": 16})
    # keep (a)/(b) at prior absolute size; grow only (c1)/(c2) a bit
    fig, axs = plt.subplots(
        1, 4, figsize=(21.8, 5.2),
        gridspec_kw={"wspace": 0.30, "width_ratios": [1.25, 1.25, 1.15, 1.15]})

    ax = axs[0]
    colors = plt.cm.tab10(np.arange(7))
    for m in range(7):
        ax.bar(m + 1, lam_e[m], color=colors[labels[m]], alpha=0.9)
        mag = abs(lam_e[m])
        ax.errorbar(m + 1, lam_e[m],
                    yerr=[[max(mag - lo[m], 0)], [max(hi[m] - mag, 0)]],
                    fmt="none", ecolor="k", lw=1.4, capsize=4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k])
               for k in (1, 2, 3, 4)]
    ax.legend(handles, [f"$k$={k}" for k in (1, 2, 3, 4)], fontsize=18)
    ax.set_xlabel("empirical mode")
    ax.set_ylabel("eigenvalue")
    ax.yaxis.set_label_coords(-0.14, 0.5)  # left of tick numbers
    ax.set_title("(a) Eigenvalues Pair")

    ax = axs[1]
    rng2 = np.random.default_rng(1)
    for d in range(7):
        iu, ju = np.triu_indices(12, k=0 if d == 0 else 1)
        vals = np.array([M[i, j] for i, j in zip(iu, ju)
                         if dist[i, j] == d])
        xj = d + rng2.uniform(-0.13, 0.13, len(vals))
        ax.scatter(xj, vals, s=26, color="tab:orange", alpha=0.45,
                   edgecolors="none",
                   label="individual month pairs" if d == 0 else None)
    ax.plot(range(7), c, "o-", color="tab:orange", lw=2.5, ms=8,
            label="mean co-occurrence\nC(d)", zorder=3)
    ax.set_xlabel(r"circular distance, $d$")
    ax.set_ylabel(r"$\mathrm{C}(d)$")
    ax.legend(fontsize=16, loc="lower left", framealpha=0.45,
              borderaxespad=0.4, handlelength=1.6, labelspacing=0.35,
              borderpad=0.4)
    ax.set_title("(b) Statistical Invariance Holds")

    mcol = plt.cm.hsv(np.arange(12) / 12.0)
    order = list(range(12)) + [0]
    for j, (sl, kk, pcs) in enumerate([(slice(0, 2), 1, ("PC 1", "PC 2")),
                                       (slice(2, 4), 2, ("PC 3", "PC 4"))]):
        ax = axs[2 + j]
        Epl, Apl = emp4[:, sl], A[:, sl]
        ax.plot(Epl[order, 0], Epl[order, 1], "-", color="gray",
                lw=1.6, alpha=0.7, zorder=1)
        # k=2 theory repeats every 6 months; one lap avoids dashed-line overlap
        th_order = (list(range(6)) + [0]) if kk == 2 else order
        ax.plot(Apl[th_order, 0], Apl[th_order, 1], "--", color="gray",
                lw=1.8, alpha=0.85, zorder=1)
        ax.scatter(Epl[:, 0], Epl[:, 1], color=mcol, s=140, zorder=3)
        ax.scatter(Apl[:, 0], Apl[:, 1], facecolors="none",
                   edgecolors=mcol, s=190, linewidths=2.4, zorder=3)
        for i in range(12):
            ax.annotate(MONTHS[i][:3], (Epl[i, 0], Epl[i, 1]),
                        fontsize=17, xytext=(5, 5),
                        textcoords="offset points")
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(bottom=False, left=False, labelbottom=False,
                       labelleft=False)
        ax.set_xlabel(pcs[0])
        ax.set_ylabel(pcs[1])
    h_emp = plt.Line2D([], [], marker="o", ls="", color="k", ms=11)
    h_th = plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                      markeredgecolor="k", ms=12, markeredgewidth=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=(0, 0.16, 1, 0.92))
    fig.subplots_adjust(wspace=0.30)
    # shared (c) title above both plane panels; legend centered under them
    pos2, pos3 = axs[2].get_position(), axs[3].get_position()
    cx = 0.5 * (pos2.x0 + pos3.x1)
    fig.text(cx, pos2.y1 + 0.04, "(c) Circles Follow",
             ha="center", va="bottom", fontsize=21,
             transform=fig.transFigure)
    fig.legend([h_emp, h_th], ["empirical", "theory"],
               loc="upper center", ncol=2, fontsize=20, frameon=False,
               bbox_to_anchor=(cx, pos2.y0 - 0.07),
               bbox_transform=fig.transFigure)
    fig.savefig(args.out, bbox_inches="tight")
    png_out = (args.out[:-4] + ".png" if args.out.lower().endswith(".pdf")
               else args.out + ".png")
    fig.savefig(png_out, bbox_inches="tight", dpi=170)
    print("saved ->", args.out)
    print("saved ->", png_out)


if __name__ == "__main__":
    main()
