"""
sky_battery.py
--------------
The "not just a sphere -- spherical harmonics" exhibit: run the
homogeneous-space (S^2 / Gelfand pair) battery on the LLM residual
activations of Berdnikov & Liokumovich's "Decodable sky" (188 sky objects
x 25 proximity prompts x layers, top-16 PCA per layer).

Per layer:
  * zonality: isotropy R^2 of the object-mean activation Gram (18 angle
    bins, density-weighted), with a location-permutation null
  * degree purity of the leading modes vs sampled harmonics (weighted)
  * the (2l+1) multiplicity test: eigenvalue spectrum colored by degree
  * Funk-Hecke spectrum of the fitted zonal kernel
  * weighted decode R^2 of (x, y, z)
  * canonical correlations vs the sampled zonal-theory modes

Outputs: printed numbers + sky_battery_{model}.pdf (+ .png preview)
(4 panels: layer sweep, degree-colored spectrum, zonal kernel,
degree-purity heatmap).

Usage: python3 sky_battery.py [--model mistrallarge123b] [--lmax 4]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_legendre

import skylib as sk

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def layer_battery(Xl, lat, lon, xyz, lmax, nperm=60, rng=None):
    n = Xl.shape[0]
    rng = rng or np.random.default_rng(0)
    w, cosT = sk.density_weights(xyz)
    ang = np.arccos(cosT)
    sw = np.sqrt(w)[:, None]

    Xc = Xl - Xl.mean(0, keepdims=True)
    M = Xc @ Xc.T
    M = M / np.sqrt(np.mean(np.diag(M) ** 2))
    iu = np.triu_indices(n, 1)
    a_p, w_p, m_p = ang[iu], w[iu[0]] * w[iu[1]], M[iu]

    # zonality with permutation null
    edges = np.linspace(0, np.pi, 19)

    def binfit(a_f, y_f, w_f, a_e):
        which = np.clip(np.digitize(a_f, edges) - 1, 0, 17)
        mu = np.full(18, np.sum(w_f * y_f) / w_f.sum())
        for b in range(18):
            m = which == b
            if m.sum() > 2:
                mu[b] = np.sum(w_f[m] * y_f[m]) / w_f[m].sum()
        return mu[np.clip(np.digitize(a_e, edges) - 1, 0, 17)]

    def wr2(y, yh, wp):
        mu = np.sum(wp * y) / wp.sum()
        return 1 - np.sum(wp * (y - yh) ** 2) / np.sum(wp * (y - mu) ** 2)

    r2_iso = wr2(m_p, binfit(a_p, m_p, w_p, a_p), w_p)
    null = []
    for _ in range(nperm):
        pm = rng.permutation(n)
        ap = ang[pm][:, pm][iu]
        wp = w[pm][iu[0]] * w[pm][iu[1]]
        null.append(wr2(m_p, binfit(ap, m_p, wp, ap), wp))
    null95 = float(np.quantile(null, 0.95))

    # modes + degree content
    lam, U = np.linalg.eigh(0.5 * (M + M.T))
    o = np.argsort(-np.abs(lam))
    lam, U = lam[o], U[:, o]
    bases = sk.weighted_degree_bases(sk.real_harmonics(lat, lon, lmax), w)

    def denergy(v):
        vw = sw[:, 0] * v
        vw /= np.linalg.norm(vw)
        return {l: float(np.sum((bases[l].T @ vw) ** 2))
                for l in range(lmax + 1)}

    modes = [denergy(U[:, k]) for k in range(min(16, n - 1))]

    # decode
    E = U[:, :16] * np.sqrt(np.abs(lam[:16]))
    beta, *_ = np.linalg.lstsq(sw * E, sw * xyz, rcond=None)
    xyz_hat = E @ beta
    mu = (w[:, None] * xyz).sum(0) / w.sum()
    r2_dec = 1 - (w[:, None] * (xyz - xyz_hat) ** 2).sum() \
        / (w[:, None] * (xyz - mu) ** 2).sum()

    # zonal theory + canonical correlations + Funk-Hecke
    c_fit = sk.estimate_c(cosT[iu], m_p, w_p)
    # local SD under the same NW weights as c_fit (for a ±1 SD band)
    x_pairs = cosT[iu]
    K = np.exp(-((x_pairs[None, :] - sk.XQ[:, None]) ** 2)
               / (2 * sk.HB ** 2)) * w_p[None, :]
    den = K.sum(1) + 1e-12
    m1 = (K * m_p[None, :]).sum(1) / den
    m2 = (K * (m_p[None, :] ** 2)).sum(1) / den
    c_std = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
    C = sk.theory_matrix(c_fit, cosT)
    lam_t, U_t = sk.weighted_modes(C, sw)
    Uw_emp, _ = np.linalg.qr(sw * U[:, :16])
    s3 = np.linalg.svd(U_t[:, :3].T @ Uw_emp, compute_uv=False)
    s8 = np.linalg.svd(U_t[:, :8].T @ Uw_emp, compute_uv=False)
    Pl = np.array([eval_legendre(l, sk.XQ) for l in range(lmax + 1)])
    lam_fh = 2 * np.pi * np.trapezoid(c_fit[None, :] * Pl, sk.XQ, axis=1)

    return dict(r2_iso=r2_iso, null95=null95, r2_dec=r2_dec,
                lam=lam[:16], modes=modes, c3=float(np.mean(s3 ** 2)),
                c8=float(np.mean(s8 ** 2)), lam_fh=lam_fh, c_fit=c_fit,
                c_std=c_std,
                theta_pairs=np.degrees(a_p), m_pairs=m_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistrallarge123b")
    ap.add_argument("--lmax", type=int, default=4)
    ap.add_argument("--out", default=None,
                    help="output path stem or .pdf/.png (writes both formats)")
    ap.add_argument("--heatmap-out", default=None,
                    help="also save the degree-purity heatmap alone to this "
                         "path stem (writes .pdf and .png)")
    args = ap.parse_args()
    out = args.out or f"sky_battery_{args.model}"
    base, ext = os.path.splitext(out)
    if ext.lower() in (".pdf", ".png"):
        out_base = base
    else:
        out_base = out

    X, lat, lon, xyz, names = sk.load_activations(args.model)
    Lyr = X.shape[0]
    print(f"{args.model}: {X.shape[1]} objects, {Lyr} layers, "
          f"{X.shape[2]} dims/layer")

    sweep_layers = list(range(0, Lyr, max(1, Lyr // 30)))
    sweep = []
    for l in sweep_layers:
        r = layer_battery(X[l], lat, lon, xyz, args.lmax, nperm=20)
        sweep.append((l, r["r2_iso"], r["r2_dec"], r["c3"]))
    best = max(sweep, key=lambda t: t[2])[0]
    print(f"best layer by decode R^2: {best}")
    r = layer_battery(X[best], lat, lon, xyz, args.lmax, nperm=100)

    print(f"\nlayer {best}:")
    print(f"  zonality (isotropy R^2 of activation Gram): {r['r2_iso']:.3f}"
          f"   (perm null95 {r['null95']:.3f})")
    print(f"  decode R^2 (weighted): {r['r2_dec']:.3f}")
    print(f"  canon^2 vs zonal theory: top3 {r['c3']:.2f}, "
          f"top8 {r['c8']:.2f}")
    print(f"  Funk-Hecke lambda_l: {np.round(r['lam_fh'][1:], 3)}")
    print(f"  {'mode':>5} {'lam':>7}  degree energies (l=0..{args.lmax})")
    for k, m in enumerate(r["modes"][:10]):
        best_l = max(m, key=m.get)
        print(f"  {k+1:>5} {r['lam'][k]:>7.2f}  " +
              " ".join(f"{m[l]:.2f}" for l in range(args.lmax + 1)) +
              f"   -> l={best_l} ({m[best_l]:.2f})")

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(2, 2, 1)
    ls, isos, decs, c3s = zip(*sweep)
    ax.plot(ls, decs, "o-", label="decode $R^2$")
    ax.plot(ls, isos, "s-", label="Gram zonality $R^2$")
    ax.plot(ls, c3s, "^-", label="canon$^2$ vs zonal theory (top3)")
    ax.axvline(best, color="gray", lw=0.6)
    ax.set_xlabel("layer")
    ax.legend(fontsize=8)
    ax.set_title(f"(a) {args.model}: sky structure across layers")

    ax = fig.add_subplot(2, 2, 2)
    colors = plt.cm.tab10(np.arange(args.lmax + 1))
    for k, m in enumerate(r["modes"]):
        bl = max(m, key=m.get)
        ax.bar(k + 1, r["lam"][k], color=colors[bl],
               alpha=0.35 + 0.65 * m[bl])
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[l])
               for l in range(args.lmax + 1)]
    ax.legend(handles, [f"$\\ell$={l}" for l in range(args.lmax + 1)],
              fontsize=8)
    ax.set_xlabel("empirical mode")
    ax.set_ylabel("eigenvalue")
    ax.set_title(f"(b) layer {best}: mode spectrum colored by degree\n"
                 "(harmonic-embedding prediction: 3/5/7 blocks)")

    ax = fig.add_subplot(2, 2, 3)
    thd = np.degrees(np.arccos(sk.XQ))
    oq = np.argsort(thd)
    ax.plot(thd[oq], r["c_fit"][oq], lw=2, color="tab:orange")
    ax.set_xlabel("true angular separation on the sky (deg)")
    ax.set_ylabel(r"$c(\theta)$")
    ax.set_title(f"(c) zonal kernel of the activation Gram\n"
                 f"(isotropy $R^2$ {r['r2_iso']:.2f} vs null "
                 f"{r['null95']:.2f})")

    ax = fig.add_subplot(2, 2, 4)
    Pm = np.array([[m[l] for l in range(args.lmax + 1)]
                   for m in r["modes"]])
    im = ax.imshow(Pm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(args.lmax + 1))
    ax.set_xticklabels([f"$\\ell$={l}" for l in range(args.lmax + 1)])
    ax.set_ylabel("empirical mode")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("(d) degree purity of the modes")
    fig.tight_layout()
    pdf_path = out_base + ".pdf"
    png_path = out_base + ".png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"saved -> {pdf_path} + {png_path}")

    if args.heatmap_out:
        hm_base = os.path.splitext(args.heatmap_out)[0]
        os.makedirs(os.path.dirname(hm_base) or ".", exist_ok=True)
        fig_h, ax = plt.subplots(figsize=(5, 6))
        im = ax.imshow(Pm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(args.lmax + 1))
        ax.set_xticklabels([f"$\\ell$={l}" for l in range(args.lmax + 1)])
        ax.set_yticks(range(Pm.shape[0]))
        ax.set_yticklabels([str(k + 1) for k in range(Pm.shape[0])])
        ax.set_ylabel("empirical mode")
        plt.colorbar(im, ax=ax, fraction=0.046, label="degree purity")
        ax.set_title(f"{args.model}, layer {best}")
        fig_h.tight_layout()
        fig_h.savefig(hm_base + ".pdf", bbox_inches="tight")
        fig_h.savefig(hm_base + ".png", dpi=200, bbox_inches="tight")
        print(f"saved -> {hm_base}.pdf + {hm_base}.png")


if __name__ == "__main__":
    main()