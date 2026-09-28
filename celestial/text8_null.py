"""
text8_null.py -- negative control: a general corpus need not
carry sky-sphere statistics. Runs the selection battery on text8
(17M tokens of cleaned Wikipedia; ~130x the star-guide corpus):

  * mention scan of all 188 sky-object names (uni- and bigrams) with the
    top-count table exhibiting the polysemy problem ('algol' the
    programming language, 'peacock' the bird, 'mimosa' the plant);
  * coherence: median angular hop between successive DISTINCT object
    mentions vs the shuffled-mention null (z);
  * lambda_1 of the corpus M* kernel with a one-sided
    location-permutation p;
  * D = corr(M*_ij, cos theta_ij) with the same location-permutation null;
  * kernel correlation with the LLM activation kernel.

Expected output (text8, Mistral Large 2 layer 72):
  coherence z ~ 3.5; lambda_1 ~ 0.000, p ~ 0.5;
  D ~ 0.01, p ~ 0.5; kernel corr r ~ +0.18.

Usage: python3 text8_null.py [--text8 ../months/text8.txt]
                                      [--model mistrallarge123b]
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.special import eval_legendre
import skylib as sk

_DEFAULT_TEXT8 = str(Path(__file__).resolve().parent.parent / "months" / "text8.txt")

WIN, ALPHA, MINC = 30, 0.25, 5
NPERM = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text8", default=_DEFAULT_TEXT8)
    ap.add_argument("--model", default="mistrallarge123b")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    names, lat, lon, aliases = sk.load_objects()
    xyzO = sk.unit_vectors(lat, lon)
    uni = {n.lower(): i for i, n in enumerate(names) if " " not in n}
    bi = {n.lower(): i for i, n in enumerate(names) if n.count(" ") == 1}

    toks = open(args.text8).read().split()
    N = len(toks)
    occ = []
    for p, t in enumerate(toks):
        if t in uni:
            occ.append((p, uni[t]))
        if p + 1 < N:
            big = t + " " + toks[p + 1]
            if big in bi:
                occ.append((p, bi[big]))
    cnt = {}
    for _, i in occ:
        cnt[i] = cnt.get(i, 0) + 1
    print(f"text8: {N} tokens; {len(occ)} sky-object mentions; "
          f"{sum(1 for v in cnt.values() if v >= MINC)} objects with "
          f">={MINC}")
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[:8]
    print("most frequent (note the polysemy):",
          [(names[i], v) for i, v in top])

    # coherence (consecutive same-object pairs masked)
    seq = np.array([i for _, i in occ])
    def med(s):
        a, b = s[:-1], s[1:]
        m = a != b
        return np.degrees(np.median(np.arccos(np.clip(
            np.sum(xyzO[a[m]] * xyzO[b[m]], 1), -1, 1))))
    obs = med(seq)
    nullc = [med(rng.permutation(seq)) for _ in range(150)]
    z = (np.mean(nullc) - obs) / max(np.std(nullc), 1e-9)
    print(f"coherence: median hop {obs:.0f} deg vs shuffled "
          f"{np.mean(nullc):.0f} deg, z = {z:.1f}")

    # ---- corpus M* over well-counted objects ----
    keep = sorted([i for i, v in cnt.items() if v >= MINC])
    ki = {i: j for j, i in enumerate(keep)}
    kpos = [(p, ki[i]) for p, i in occ if i in ki]
    K = len(keep)
    Wm = np.zeros((K, K))
    for a in range(len(kpos)):
        b = a + 1
        while b < len(kpos) and kpos[b][0] - kpos[a][0] <= WIN:
            if kpos[b][0] > kpos[a][0]:
                Wm[kpos[a][1], kpos[b][1]] += 1
                Wm[kpos[b][1], kpos[a][1]] += 1
            b += 1
    c_k = np.array([cnt[i] for i in keep], float)
    P = (Wm + ALPHA) / (2.0 * WIN * N)
    pv = c_k / N
    M = (P - np.outer(pv, pv)) / (0.5 * (P + np.outer(pv, pv)))
    M = 0.5 * (M + M.T)
    V = xyzO[keep]
    CT = np.clip(V @ V.T, -1, 1)
    iu = np.triu_indices(K, k=1)

    # Funk-Hecke lambda_1 (NW kernel + P1 projection)
    grid = np.linspace(-0.999, 0.999, 201)
    P1g = eval_legendre(1, grid)
    def lam1(CTm):
        t, y = CTm[iu], M[iu]
        w = np.exp(-0.5 * ((grid[:, None] - t[None, :]) / 0.08) ** 2)
        chat = (w @ y) / np.maximum(w.sum(1), 1e-12)
        return float(np.trapezoid(chat * P1g, grid)), chat
    l1, chat = lam1(CT)
    nulls = np.array([lam1(CT[np.ix_(pm, pm)])[0]
                      for pm in (rng.permutation(K) for _ in range(NPERM))])
    p1 = float((nulls >= l1).mean())
    print(f"lambda_1 = {l1:+.4f}, p = {p1:.2f}")

    # D = corr(M*, cos theta)
    zc = lambda v: (v - v.mean()) / max(v.std(), 1e-12)
    D = float(np.mean(zc(M[iu]) * zc(CT[iu])))
    nullD = np.array([float(np.mean(zc(M[iu]) *
                                    zc(CT[np.ix_(pm, pm)][iu])))
                      for pm in (rng.permutation(K)
                                 for _ in range(NPERM))])
    pD = float((nullD >= D).mean())
    print(f"D = {D:+.3f}, p = {pD:.2f}")

    # kernel corr with the model
    X, latL, lonL, oxyz, onames = sk.load_activations(args.model)
    Xl = X[72] - X[72].mean(0)
    G = Xl @ Xl.T
    G = G / np.sqrt(np.mean(np.diag(G) ** 2))
    CTf = np.clip(oxyz @ oxyz.T, -1, 1)
    iuf = np.triu_indices(len(G), k=1)
    w = np.exp(-0.5 * ((grid[:, None] - CTf[iuf][None, :]) / 0.08) ** 2)
    chat_llm = (w @ G[iuf]) / np.maximum(w.sum(1), 1e-12)
    r = float(np.corrcoef(zc(chat), zc(chat_llm))[0, 1])
    print(f"kernel corr with {args.model} layer-72 kernel: r = {r:+.2f}")


if __name__ == "__main__":
    main()