"""
mine_cube_trichotomy.py
-----------------------
Mine a text corpus for the gender x number hypercube vocabulary (12
complete Z_2^2 cubes from the Mikolov analogy families: king/queen/
kings/queens, ...), build the windowed co-occurrence M* restricted to
these words, and produce a three-panel figure in the style of the
paper's Figure 3:

  (a) the centered PMI matrix over the mined words, ordered by
      (family, gender, number), showing the Z_2^2 block stamp;
  (b) the symmetry and law layers measured per family: Z_2^2
      invariance R^2 (pooling each family's 4x4 kernel by agreement
      class) and the Walsh amplitude spectrum C^(S) for
      S in {gender, number, gender x number} -- the last being the
      interaction dial that the independent-attribute model fixes to
      zero;
  (c) the irrep-level consequence: per-channel alignment of the two
      analogy differences (leave-one-out Walsh channels), exhibiting
      the predicted trichotomy +1 / 0 / -1, including the anti-aligned
      (enemy) interaction sector.

Corpus input (choose one):
  --wiki  path/to/enwiki-...-pages-articles.xml.bz2   (gensim WikiCorpus)
  --text  one or more plain-text files (whitespace tokenized, lowercased)
  --demo  download and use gensim's text8 (17M words) for a quick run

Example:
  python3 mine_cube_trichotomy.py --demo --window 10
  python3 mine_cube_trichotomy.py --wiki enwiki-latest.xml.bz2 --window 10

Writes cube_trichotomy.(png|pdf) and cube_trichotomy_metrics.json.
"""
import argparse
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (masc_sg, fem_sg, masc_pl, fem_pl) -- gender g = +/-1, number n = +/-1
QUADS = [("king", "queen", "kings", "queens"),
         ("prince", "princess", "princes", "princesses"),
         ("actor", "actress", "actors", "actresses"),
         ("waiter", "waitress", "waiters", "waitresses"),
         ("god", "goddess", "gods", "goddesses"),
         ("duke", "duchess", "dukes", "duchesses"),
         ("emperor", "empress", "emperors", "empresses"),
         ("hero", "heroine", "heroes", "heroines"),
         ("host", "hostess", "hosts", "hostesses"),
         ("lion", "lioness", "lions", "lionesses"),
         ("heir", "heiress", "heirs", "heiresses"),
         ("steward", "stewardess", "stewards", "stewardesses")]
SIGNS = [(+1, +1), (-1, +1), (+1, -1), (-1, -1)]   # (g, n) per slot


def iter_docs(args):
    if args.demo:
        import gensim.downloader as gdl
        path = gdl.load("text8", return_path=True)
        import gensim.utils as gu
        with gu.open(path, "rb") as fh:
            text = fh.read().decode("utf-8")
        yield text.split()
        return
    if args.wiki:
        from gensim.corpora import WikiCorpus
        wc = WikiCorpus(args.wiki, dictionary={}, lower=True)
        for toks in wc.get_texts():
            yield [t if isinstance(t, str) else t.decode("utf-8")
                   for t in toks]
        return
    for path in args.text:
        with open(path) as fh:
            for line in fh:
                toks = line.lower().split()
                if toks:
                    yield toks
    if not (args.demo or args.wiki or args.text):
        raise SystemExit("provide --demo, --wiki, or --text")


def count_corpus(args, targets):
    """Single pass: total windowed pair count, target unigrams, and
    pair counts among targets within the window."""
    tset = {w: i for i, w in enumerate(targets)}
    V = len(targets)
    pair = np.zeros((V, V))
    uni = np.zeros(V)
    total_tokens = 0
    W = args.window
    for toks in iter_docs(args):
        total_tokens += len(toks)
        hits = [(p, tset[t]) for p, t in enumerate(toks) if t in tset]
        for h, (p, i) in enumerate(hits):
            uni[i] += 1
            for (p2, j) in hits[h + 1:]:
                if p2 - p > W:
                    break
                pair[i, j] += 1
                pair[j, i] += 1
    return pair, uni, total_tokens


def build_pmi(pair, uni, total_tokens, window, smooth=0.5):
    """PMI_ij = log [ N_ij / E_indep(i,j) ] with the standard windowed
    normalization E = 2 W p_i p_j T."""
    p = uni / max(total_tokens, 1)
    E = 2.0 * window * total_tokens * np.outer(p, p)
    zero_frac = float(np.mean(pair == 0))
    PMI = np.log((pair + smooth) / np.maximum(E, 1e-300))
    PMI = 0.5 * (PMI + PMI.T)
    return PMI, zero_frac


def walsh_family(PMI, base):
    """Pooled agreement-class kernel and Walsh amplitudes of one
    family's 4x4 block. Returns (R^2 of Z_2^2 pooling, dict of
    amplitudes)."""
    idx = [base + s for s in range(4)]
    B = PMI[np.ix_(idx, idx)]
    G = np.array([SIGNS[s] for s in range(4)])
    # agreement class of (s, s') is (g s g s', n s n s')
    cls = {}
    for a in range(4):
        for b in range(4):
            key = (G[a, 0] * G[b, 0], G[a, 1] * G[b, 1])
            cls.setdefault(key, []).append((a, b))
    pooled = {k: float(np.mean([B[a, b] for a, b in v]))
              for k, v in cls.items()}
    fit = np.zeros_like(B)
    for k, v in cls.items():
        for a, b in v:
            fit[a, b] = pooled[k]
    r2 = 1 - np.var(B - fit) / max(np.var(B), 1e-300)
    keys = sorted(pooled)
    amp = {}
    for name, chi in [("gender", lambda c: c[0]),
                      ("number", lambda c: c[1]),
                      ("interaction", lambda c: c[0] * c[1])]:
        amp[name] = float(np.mean([pooled[k] * chi(k) for k in keys]))
    return r2, amp


def trichotomy(PMI, families, K):
    """Leave-one-out Walsh-channel alignment of the two analogy
    differences, from the spectral embedding of the mined PMI."""
    N = PMI.shape[0]
    H = np.eye(N) - np.ones((N, N)) / N
    lam, U = np.linalg.eigh(H @ PMI @ H)
    o = np.argsort(lam)[::-1]
    lam, U = lam[o][:K], U[:, o][:, :K]
    Wv = U * np.sqrt(np.clip(lam, 0, None))
    Vg, Vn, Vgn, D1, D2 = [], [], [], [], []
    for f in families:
        A, B, C, D = (Wv[4 * f + s] for s in range(4))
        Vg.append((A - B + C - D) / 4)
        Vn.append((A + B - C - D) / 4)
        Vgn.append((A - B - C + D) / 4)
        D1.append(A - B)
        D2.append(C - D)
    Vg, Vn, Vgn = map(np.stack, (Vg, Vn, Vgn))
    out = {"gender": [], "number": [], "interaction": []}
    energy = {k: [] for k in out}
    for q in range(len(families)):
        oth = [i for i in range(len(families)) if i != q]
        for key, Vmat in [("gender", Vg), ("number", Vn),
                          ("interaction", Vgn)]:
            _, _, Vt = np.linalg.svd(Vmat[oth], full_matrices=False)
            P = Vt[:2]
            p1, p2 = P @ D1[q], P @ D2[q]
            out[key].append(float(
                np.dot(p1, p2) / max(np.linalg.norm(p1)
                                     * np.linalg.norm(p2), 1e-12)))
            energy[key].append(float(
                (np.sum(p1**2) + np.sum(p2**2))
                / (np.sum(D1[q]**2) + np.sum(D2[q]**2))))
    return out, energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki")
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--min_pair", type=float, default=30,
                    help="min within-family pair mass to keep a family")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--out", default="cube_trichotomy")
    args = ap.parse_args()

    targets = [w for q in QUADS for w in q]
    pair, uni, T = count_corpus(args, targets)
    print(f"corpus: {T:,} tokens; target occurrences {int(uni.sum()):,}")
    keep = []
    for f in range(len(QUADS)):
        idx = slice(4 * f, 4 * f + 4)
        inpair = pair[idx, idx].sum()
        if uni[idx].min() > 0 and inpair >= args.min_pair:
            keep.append(f)
        else:
            print(f"  dropping family {QUADS[f][0]!r}: "
                  f"min unigram {int(uni[idx].min())}, "
                  f"in-family pairs {int(inpair)}")
    sel = [4 * f + s for f in keep for s in range(4)]
    pair = pair[np.ix_(sel, sel)]
    uni_s = uni[sel]
    PMI, zero_frac = build_pmi(pair, uni_s, T, args.window)
    print(f"{len(keep)} families kept; zero-count pair fraction "
          f"{100 * zero_frac:.1f}%")

    r2s, amps = [], []
    for f in range(len(keep)):
        r2, amp = walsh_family(PMI, 4 * f)
        r2s.append(r2)
        amps.append(amp)
    tri, energy = trichotomy(PMI, list(range(len(keep))), args.K)

    metrics = {"families": [QUADS[f][0] for f in keep],
               "tokens": T, "zero_frac": zero_frac,
               "invariance_r2": r2s, "walsh_amplitudes": amps,
               "trichotomy": tri, "channel_energy": energy}
    with open(f"{args.out}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=1)

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(14.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    N = PMI.shape[0]
    H = np.eye(N) - np.ones((N, N)) / N
    ax = fig.add_subplot(gs[0])
    Mc = H @ PMI @ H
    vmax = np.percentile(np.abs(Mc), 97)
    ax.imshow(Mc, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for f in range(1, len(keep)):
        ax.axhline(4 * f - 0.5, color="k", lw=0.4)
        ax.axvline(4 * f - 0.5, color="k", lw=0.4)
    ax.set_xticks([4 * f + 1.5 for f in range(len(keep))])
    ax.set_xticklabels([QUADS[f][0] for f in keep], rotation=90,
                       fontsize=7)
    ax.set_yticks([4 * f + 1.5 for f in range(len(keep))])
    ax.set_yticklabels([QUADS[f][0] for f in keep], fontsize=7)
    ax.set_title("(a) centered PMI over mined cubes\n"
                 "(family, gender, number ordering)", fontsize=10)

    ax = fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    for i, name in enumerate(["gender", "number", "interaction"]):
        vals = [abs(a[name]) for a in amps]
        x = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=22,
                   color=["steelblue", "gray", "crimson"][i])
        ax.hlines(np.mean(vals), i - 0.28, i + 0.28,
                  color=["steelblue", "gray", "crimson"][i], lw=2.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["|C(g)|", "|C(n)|", "|C(gn)|"], fontsize=9)
    ax.set_ylabel("Walsh amplitude of the family kernel")
    ax.set_title("(b) law layer: interaction dial is small\n"
                 f"but nonzero; invariance $R^2$ = "
                 f"{np.mean(r2s):.2f} $\\pm$ {np.std(r2s):.2f}",
                 fontsize=10)

    ax = fig.add_subplot(gs[2])
    for i, name in enumerate(["gender", "number", "interaction"]):
        vals = tri[name]
        sizes = 500 * np.array(energy[name]) + 12
        x = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=sizes, alpha=0.75, edgecolor="k",
                   linewidth=0.4,
                   color=["steelblue", "gray", "crimson"][i])
        ax.hlines(np.mean(vals), i - 0.28, i + 0.28,
                  color=["steelblue", "gray", "crimson"][i], lw=2.5)
    for yv, lab in [(1, "+1"), (0, "0"), (-1, "-1")]:
        ax.axhline(yv, color="k", lw=0.5, ls=":" if yv else "-")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["gender\n(equal)", "number\n(zero)",
                        "interaction\n(opposite)"], fontsize=8)
    ax.set_ylim(-1.15, 1.15)
    ax.set_ylabel("alignment of $W_A{-}W_B$ vs $W_C{-}W_D$")
    ax.set_title("(c) irrep trichotomy from the mined $M^*$\n"
                 "(leave-one-out channels; dot size = energy)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=200, bbox_inches="tight")
    print(f"saved -> {args.out}.pdf/.png and {args.out}_metrics.json")
    print(f"trichotomy means: "
          + ", ".join(f"{k}: {np.mean(v):+.3f}" for k, v in tri.items()))


if __name__ == "__main__":
    main()