"""
skylib.py
---------
Self-contained core for the sky (S^2 / Gelfand pair) analyses. The only
dependencies are numpy, scipy, matplotlib, and two data sources (paths
relative to the working directory; edit the constants below to relocate):

  OBJECTS_CSV  catalogue of 188 sky objects (names + RA/Dec);
  PCA_DIR      Decodable-sky PCA128/{model}_pca128.npz activations
               (Berdnikov & Liokumovich, arXiv:2607.27092);
  BOOKS_DIR    the GITenberg astronomy corpus (see fetch_astrobooks.sh);
  CONST_JSON constellation metadata (d3-celestial constellations.json),
             used only for genitive aliases ("Lyrae", "Orionis") in book
             tokenization.

Contents: sampled real spherical harmonics and weighted degree bases;
density weights and weighted-frame eigenproblems; the star-guide corpus
-> M* -> zonal kernel c(theta) pipeline (multi-book M* pools by summing
per-book co-occurrence matrices, with no cross-book windows); theory
matrices at arbitrary target positions; block comparisons with chance
nulls; the LLM activation loader; and the two-view 3D overlay plot.

Conventions (established in the analysis): density weights w = 1/rho
(on-sphere KDE, bandwidth HK) throughout; eigenproblems in the sqrt(w)
frame of the doubly-centered matrix; M* diagonals pinned to c(0) before
eigendecomposition (uniform spectral shift; prevents a mode-ordering
artifact from the self-pair smoothing floor); theory matrix = fitted
zonal kernel evaluated at the TARGET objects' positions (out-of-sample
extrapolation when the target is the LLM).
"""
import csv
import json
import re

import numpy as np
from scipy.special import sph_harm_y

# ---------------------------------------------------------------------------
# paths and constants
# ---------------------------------------------------------------------------
OBJECTS_CSV = "astro_objects.csv"
PCA_DIR = "PCA128"
BOOKS_DIR = "astrobooks"
CONST_JSON = "geo/constellations.json"

HK = 0.35            # on-sphere KDE bandwidth (radians) for density weights
HB = 0.08            # kernel-smoothing bandwidth in x = cos(theta)
WINDOW = 30          # co-occurrence window (tokens)
ALPHA = 0.25         # pair-count smoothing
MIN_COUNT = 5        # object count floor for kernel fits
LMAX = 4             # highest harmonic degree used in purity analyses
XQ = np.linspace(-0.999, 0.999, 160)

BOOKS = ["20769", "36741", "68391", "57091"]   # coherence-selected corpus
BOOK_NAMES = {"20769": "A Field Book of the Stars (Olcott 1907)",
              "36741": "Astronomy with an Opera-Glass (Serviss 1888)",
              "68391": "Round the Year with the Stars (Olcott 1912)",
              "57091": "Astronomy for Young Australians (Bonwick 1866)"}


# ---------------------------------------------------------------------------
# sampled spherical harmonics
# ---------------------------------------------------------------------------
def real_harmonics(lat_deg, lon_deg, lmax=LMAX):
    """Real spherical harmonics at the points, per degree:
    dict l -> (n, 2l+1) array."""
    theta = np.radians(90.0 - np.asarray(lat_deg))       # polar angle
    phi = np.radians(np.asarray(lon_deg) % 360.0)
    out = {}
    for l in range(lmax + 1):
        cols = [np.real(sph_harm_y(l, 0, theta, phi))]
        for m in range(1, l + 1):
            Y = sph_harm_y(l, m, theta, phi)
            cols.append(np.sqrt(2) * np.real(Y))
            cols.append(np.sqrt(2) * np.imag(Y))
        out[l] = np.column_stack(cols)
    return out


def weighted_degree_bases(Yl, w):
    """Orthonormalize the sampled harmonics in the w-weighted inner
    product, by increasing degree (lower degrees stay pure; higher degrees
    are orthogonalized against everything below)."""
    sw = np.sqrt(w)[:, None]
    done = np.zeros((len(w), 0))
    bases = {}
    for l in sorted(Yl):
        B = sw * Yl[l]
        B = B - done @ (done.T @ B)
        Q, _ = np.linalg.qr(B)
        bases[l] = Q                       # orthonormal in sqrt(w)-space
        done = np.column_stack([done, Q])
    return bases


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def unit_vectors(lat_deg, lon_deg):
    lr, gr = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(lr) * np.cos(gr), np.cos(lr) * np.sin(gr),
                     np.sin(lr)], 1)


def density_weights(xyz):
    """1/rho weights (sum n) and the cos-angle matrix."""
    cosT = np.clip(xyz @ xyz.T, -1, 1)
    rho = np.exp(-np.arccos(cosT) ** 2 / (2 * HK ** 2)).sum(1)
    w = 1.0 / rho
    return w / w.sum() * len(w), cosT


def weighted_modes(M, sw, k=None):
    """Eigenmodes of the doubly-centered matrix in the sqrt(w) frame,
    sorted by |eigenvalue|. Centering uses the weighted projector
    P_w = I - 1 w^T / sum(w) (constants removed in the weighted inner
    product), so the ell=0 direction sqrt(w) is annihilated exactly."""
    n = M.shape[0]
    w = (sw ** 2).ravel()
    Pw = np.eye(n) - np.outer(np.ones(n), w) / w.sum()
    Mw = (sw * (Pw @ M @ Pw.T)) * sw.T
    lam, U = np.linalg.eigh(0.5 * (Mw + Mw.T))
    o = np.argsort(-np.abs(lam))
    if k:
        o = o[:k]
    return lam[o], U[:, o]


# ---------------------------------------------------------------------------
# objects, corpus, M*, zonal kernel
# ---------------------------------------------------------------------------
def load_objects():
    """The 188 sky objects (names + coordinates) from the Decodable-sky
    repo, plus genitive aliases for constellations from CONST_JSON.
    Returns (names, lat_deg, lon_deg, aliases: name -> [lowercase forms])."""
    rows = list(csv.DictReader(open(OBJECTS_CSV)))
    names = [r["name"] for r in rows]
    lat = np.array([float(r["a_dec_deg"]) for r in rows])
    lon = np.array([float(r["l_ra_deg"]) for r in rows])
    gen = {}
    try:
        d = json.load(open(CONST_JSON))
        for f in d["features"]:
            gen[f["properties"]["name"].lower()] = \
                f["properties"]["gen"].lower()
    except Exception:
        pass
    aliases = {nm: [nm.lower()] + ([gen[nm.lower()]]
                                   if nm.lower() in gen else [])
               for nm in names}
    return names, lat, lon, aliases


def tokenize_with_objects(text, names, aliases):
    """Lowercase; map every alias (incl. genitives) of an object to one
    canonical token; multiword aliases joined first, longest first.
    Returns (tokens, canon: name -> canonical token)."""
    text = text.lower()
    canon = {nm: nm.lower().replace(" ", "_") for nm in names}
    pairs = [(al, canon[nm]) for nm in names for al in aliases[nm]]
    for al, tok in sorted(pairs, key=lambda p: -len(p[0])):
        if al != tok:
            text = text.replace(al, " " + tok + " ")
    return re.findall(r"[a-z_]+", text), canon


def book_tokens(book_id, names, aliases):
    text = open(f"{BOOKS_DIR}/{book_id}.txt", encoding="utf-8",
                errors="ignore").read()
    return tokenize_with_objects(text, names, aliases)


def corpus_stream(books=BOOKS):
    """Per-book token streams of the selected books (not concatenated).

    Returns (names, lat, lon, streams, canon) where streams is a list of
    token lists, one per book. Pass streams to corpus_mstar to pool by
    summing per-book co-occurrence counts (no cross-book window pairs).
    """
    names, lat, lon, aliases = load_objects()
    streams, canon = [], None
    for bid in books:
        toks, canon = book_tokens(bid, names, aliases)
        streams.append(toks)
    return names, lat, lon, streams, canon


def _as_token_streams(acc):
    """Normalize acc to a list of token streams.

    A flat token list (elements are str) is one stream; a list of token
    lists is already multi-stream.
    """
    if not acc:
        return []
    if isinstance(acc[0], str):
        return [acc]
    return [list(s) for s in acc]


def corpus_mstar(names, acc, canon, min_count=MIN_COUNT, keep=None):
    """Windowed M* over the corpus's well-counted objects.

    acc may be a single token list or a list of token lists (e.g. from
    corpus_stream). Multiple streams are pooled by summing per-stream
    co-occurrence matrices and unigram counts; windows never cross
    stream boundaries. Returns (keep indices, M, unigram p over kept,
    n_tokens).
    """
    streams = _as_token_streams(acc)
    tok_of = {canon[nm]: i for i, nm in enumerate(names)}
    counts = np.zeros(len(names))
    n_tok = 0
    parsed = []
    for stream in streams:
        n_tok += len(stream)
        pos, labs = [], []
        for i, t in enumerate(stream):
            if t in tok_of:
                pos.append(i)
                labs.append(tok_of[t])
                counts[tok_of[t]] += 1
        parsed.append((pos, labs))
    if keep is None:
        keep = [i for i in range(len(names)) if counts[i] >= min_count]
    idx = {i: k for k, i in enumerate(keep)}
    nb = len(keep)
    W = np.zeros((nb, nb))
    for pos, labs in parsed:
        for a in range(len(pos)):
            if labs[a] not in idx:
                continue
            b = a + 1
            while b < len(pos) and pos[b] - pos[a] <= WINDOW:
                if labs[b] in idx and labs[b] != labs[a]:
                    W[idx[labs[a]], idx[labs[b]]] += 1
                    W[idx[labs[b]], idx[labs[a]]] += 1
                b += 1
    W += ALPHA
    P = W / (2.0 * WINDOW * max(n_tok, 1))
    p = counts[keep] / max(n_tok, 1)
    M = (P - np.outer(p, p)) / (0.5 * (P + np.outer(p, p)))
    return keep, 0.5 * (M + M.T), p, n_tok


def estimate_c(x_pairs, m_pairs, w_pairs, xq=XQ, hb=HB):
    """Weighted Nadaraya-Watson estimate of c(x) = E[M | cos theta = x]."""
    K = np.exp(-((x_pairs[None, :] - xq[:, None]) ** 2)
               / (2 * hb ** 2)) * w_pairs[None, :]
    return (K * m_pairs[None, :]).sum(1) / (K.sum(1) + 1e-12)


def fit_zonal(M, cosT, w):
    """Density-weighted zonal kernel from M's off-diagonal pairs."""
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    return estimate_c(cosT[iu], M[iu], w[iu[0]] * w[iu[1]])


def theory_matrix(c_fit, cosT):
    """Zonal kernel evaluated at target positions; diagonal = c(0)."""
    C = np.interp(cosT, XQ, c_fit)
    np.fill_diagonal(C, np.interp(1.0, XQ, c_fit))
    return C


def pin_diagonal(M, c_fit):
    M = M.copy()
    np.fill_diagonal(M, np.interp(1.0, XQ, c_fit))
    return M


# ---------------------------------------------------------------------------
# block comparison
# ---------------------------------------------------------------------------
def _pool_cols(k):
    """Normalize a pool spec: int -> prefix range, else a column list."""
    if isinstance(k, (int, np.integer)):
        return list(range(int(k)))
    return list(k)


def block_compare(U_th, lam_th, sl, E_emp, U_emp, sw, k_emp=12,
                  match="canonical", scale=True):
    """Theory eigen-block (columns sl) vs the empirical modes.

    match="canonical": locate the closest subspace inside the empirical
      pool via canonical angles (use when a documented competitor
      direction displaces the block from its nominal slot, e.g. the
      LLM's object-type direction, their Appendix F). k_emp is the
      pool: an int (top-k prefix) or an explicit list of mode columns
      (degree-specific pools, e.g. dipole in modes 0-3).
    match="fixed": compare against the empirical modes in the SAME slice
      sl, no selection (the strict reading; use e.g. for books-internal).
    scale: if True (default), apply one global Frobenius scale so
      ||B||=||A|| before Procrustes; if False, keep theory amplitudes.

    Returns (canonical cos^2, per-object cos mean, median, A, B_aligned):
    A = empirical coordinates in the matched frame; B_aligned = theory
    block coordinates after one rotation (+ optional global scale)."""
    n = U_th.shape[0]
    Tb = U_th[:, sl]
    if match == "fixed":
        Qe, _ = np.linalg.qr(sw * U_emp[:, sl])
        s = np.linalg.svd(Tb.T @ Qe, compute_uv=False)
        canon2 = float(np.mean(s[:Tb.shape[1]] ** 2))
        A = E_emp[:, sl]
    else:
        cols = _pool_cols(k_emp)
        Qe, _ = np.linalg.qr(sw * U_emp[:, cols])
        _, s, Vt = np.linalg.svd(Tb.T @ Qe)
        canon2 = float(np.mean(s[:Tb.shape[1]] ** 2))
        dirs = Qe @ Vt[:Tb.shape[1]].T
        A = (sw * E_emp[:, cols]) @ (Qe.T @ dirs)
        A = A / sw
    B = (Tb / sw) * np.sqrt(np.abs(lam_th[sl]))
    if scale:
        B = B * (np.linalg.norm(A) / max(np.linalg.norm(B), 1e-12))
    U2, _, V2 = np.linalg.svd(B.T @ A)
    Ba = B @ (U2 @ V2)
    cos = [A[i] @ Ba[i] / max(np.linalg.norm(A[i]) *
                              np.linalg.norm(Ba[i]), 1e-12)
           for i in range(n)]
    return canon2, float(np.mean(cos)), float(np.median(cos)), A, Ba


def cosine_null(A, Ba, nperm=200, seed=0):
    """Label-shuffle control for the per-object cosine: permute which
    theory point is assigned to which object, refit the one global scale
    and one Procrustes rotation, and record the mean per-object cosine.
    Prices in the alignment freedom (Procrustes soaks ~sqrt(c/n) of
    noise) plus any anisotropy of the configurations.
    Returns (null mean, null 95th percentile)."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    out = []
    for _ in range(nperm):
        Bp = Ba[rng.permutation(n)]
        Bp = Bp * (np.linalg.norm(A) / max(np.linalg.norm(Bp), 1e-12))
        U2, _, V2 = np.linalg.svd(Bp.T @ A)
        Bq = Bp @ (U2 @ V2)
        cos = [A[i] @ Bq[i] / max(np.linalg.norm(A[i]) *
                                  np.linalg.norm(Bq[i]), 1e-12)
               for i in range(n)]
        out.append(np.mean(cos))
    return float(np.mean(out)), float(np.quantile(out, 0.95))


def block_null(U_emp, sw, dim, k_emp=12, nperm=60, seed=0):
    """Chance canonical cos^2 of a random dim-block vs the empirical
    pool (int prefix, or an explicit list of mode columns)."""
    rng = np.random.default_rng(seed)
    n = U_emp.shape[0]
    Qe, _ = np.linalg.qr(sw * U_emp[:, _pool_cols(k_emp)])
    out = []
    for _ in range(nperm):
        R, _ = np.linalg.qr(rng.standard_normal((n, dim)))
        s = np.linalg.svd(R.T @ Qe, compute_uv=False)
        out.append(np.mean(s[:dim] ** 2))
    return float(np.mean(out))


# ---------------------------------------------------------------------------
# LLM activations
# ---------------------------------------------------------------------------
def load_activations(model):
    """Per-object mean activations from the Decodable-sky npz.
    Returns (X: layers x n_obj x K, lat, lon, xyz, names)."""
    z = np.load(f"{PCA_DIR}/{model}_pca128.npz", allow_pickle=True)
    pca = z["pca"].astype(np.float32)
    obj = z["obj_ids"]
    Y = z["Yunit"].astype(np.float64)
    names = z["names"]
    n_obj = Y.shape[0]
    X = np.zeros((pca.shape[1], n_obj, pca.shape[2]))
    for o in range(n_obj):
        X[:, o] = pca[obj == o].mean(0)
    lat = np.degrees(np.arcsin(np.clip(Y[:, 0], -1, 1)))
    lon = np.degrees(np.arctan2(Y[:, 1], Y[:, 2])) % 360.0
    return X, lat, lon, unit_vectors(lat, lon), names


def llm_modes(model="mistrallarge123b", layer=72, k=16):
    """Top-k Gram eigenmodes of the object-mean activations at a layer."""
    X, lat, lon, xyz, names = load_activations(model)
    Xc = X[layer] - X[layer].mean(0, keepdims=True)
    G = Xc @ Xc.T
    lam, U = np.linalg.eigh(0.5 * (G + G.T))
    o = np.argsort(-np.abs(lam))[:k]
    return lat, lon, xyz, lam[o], U[:, o], names


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def overlay_3d(A, Ba, lon, title, path, labels=None):
    """Two-view 3D overlay: empirical (filled, colored by RA) vs theory
    (open circles), gray connectors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = A.shape[0]
    fig = plt.figure(figsize=(13, 6))
    for p, (elev, azim) in enumerate([(18, -60), (18, 120)]):
        ax = fig.add_subplot(1, 2, p + 1, projection="3d")
        for q in range(n):
            ax.plot([Ba[q, 0], A[q, 0]], [Ba[q, 1], A[q, 1]],
                    [Ba[q, 2], A[q, 2]], color="gray", lw=0.5, alpha=0.55)
        sc = ax.scatter(A[:, 0], A[:, 1], A[:, 2], c=lon, cmap="hsv", s=28)
        ax.scatter(Ba[:, 0], Ba[:, 1], Ba[:, 2], facecolors="none",
                   edgecolors="k", s=40, linewidths=0.65)
        if labels is not None:
            for q in range(n):
                ax.text(A[q, 0], A[q, 1], A[q, 2], str(labels[q])[:9],
                        fontsize=5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    fig.suptitle(title)
    fig.colorbar(sc, ax=fig.axes, shrink=0.5, label="true RA (deg)")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return path