"""
chordlib.py
-----------
Shared machinery for the Bach-chorale dihedral-symmetry analyses.

Contents
  * The T/I group (dihedral, order 24) acting simply transitively on the 24
    major/minor triads, via abstract elements g = (k, e):
        (k, 0) = T_k : pitch class x -> x + k     (transposition)
        (k, 1) = I_k : pitch class x -> k - x     (inversion)
  * Pair labelings of the 24x24 chord pairs at three resolutions:
        z12_labels  (48 classes)  transposition only
        d12_labels  (24 classes)  transposition + inversion  [g_i^{-1} g_j]
        conj_labels ( 9 classes)  class function / radial
  * Corpus statistics: windowed co-occurrence counts and the paper's
    normalized matrix M* = (P - pp)/((P + pp)/2), over an arbitrary chord
    vocabulary, with helpers to slice out the 24-triad block.
  * Reynolds (group) averaging, centering, and variance-explained R^2.

Token conventions (shared by all extractions):
    r        major triad, root r (0..11)      12 + r   minor triad
    24 + r   diminished      36 + r  augmented
    48 + r   dominant 7th    60 + r  minor 7th
    72 + r   half-dim 7th    84 + r  diminished 7th    96 + r  major 7th
"""
import numpy as np

N = 24
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TRIAD_NAMES = [n + "M" for n in NOTE] + [n + "m" for n in NOTE]
FAMILY_NAMES = ["maj", "min", "dim", "aug", "dom7", "min7", "hdim7", "dim7", "maj7"]

ELEMS = [(k, e) for e in (0, 1) for k in range(12)]
EIDX = {g: i for i, g in enumerate(ELEMS)}
# G_OF[token] = group element sending CM (=0) to that triad token
G_OF = [(k, 0) for k in range(12)] + [((k + 7) % 12, 1) for k in range(12)]

IRREPS = ["A1", "A2", "B1", "B2", "E1", "E2", "E3", "E4", "E5"]
IRREP_DIM = {"A1": 1, "A2": 1, "B1": 1, "B2": 1,
             "E1": 2, "E2": 2, "E3": 2, "E4": 2, "E5": 2}

# Empirically adapted interleaved-fifths 24-cycle (token order).
Z24_CYCLE = [0, 21, 7, 16, 2, 23, 9, 18, 4, 13, 11, 20,
             6, 15, 1, 22, 8, 17, 3, 12, 10, 19, 5, 14]


def gmul(a, b):
    """(k1,e1)(k2,e2) = (k1 + (-1)^e1 k2, e1 xor e2)."""
    k1, e1 = a
    k2, e2 = b
    return ((k1 + (1 - 2 * e1) * k2) % 12, e1 ^ e2)


def ginv(g):
    k, e = g
    return ((-(1 - 2 * e) * k) % 12, e)


def act(g, token):
    """Action of g on a triad token (I_n maps major r -> minor n-r-7)."""
    k, e = g
    q, r = divmod(token, 12)
    if e == 0:
        return q * 12 + (r + k) % 12
    return (1 - q) * 12 + (k - r - 7) % 12


def perm_of(g):
    """The permutation of the 24 triad tokens induced by g (as a tuple)."""
    return tuple(act(g, t) for t in range(N))


def perm_matrix(g):
    P = np.zeros((N, N))
    for t in range(N):
        P[act(g, t), t] = 1.0
    return P


def d12_labels():
    """label[i,j] = index of g_i^{-1} g_j  (24 classes; left/T-I action)."""
    lab = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            lab[i, j] = EIDX[gmul(ginv(G_OF[i]), G_OF[j])]
    return lab


def plr_labels():
    """label[i,j] = index of g_i g_j^{-1}  (24 classes; right/PLR action).
    The other simply-transitive dihedral action on the triads."""
    lab = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            lab[i, j] = EIDX[gmul(G_OF[i], ginv(G_OF[j]))]
    return lab


def z12_labels():
    """label[i,j] = (quality_i, quality_j, root_j - root_i)  (48 classes)."""
    ids = {}
    lab = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            key = (i // 12, j // 12, (j % 12 - i % 12) % 12)
            if key not in ids:
                ids[key] = len(ids)
            lab[i, j] = ids[key]
    return lab, len(ids)


def conj_labels():
    """label[i,j] = conjugacy class of g_i^{-1} g_j  (9 classes)."""
    def cls_of(g):
        k, e = g
        if e == 0:
            return ("T", min(k, (-k) % 12))
        return ("I", k % 2)

    ids = {}
    lab = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            c = cls_of(gmul(ginv(G_OF[i]), G_OF[j]))
            if c not in ids:
                ids[c] = len(ids)
            lab[i, j] = ids[c]
    return lab, len(ids)


def cooccurrence_counts(sequences, vocab_size, window=3):
    """Symmetric windowed pair counts over the given vocabulary."""
    C = np.zeros((vocab_size, vocab_size))
    for s in sequences:
        s = np.asarray(s)
        for d in range(1, window + 1):
            for i, j in zip(s[:-d], s[d:]):
                C[i, j] += 1.0
                C[j, i] += 1.0
    return C


def mstar(sequences, window=3, smoothing=0.5, min_count=20):
    """The paper's normalized co-occurrence matrix over the observed
    vocabulary.

    Rare non-triad tokens (row count < min_count) are dropped before
    normalization; the 24 triads are always kept. Returns (M, kept) where
    kept lists the token ids indexing M's rows/columns. Marginals are taken
    over the whole kept vocabulary, so with a full-vocabulary extraction the
    other chord families genuinely participate in the statistics.
    """
    V = max(max(s) for s in sequences) + 1
    counts = cooccurrence_counts(sequences, V, window)
    rowsum = counts.sum(axis=1)
    kept = [t for t in range(V) if t < N or rowsum[t] >= min_count]
    counts = counts[np.ix_(kept, kept)] + smoothing
    P = counts / counts.sum()
    p = P.sum(axis=1)
    pp = np.outer(p, p)
    M = (P - pp) / (0.5 * (P + pp))
    return 0.5 * (M + M.T), kept


def triad_block(M, kept):
    """Slice the 24-triad block out of a (possibly larger) M."""
    pos = {t: i for i, t in enumerate(kept)}
    idx = [pos[t] for t in range(N)]
    return M[np.ix_(idx, idx)]


def reynolds(M24):
    """Average of M over the diagonal D12 action: the exactly-invariant
    (Frobenius-closest invariant) matrix."""
    out = np.zeros_like(M24)
    for g in ELEMS:
        P = perm_matrix(g)
        out += P @ M24 @ P.T
    return out / len(ELEMS)


def center(M24):
    P = np.eye(N) - np.ones((N, N)) / N
    return P @ M24 @ P


def r2(M24, labels):
    """Variance explained by the labeling, over all matrix entries
    (including the diagonal)."""
    vals, labs = [], []
    for i in range(N):
        for j in range(N):
            vals.append(M24[i, j])
            labs.append(labels[i, j])
    vals, labs = np.array(vals), np.array(labs)
    grand = vals.mean()
    ss_tot = ((vals - grand) ** 2).sum()
    ss_between = sum((labs == c).sum() * (vals[labs == c].mean() - grand) ** 2
                     for c in np.unique(labs))
    return float(ss_between / ss_tot) if ss_tot > 0 else 0.0


def global_marginal_null(sequences, vocab_size=None, seed=0):
    """Redraw every token i.i.d. from the corpus marginal (kills all
    co-occurrence structure while keeping unigram frequencies)."""
    flat = np.concatenate([np.asarray(s) for s in sequences])
    V = (int(flat.max()) + 1) if vocab_size is None else vocab_size
    counts = np.bincount(flat, minlength=V).astype(float)
    p = counts / counts.sum()
    rng = np.random.default_rng(seed)
    return [rng.choice(V, size=len(s), p=p).tolist() for s in sequences]


def irrep_matrix(name, g):
    """rho(g) for the named irrep: 1x1 signs, or 2x2 rotations/reflections
    with rho(T_1) = rotation by h*30 degrees for E_h."""
    k, e = g
    if name == "A1":
        return np.array([[1.0]])
    if name == "A2":
        return np.array([[1.0 if e == 0 else -1.0]])
    if name == "B1":
        return np.array([[(-1.0) ** k]])
    if name == "B2":
        return np.array([[(-1.0) ** (k + e)]])
    h = int(name[1])
    th = h * np.pi / 6.0
    c, s = np.cos(k * th), np.sin(k * th)
    R = np.array([[c, -s], [s, c]])
    if e == 0:
        return R
    return R @ np.array([[1.0, 0.0], [0.0, -1.0]])


def isotypic_projectors():
    """P_rho = (d_rho/24) sum_g chi_rho(g) L(g); orthogonal projectors of
    rank d_rho^2 summing to the identity."""
    proj = {}
    for r, d in IRREP_DIM.items():
        P = np.zeros((N, N))
        for g in ELEMS:
            chi = np.trace(irrep_matrix(r, g))
            P += chi * perm_matrix(g)
        proj[r] = (d / N) * P
    return proj


def theory_modes(M24):
    """Peter-Weyl prediction from the group-averaged kernel of M24.

    Estimates C(g) as the class mean of M24 over pairs with g_i^{-1}g_j = g,
    forms C_hat(rho) = sum_g C(g) rho(g), and diagonalizes. Returns modes
    [(irrep, lam, V)] with V the (24 x d_rho) eigenfunction block
    psi(g) = sqrt(d_rho/24) [rho(g) u]_a, sorted by lam descending, with
    the trivial (A1) mode -- removed by centering -- excluded.
    """
    C = {g: float(np.mean([
        M24[i, act(gmul(G_OF[i], g), 0)] for i in range(N)
    ])) for g in ELEMS}
    modes = []
    for r in IRREPS:
        if r == "A1":
            continue
        d = IRREP_DIM[r]
        Ch = sum(C[g] * irrep_matrix(r, g) for g in ELEMS)
        Ch = 0.5 * (Ch + Ch.T)
        lam, U = np.linalg.eigh(Ch)
        for s in range(d):
            u = U[:, s]
            V = np.zeros((N, d))
            for i in range(N):
                V[i] = np.sqrt(d / N) * (irrep_matrix(r, G_OF[i]) @ u)
            modes.append((r, float(lam[s]), V))
    modes.sort(key=lambda m: -m[1])
    return modes


def embedding_modes(M24, k=6):
    """Top-k eigenpairs of the centered matrix: (lam, V) with V 24 x k.
    These are the model embeddings by the factorization theory
    (QWEM / Eckart-Young; coordinates are V * sqrt(lam))."""
    Mc = center(M24)
    Mc = 0.5 * (Mc + Mc.T)
    lam, V = np.linalg.eigh(Mc)
    order = np.argsort(-lam)[:k]
    return lam[order], V[:, order]


def embedding_coords(M24, k=6):
    lam, V = embedding_modes(M24, k)
    return V * np.sqrt(np.clip(lam, 0.0, None))


def procrustes(A, B):
    """Best orthogonal R with A ~ B R; returns (R, relative residual)."""
    U, _, Vt = np.linalg.svd(B.T @ A)
    R = U @ Vt
    return R, np.linalg.norm(A - B @ R) / max(np.linalg.norm(A), 1e-12)


def plane_numbers(E):
    """For a 24x2 plane: rotation angle under T_1, dets under T_1 and I_0,
    equivariance residuals, and the dihedral braid residual."""
    Rt, res_t = procrustes(perm_matrix((1, 0)) @ E, E)
    Ri, res_i = procrustes(perm_matrix((0, 1)) @ E, E)
    angle = abs(np.degrees(np.arctan2(Rt[1, 0], Rt[0, 0])))
    braid = np.linalg.norm(Ri @ Rt @ Ri - Rt.T) / np.sqrt(2)
    return dict(angle=angle, det_t=float(np.linalg.det(Rt)),
                det_i=float(np.linalg.det(Ri)),
                res_t=float(res_t), res_i=float(res_i), braid=float(braid))


def plane_purity(V2, projectors=None):
    """Mean isotypic energy of a 24x2 plane span in each irrep
    (chance level d_rho^2 / 24)."""
    if projectors is None:
        projectors = isotypic_projectors()
    # orthonormalize columns
    Q, _ = np.linalg.qr(V2)
    out = {}
    for r, P in projectors.items():
        out[r] = float(np.mean([np.linalg.norm(P @ Q[:, a]) ** 2
                                for a in range(Q.shape[1])]))
    return out


def subspace_overlap(A, B):
    """Mean squared cosine of principal angles between column spans."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(s ** 2))


def z24_labels():
    """Intervals on the empirically-adapted interleaved-fifths 24-cycle
    (the most charitable simply-transitive cyclic alternative)."""
    pos = {t: i for i, t in enumerate(Z24_CYCLE)}
    lab = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            lab[i, j] = (pos[j] - pos[i]) % 24
    return lab


def z12xz12_labels():
    """Independent transposition of majors and minors ('two decoupled
    circles'): same-quality blocks keep intervals, cross blocks get a
    single shared class per directed quality pair."""
    ids = {}
    lab = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i // 12 == j // 12:
                key = ("same", i // 12, (j % 12 - i % 12) % 12)
            else:
                key = ("cross", i // 12, j // 12)
            if key not in ids:
                ids[key] = len(ids)
            lab[i, j] = ids[key]
    return lab


def latent_z12_labels(offset=9):
    """One shared 12-site circle: major r and minor (r + offset) occupy the
    same latent site; kernel depends only on latent distance (12 classes).
    offset=9 pairs relative keys (C ~ Am)."""
    site = list(range(12)) + [(r - offset) % 12 for r in range(12)]
    lab = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            lab[i, j] = (site[j] - site[i]) % 12
    return lab


def class_average(M24, labels):
    """Average M over the labeling's classes: the closest labels-invariant
    matrix (class_average with d12_labels == reynolds)."""
    A = np.zeros_like(M24)
    for c in np.unique(labels):
        mask = labels == c
        A[mask] = M24[mask].mean()
    return A