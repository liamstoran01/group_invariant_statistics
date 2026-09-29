"""
chordbert_dihedral.py
---------------------
Probe ChordBERT (StravynDynamics/ChordBert, DeBERTa-v2 MLM over
Chordonomicon) for D12 structure in its layer-wise activations, using the
same metrics as the paper's Bach analysis (via chordlib).

Pipeline, per layer L:
  1. Feed chord progressions; mean-pool the hidden state at every
     occurrence of each of the 24 major/minor triads -> X_L in R^{24 x h}.
  2. Doubly center the Gram: M_L = P X_L X_L^T P (RMS-normalized diagonal),
     the same M* proxy as the sky section.
  3. Structural invariance: D12 homogeneity R^2 of M_L (cl.r2 against
     cl.d12_labels), Reynolds residual ||M - M_sym|| / ||M||.
  4. Modes: top-3 empirical planes vs theory modes of the symmetrized
     kernel; isotypic energies (subspace overlap), per-chord cosines
     after Procrustes; angular step/semitone of the leading plane
     (150 deg <-> fifths / E5).
  5. Cache activations + per-layer Grams to --save_npz
     (default chordbert_grams.npz) for heatmap_chordbert.py.
  6. E5 isotypic-purity heatmap of empirical modes across layers
     (--purity_fig, default chordbert_e5_purity.png/.pdf).

Contexts: either your Bach JSON (--sequences chord_sequences_full.json)
or a plain text file with one space-separated progression per line
(e.g. dumped from Chordonomicon). Defaults to Bach if present.

Usage example:
  python3 chordbert_dihedral.py --sequences chord_sequences_full.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

import chordlib as cl

MODEL_ID = "StravynDynamics/ChordBert"

# Candidate token spellings per triad, tried in order against the vocab.
MAJ_CANDS = ["{n}", "{n}maj", "{n}:maj", "{n}M"]
MIN_CANDS = ["{n}m", "{n}min", "{n}:min"]
ENHARM = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}

def resolve_triad_tokens(tokenizer):
    """Map the 24 chordlib triads to single vocab ids (None if absent)."""
    vocab = tokenizer.get_vocab()

    def find(note, cands):
        names = [note] + ([ENHARM[note]] if note in ENHARM else [])
        for n in names:
            for c in cands:
                tok = c.format(n=n)
                for probe in (tok, tok.lower()):
                    if probe in vocab:
                        return probe
        return None

    toks = ([find(n, MAJ_CANDS) for n in cl.NOTE] +
            [find(n, MIN_CANDS) for n in cl.NOTE])
    names = cl.TRIAD_NAMES if hasattr(cl, "TRIAD_NAMES") else \
        [n + "M" for n in cl.NOTE] + [n + "m" for n in cl.NOTE]
    for name, tok in zip(names, toks):
        print(f"  {name:5s} -> {tok}")
    missing = [names[i] for i, t in enumerate(toks) if t is None]
    if missing:
        sys.exit(f"unresolved triads {missing}; inspect tokenizer vocab "
                 f"and extend MAJ_CANDS/MIN_CANDS.")
    return toks


def load_progressions(args, triad_names):
    """Return list of progressions, each a list of triad indices (0..23).

    Bach JSON (chord_sequences_full.json) stores chordlib integer tokens
    (0..11 major, 12..23 minor, higher = other chord types). Plain .txt
    files use space-separated chordlib names (CM, Am, ...) or ChordBERT
    spellings resolvable via triad_names.
    """
    name_to_idx = {n: i for i, n in enumerate(triad_names)}
    # Also accept ChordBERT vocab spellings if passed through later.
    if args.sequences.endswith(".json"):
        seqs = json.load(open(args.sequences))
        out = []
        for s in seqs:
            idx = []
            for c in s:
                if isinstance(c, int):
                    if 0 <= c < cl.N:
                        idx.append(c)
                elif c in name_to_idx:
                    idx.append(name_to_idx[c])
            if len(idx) >= args.min_len:
                out.append(idx)
        return out
    out = []
    for line in open(args.sequences):
        idx = [name_to_idx[c] for c in line.split() if c in name_to_idx]
        if len(idx) >= args.min_len:
            out.append(idx)
    return out


@torch.no_grad()
def pooled_activations(model, tokenizer, progressions, triad_toks, args):
    """Mean hidden state per triad per layer.
    Returns X: (n_layers+1, 24, hidden); counts: (24,)."""
    n_layers = model.config.num_hidden_layers
    hid = model.config.hidden_size
    X = np.zeros((n_layers + 1, cl.N, hid))
    counts = np.zeros(cl.N)
    ids_of = [tokenizer.convert_tokens_to_ids(t) for t in triad_toks]
    id_to_triad = {tid: i for i, tid in enumerate(ids_of)}
    for start in range(0, len(progressions), args.batch):
        batch = progressions[start:start + args.batch]
        texts = [" ".join(triad_toks[i] for i in prog) for prog in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=args.max_len)
        out = model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states)          # (L+1, B, T, h)
        for b in range(len(batch)):
            for pos, tid in enumerate(enc["input_ids"][b].tolist()):
                tri = id_to_triad.get(tid)
                if tri is None or enc["attention_mask"][b, pos] == 0:
                    continue
                X[:, tri, :] += hs[:, b, pos, :].numpy()
                counts[tri] += 1
        if (start // args.batch) % 20 == 0:
            print(f"  {start + len(batch)}/{len(progressions)} progressions")
    if (counts == 0).any():
        bad = np.where(counts == 0)[0]
        sys.exit(f"no occurrences pooled for triads {bad}; need richer "
                 f"contexts.")
    X /= counts[None, :, None]
    return X, counts


def gram(Xl):
    """Doubly centered, RMS-normalized Gram (the sky-section M* proxy)."""
    P = np.eye(cl.N) - np.ones((cl.N, cl.N)) / cl.N
    M = P @ Xl @ Xl.T @ P
    rms = np.sqrt(np.mean(np.diag(M) ** 2))
    return M / max(rms, 1e-12)


def leading_step_deg(E):
    """Median |angular step| per semitone of majors in a 2-dim plane."""
    ang = np.unwrap(np.arctan2(E[:12, 1], E[:12, 0]))
    steps = np.degrees(np.diff(ang))
    return np.median(np.abs(steps))


def analyze_layer(M, k_planes=3, e5_modes=None):
    row = {}
    row["r2_d12"] = cl.r2(M, cl.d12_labels())
    M_sym = cl.reynolds(M)
    row["resid"] = np.linalg.norm(M - M_sym) / np.linalg.norm(M)
    # Prefer 2D (E-type) theory modes so Emp/Th widths match for Procrustes.
    # A 1D irrep in the top-k would make Th 5-D while Emp is 6-D.
    modes = [m for m in cl.theory_modes(M_sym) if m[2].shape[1] == 2][:k_planes]
    if len(modes) < k_planes:
        modes = cl.theory_modes(M_sym)[:k_planes]
    cols = []
    for r, lam, V in modes:
        for a in range(V.shape[1]):
            cols.append(np.sqrt(max(lam, 0)) * V[:, a])
    Th = np.column_stack(cols)
    k = Th.shape[1]
    lam_e, V_e = cl.embedding_modes(M, k=k)
    Emp = V_e * np.sqrt(np.clip(lam_e, 0, None))
    R, _ = cl.procrustes(Emp, Th)
    Th = Th @ R
    planes = []
    col = 0
    for r, lam, V in modes:
        d = V.shape[1]
        ov = cl.subspace_overlap(V_e[:, col:col + d], V)
        E, T = Emp[:, col:col + d], Th[:, col:col + d]
        cos = np.mean([E[i] @ T[i] /
                       max(np.linalg.norm(E[i]) * np.linalg.norm(T[i]),
                           1e-12) for i in range(cl.N)])
        planes.append((r, lam, ov, cos))
        col += d
    row["planes"] = planes
    row["step_deg"] = leading_step_deg(Emp[:, :2])

    # ---- E5 plane by SUBSPACE alignment (sky-style), not per-chord ----
    # e5_modes: 1-indexed empirical mode numbers (heatmap labels) to search
    # within. Exactly 2 modes -> use that plane directly (no alignment);
    # more -> SVD of V_sel^T V_th picks the best-aligned 2-dim subspace
    # (canonical directions), the stage-1 locate step of the sky pipeline.
    if e5_modes is None:
        e5_modes = [1, 2, 3, 4, 5]
    sel = [m - 1 for m in e5_modes]
    lam_a, V_a = cl.embedding_modes(M, k=max(sel) + 1)
    V_sel = V_a[:, sel]
    W_sel = V_sel * np.sqrt(np.clip(lam_a[sel], 0, None))
    e5 = next((m for m in cl.theory_modes(M_sym) if m[0] == "E5"), None)
    if e5 is None:
        e5 = cl.theory_modes(M_sym)[0]
    _, lam_th, V_th = e5
    A = V_sel.T @ V_th                                # |sel| x 2
    U, S, _ = np.linalg.svd(A)
    row["canon_cos"] = S[:2]
    row["e5_modes"] = list(e5_modes)
    if len(sel) == 2:
        E_emp = W_sel                                 # the plane as given
    else:
        E_emp = W_sel @ U[:, :2]                      # best-aligned plane
    Th5 = np.sqrt(max(lam_th, 0)) * V_th              # theory fifths ring
    R2, _ = cl.procrustes(E_emp, Th5)                 # in-plane gauge only
    T5 = Th5 @ R2
    cos5 = [E_emp[i] @ T5[i] /
            max(np.linalg.norm(E_emp[i]) * np.linalg.norm(T5[i]), 1e-12)
            for i in range(cl.N)]
    row["E5_cos_mean"] = float(np.mean(cos5))
    row["E5_cos_median"] = float(np.median(cos5))
    row["E5_cos_min"] = float(np.min(cos5))
    row["E5_emp"] = E_emp
    row["E5_th"] = T5
    row["E5_step"] = leading_step_deg(E_emp)
    return row


def d12_perm(k, inv):
    """Left action of g=(k,inv) on chord indices (majors 0-11 = T_m,
    minors 12-23 = I_m, base chord CM = identity)."""
    p = np.zeros(cl.N, dtype=int)
    for m in range(12):
        if not inv:
            p[m] = (k + m) % 12                       # T_k T_m = T_{k+m}
            p[12 + m] = 12 + (k + m) % 12             # T_k I_m = I_{k+m}
        else:
            p[m] = 12 + (k - m) % 12                  # I_k T_m = I_{k-m}
            p[12 + m] = (k - m) % 12                  # I_k I_m = T_{k-m}
    return p


def isotypic_projector(h):
    """Orthogonal projector onto the E_h isotypic component (dim 4) of the
    regular action on the 24 triads: Pi = (d/|G|) sum_g chi(g) pi(g);
    reflections have chi = 0, so only rotations contribute."""
    Pi = np.zeros((cl.N, cl.N))
    for k in range(12):
        p = d12_perm(k, 0)
        Pmat = np.zeros((cl.N, cl.N))
        Pmat[p, np.arange(cl.N)] = 1.0
        Pi += 2 * np.cos(2 * np.pi * h * k / 12) * Pmat
    Pi *= 2.0 / 24.0
    return Pi


def e5_purity_figure(grams, out_path, k_modes=8, h=5):
    """Single heatmap: layers x top-k empirical modes, cell = energy of the
    mode in the E_h isotypic component, ||Pi_h v||^2. Chance = 4/23."""
    import matplotlib.pyplot as plt

    Pi = isotypic_projector(h)
    assert np.allclose(Pi @ Pi, Pi, atol=1e-8), "not a projector"
    assert abs(np.trace(Pi) - 4) < 1e-6, "isotypic dim != 4"
    P = np.zeros((len(grams), k_modes))
    for L, M in enumerate(grams):
        lam, V = np.linalg.eigh(M)
        idx = np.argsort(-lam)[:k_modes]
        for j, i in enumerate(idx):
            P[L, j] = np.linalg.norm(Pi @ V[:, i]) ** 2
    fig, ax = plt.subplots(figsize=(1.05 * k_modes, 0.85 * len(grams) + 1.4))
    im = ax.imshow(P, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for L in range(P.shape[0]):
        for j in range(k_modes):
            ax.text(j, L, f"{P[L, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="w" if P[L, j] < 0.6 else "k")
    ax.set_xticks(range(k_modes),
                  [f"{j + 1}" for j in range(k_modes)], fontsize=9)
    ax.set_yticks(range(P.shape[0]),
                  [f"{L}" + (" (emb)" if L == 0 else "")
                   for L in range(P.shape[0])], fontsize=9)
    ax.set_xlabel("empirical mode (by eigenvalue)", fontsize=10)
    ax.set_ylabel("layer", fontsize=10)
    ax.set_title(f"$E_{h}$ isotypic purity of empirical modes "
                 f"(chance $= 4/23 \\approx {4/23:.2f}$)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    if out_path.lower().endswith(".png"):
        fig.savefig(out_path[:-4] + ".pdf", bbox_inches="tight")
    print(f"purity heatmap -> {out_path}")


def parse_mode_spec(spec):
    """Parse "0:1,4,5;3:1,2" -> {0: [1, 4, 5], 3: [1, 2]} (1-indexed)."""
    out = {}
    if not spec:
        return out
    for part in spec.split(";"):
        if not part.strip():
            continue
        layer, modes = part.split(":")
        out[int(layer)] = [int(m) for m in modes.split(",")]
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="chord_sequences_full.json",
                    help=".json (Bach, chordlib names) or .txt progressions")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--min_len", type=int, default=4)
    ap.add_argument("--singles", action="store_true",
                    help="ignore --sequences; feed each of the 24 triads "
                         "alone (one chord per input, 24 inputs total)")
    ap.add_argument("--save_npz", default="chordbert_grams.npz")
    ap.add_argument("--purity_fig", default="chordbert_e5_purity.png")
    ap.add_argument("--e5_modes", default="",
                    help='per-layer modes for the E5 plane, 1-indexed as '
                         'on the heatmap, e.g. "0:1,4,5;3:1,2". Exactly 2 '
                         'modes = that plane directly; more = best-aligned '
                         '2-dim subspace. Unlisted layers use 1-5.')
    args = ap.parse_args()
    if args.singles:
        for attr in ("save_npz", "purity_fig"):
            val = getattr(args, attr)
            root, ext = os.path.splitext(val)
            setattr(args, attr, root + "_singles" + ext)

    print(f"loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.eval()

    print("resolving triad tokens:")
    triad_toks = resolve_triad_tokens(tokenizer)
    triad_names = ([n + "M" for n in cl.NOTE] +
                   [n + "m" for n in cl.NOTE])
    if args.singles:
        progs = [[i] for i in range(cl.N)]
        print("singles mode: 24 one-chord inputs (no context)")
    else:
        progs = load_progressions(args, triad_names)
        print(f"{len(progs)} progressions with >= {args.min_len} mapped "
              f"triads")

    X, counts = pooled_activations(model, tokenizer, progs, triad_toks,
                                   args)
    print("occurrences per triad:", counts.astype(int))

    mode_spec = parse_mode_spec(args.e5_modes)
    grams, rows = [], []
    print(f"\n{'layer':>5} {'R2(D12)':>8} {'resid':>6} {'step/semi':>9}  "
          f"planes (irrep, lam, overlap, cos)")
    for L in range(X.shape[0]):
        M = gram(X[L])
        grams.append(M)
        row = analyze_layer(M, e5_modes=mode_spec.get(L))
        rows.append(row)
        ps = "  ".join(f"[{r} {lam:+.2f} ov {ov:.2f} cos {c:.2f}]"
                       for r, lam, ov, c in row["planes"])
        print(f"{L:>5} {row['r2_d12']:>8.3f} {row['resid']:>6.3f} "
              f"{row['step_deg']:>8.1f}\u00b0  "
              f"E5[canon {row['canon_cos'][0]:.2f}/{row['canon_cos'][1]:.2f} "
              f"cos {row['E5_cos_mean']:.2f} med {row['E5_cos_median']:.2f} "
              f"min {row['E5_cos_min']:.2f}]  {ps}")

    e5_purity_figure(grams, args.purity_fig)

    np.savez(args.save_npz, X=X, grams=np.array(grams), counts=counts)
    print(f"\nsaved activations + grams -> {args.save_npz}")
    print("layer 0 = static input embeddings (closest to the paper's "
          "word-embedding theory); later layers = contextual.")


if __name__ == "__main__":
    main()