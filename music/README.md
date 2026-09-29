# Music


## Get the chords

We must first assemble our corpus. Before running any other script `extract_chords.py` pulls the Bach chorales from music21. 
Each score is chordified.  Downstream scripts
default to the **full** vocabulary, namely

```bash
python3 extract_chords.py --vocab full
```

This writes `chord_sequences_full.json`, where the corpus considers all chords, not just the triads. Using `--vocab triads` instead
forms `chord_sequences.json` which only contains the 24 triads.

## Bach analyses

These all read `chord_sequences_full.json` by default (override with `--sequences`):

```bash
python3 heatmap_theory.py
python3 window_sweep.py
python3 symmetry_model_comparison.py
```

- `heatmap_theory.py` — raw \(M^*\), group-averaged \(M_\mathrm{sym}\), and the
  first theory-overlay planes. Writes `heatmap_theory.pdf`/`.png`.
  `--place` opens a label editor; saved positions go to a sidecar JSON.
- `window_sweep.py` — D12 homogeneity \(R^2\), irrep purities, and
  eigenvalues vs co-occurrence window. Writes `window_sweep.pdf`/`.png`.
- `symmetry_model_comparison.py` — nested Z12 / D12 / conjugacy
  homogeneity \(R^2\), plus full-fit and held-out overlap/cosine for each
  candidate symmetry. Writes `model_comparison.pdf`/`.png` and
  `model_comparison_metrics.json`.

`chordlib.py` is the shared library (tokens, \(M^*\), labels, irreps).
Do not run it directly.

## ChordBERT

First cache layer-wise Grams (Hugging Face
`StravynDynamics/ChordBert`; needs the Bach JSON):

```bash
python3 chordbert_dihedral.py
```

Prints per-layer D12 \(R^2\) and E5 alignment, writes
`chordbert_grams.npz`, and the E5 isotypic-purity heatmap
`chordbert_e5_purity.pdf`/`.png`. Then:

```bash
python3 heatmap_chordbert.py
```

Same layout as `heatmap_theory.py`, with a ChordBERT E5 panel (default
layer 3). Writes `heatmap_chordbert.pdf`/`.png` and
`fig_chordbert_l3_irreps.pdf`/`.png`. Reuses
`label_positions_heatmap_chordbert.json` unless you pass `--place`.
This is the main music figure in the paper.
