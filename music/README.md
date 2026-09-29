# Music


## Get the chords

We must first assemble our corpus. Before running any other script `extract_chords.py` pulls the Bach chorales from music21. 
Each score is chordified.  Downstream scripts
default to the **full** vocabulary, namely activate the environment repo and run

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
- `window_sweep.py` — rotation angles, irrep purities, and
  eigenvalues vs co-occurrence window. Writes `window_sweep.pdf`/`.png`.
- `symmetry_model_comparison.py` — for candidate symmetries, computes
  homogeneity \(R^2\) for \(M^*\) along as held-out cosine sim when fitted
  on half the data or some fraction of it. Writes `model_comparison.pdf`/`.png` and
  `model_comparison_metrics.json`.

`chordlib.py` is the shared library (tokens, \(M^*\), labels, irreps).
Do not run it directly.

## ChordBERT

First cache layer-wise Grams (Hugging Face
`StravynDynamics/ChordBert`; needs the Bach JSON)
by running:

```bash
python3 chordbert_dihedral.py
```

 This writes
`chordbert_grams.npz`, and generates the isotypic-purity heatmap
`chordbert_e5_purity.pdf`/`.png`. Then:

```bash
python3 heatmap_chordbert.py
```

Same layout as `heatmap_theory.py`, with a ChordBERT E5 panel (default
layer 3). Writes `heatmap_chordbert.pdf`/`.png` and
`fig_chordbert_l3_irreps.pdf`/`.png`. Uses
`label_positions_heatmap_chordbert.json` for chord labels in plot unless you pass `--place`.
This is the main music figure in the paper.
