# Celestial

Run everything from this directory after activating the repo environment
(see the top-level README). `skylib.py` is a shared library used a across the scripts. Do not run it directly
Paths in `skylib.py` are relative to here.

## Data

**Sky objects and LLM activations** come with this repo:

- `astro_objects.csv` — 188 named objects with RA/Dec
- `PCA128/{model}_pca128.npz` — per-object mean activations (top-128 PCA
  per layer)

Both are from Berdnikov & Liokumovich, *Sky sphere representation in language models*
([arXiv:2607.27092](https://arxiv.org/abs/2607.27092)). Default model
id is `mistrallarge123b`. `constellations.json` is only used for
genitive aliases in book tokenization (e.g. “Lyrae”).

**Astronomy books** Fetch the 24-book GITenberg corpus:

```bash
./fetch_astrobooks.sh
```

That writes `astrobooks/{gutenberg_id}.txt`. The four star guides used
in the main analyses are 20769, 36741, 68391, and 57091; the rest feed
the book-selection table.

## Main figure

```bash
python3 celestial_main_figure.py
```

Zonal kernel, degree-purity heatmap, and
Mistral Large 2 vs books 3D overlays. Writes `sky_kernel_overlay.pdf`/`.png`.
Optional `--sweep` also sweeps linear decodability from Mistral Large 2 across layers.

## Other plots and tables

```bash
python3 book_corpus_size_effect.py
python3 book_corpus_table.py
./run_all_batteries.sh
python3 text8_null.py
```

- `book_corpus_size_effect.py` — self-fit cosine vs corpus size
  (Poisson sims under the books’ zonal kernel). Writes
  `book_corpus_size_effect.pdf`/`.png`. 
- `book_corpus_table.py` — per-book \(D\), coherence, \(\lambda_1\),
  kernel corr. Writes `book_genre_table.md` and `.tex`.
- `run_all_batteries.sh` — degree-purity heatmap for every model in
  `PCA128/`. Writes `model_sky_heatmaps/sky_heatmap_{model}.pdf`/`.png`.
  For only one model, run: `python3 sky_battery.py --model mistrallarge123b`
- `text8_null.py` — negative control on `../months/text8.txt` (download
  that corpus first; see `months/README.md`). Prints coherence, \(\lambda_1\),
  \(D\), and kernel corr; no figure.

`sky_overlay_harmonics.py` is the standalone books | LLM overlay (same
as panel (c) of the main figure).
