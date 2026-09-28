# Months

## Get text8

From this directory, download Matt Mahoney's text8 corpus (first 10^8 bytes
of cleaned English Wikipedia), unzip it, and rename the extracted file to
`text8.txt` (that is the default `--corpus` path):

```bash
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip && mv text8 text8.txt
```

## Run

Activate the repo environment, then run `months.py` from this directory:

```bash
source ../.venv/bin/activate
python3 months.py
```

Optional flags: `--corpus text8.txt` (default) and `--out months.pdf`.

## Output

The script prints corpus diagnostics (month-sense `"may"` counts, `M*`
range, circular kernel `C(d)` with bootstrap CIs, Fourier eigenvalues,
class-averaging R^2, per-month theory cosines, angular steps).

It also writes a four-panel figure:

- `(a)` centered-`M*` eigenvalues, colored by Fourier wavenumber
- `(b)` co-occurrence vs circular month distance
- `(c)` empirical vs theory embeddings in the `k=1` and `k=2` planes

Saved as `months.pdf` and `months.png` (or the stem of `--out`).
