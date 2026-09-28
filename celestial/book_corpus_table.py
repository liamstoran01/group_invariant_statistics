"""
book_corpus_table.py
--------------------
Regenerates the corpus-selection table (book_genre_table.md / .tex):
one row per candidate astronomy book, sorted by the gate statistic D
(descending; untestable books last). SELECTION IS THE SINGLE GATE:
D > 0 with one-sided location-permutation p < 0.05 (boldface rows).

  * D (p) -- Pearson corr(M*_ij, cos theta_ij) with one-sided
    location-permutation p over 500 shuffles; this is the selection gate.
  * coherence  -- genre DIAGNOSTIC (not a gate): median angular hop
    (degrees) between successive sky-object mentions; permutation
    z-score vs the book's shuffled-mention null;
  * lambda_1   -- descriptive kernel statistic: Funk-Hecke dipole
    coefficient of the book's own zonal kernel, with permutation p;
  * kernel corr -- graded validation: correlation of the book's kernel
    with the LLM layer-72 activation kernel (never used for selection).

Selection uses text + coordinates only (no model). ^ marks kernels
fit at count floor 3 instead of 5.

Usage: python3 book_corpus_table.py   (books in BOOKS_DIR; writes
       book_genre_table.md and book_genre_table.tex)

Gate statistics (D, kernels, lambda_1) use off-diagonal pairs only, so
there is no diagonal pinning step (--no-pin is the only convention).
"""
import glob

import numpy as np
from scipy.special import eval_legendre

import skylib as sk

TITLES = {
 "20769": ("A Field Book of the Stars", "Olcott, 1907"),
 "36741": ("Astronomy with an Opera-Glass", "Serviss, 1888"),
 "68391": ("Round the Year with the Stars", "Olcott, 1912"),
 "57091": ("Astronomy for Young Australians", "Bonwick, 1866"),
 "28752": ("Pleasures of the Telescope", "Serviss, 1901"),
 "23300": ("Half-Hours with the Stars", "Proctor, 1887"),
 "16767": ("Half-Hours with the Telescope", "Proctor, 1868"),
 "25267": ("Astronomy for Amateurs", "Flammarion, 1904"),
 "45112": ("Astronomy for Young Folks", "Lewis, 1922"),
 "15620": ("Recreations in Astronomy", "Warren, 1879"),
 "27378": ("The Story of the Heavens", "Ball, 1885"),
 "58810": ("The Heavens Above", "Gillet & Rolfe, 1882"),
 "53172": ("Stargazing Past and Present", "Lockyer, 1878"),
 "54913": ("Stories of Starland", "Proctor (M.), 1895"),
 "28853": ("The Children's Book of Stars", "Mitton, 1907"),
 "36495": ("Astronomical Myths", "Blake, 1877"),
 "67234": ("The Book of Stars", "Collins, 1915"),
 "19395": ("The New Heavens", "Hale, 1922"),
 "34834": ("A Text-Book of Astronomy", "Todd, 1897"),
 "40240": ("Letters on Astronomy", "Olmsted, 1840"),
 "26556": ("Myths and Marvels of Astronomy", "Proctor, 1877"),
 "70052": ("Ancient Calendars and Constellations", "Plunket, 1903"),
 "71943": ("Popular Lessons in Astronomy", "(reader)"),
 "60318": ("Star-Land", "Ball, 1889")}

P1 = eval_legendre(1, sk.XQ)


def main():
    names, lat, lon, aliases = sk.load_objects()
    xyzO = sk.unit_vectors(lat, lon)

    # LLM reference kernel (graded-validation target)
    latL, lonL, xyzL, lamL, UL, _ = sk.llm_modes()
    wL, cosL = sk.density_weights(xyzL)
    E_L = UL * np.sqrt(np.abs(lamL))
    G = E_L @ E_L.T
    c_llm = sk.fit_zonal(G / np.sqrt(np.mean(np.diag(G) ** 2)), cosL, wL)
    z = lambda v: (v - v.mean()) / v.std()
    zl = z(c_llm)

    # pooled 4-book kernel vs model kernel
    names_, lat_, lon_, acc, canon = sk.corpus_stream()
    keep, Mb, _, _ = sk.corpus_mstar(names_, acc, canon)
    xyzb = sk.unit_vectors(lat_[keep], lon_[keep])
    wb, cosb = sk.density_weights(xyzb)
    c_books = sk.fit_zonal(Mb, cosb, wb)
    r_raw = float(np.corrcoef(z(c_books), zl)[0, 1])
    print(f"POOLED corpus kernel vs model kernel: r = {r_raw:.2f}")

    rng = np.random.default_rng(0)
    rows = []
    paths = sorted(glob.glob(f"{sk.BOOKS_DIR}/*.txt"))
    n_skip_m = 0
    for path in paths:
        bid = path.split("/")[-1].replace(".txt", "")
        toks, canon = sk.book_tokens(bid, names, aliases)
        tok_of = {canon[nm]: i for i, nm in enumerate(names)}
        seq = np.array([tok_of[t] for t in toks if t in tok_of])
        m = len(seq)
        if m < 40:
            n_skip_m += 1
            continue

        # stage 1: coherence
        def med(s):
            a, b = s[:-1], s[1:]
            msk = a != b
            return np.degrees(np.median(np.arccos(np.clip(
                np.sum(xyzO[a[msk]] * xyzO[b[msk]], 1), -1, 1))))
        obs = med(seq)
        nullc = [med(rng.permutation(seq)) for _ in range(150)]
        coh = obs                               # median angular hop (deg)
        zsc = (np.mean(nullc) - obs) / max(np.std(nullc), 1e-9)

        entry = dict(coh=coh, z=zsc, m=m, mc=None, lam1=None,
                     p1=None,
                     corr=None, D=None, pD=None,
                     title=TITLES.get(bid, (bid, ""))[0],
                     auth=TITLES.get(bid, (bid, ""))[1])
        for mc in (5, 3):
            keep, Mb, p, ntok = sk.corpus_mstar(names, list(toks), canon,
                                                min_count=mc)
            if len(keep) < 15:
                continue
            entry["mc"] = mc
            xyzb = sk.unit_vectors(lat[keep], lon[keep])
            wb, cosb = sk.density_weights(xyzb)
            c_fit = sk.fit_zonal(Mb, cosb, wb)
            entry["lam1"] = float(2 * np.pi * np.trapezoid(c_fit * P1,
                                                           sk.XQ))
            entry["corr"] = float(np.corrcoef(z(c_fit), zl)[0, 1])
            # single-gate field statistic D = corr(M*, cos theta)
            nbD = len(keep)
            iuD = np.triu_indices(nbD, 1)
            Mv = Mb[iuD]
            cosM = np.clip(xyzb @ xyzb.T, -1, 1)
            entry["D"] = float(np.corrcoef(Mv, cosM[iuD])[0, 1])
            cD = 0
            for _ in range(500):
                pm2 = rng.permutation(nbD)
                cp = cosM[np.ix_(pm2, pm2)][iuD]
                cD += np.corrcoef(Mv, cp)[0, 1] >= entry["D"]
            entry["pD"] = float((cD + 1) / 501)
            # stage 2: one-sided permutation p for lambda_1
            nb = len(keep)
            iub = np.triu_indices(nb, 1)
            nl = []
            for _ in range(300):
                pm = rng.permutation(nb)
                xp = np.clip((xyzb[pm] @ xyzb[pm].T), -1, 1)[iub]
                wp = wb[pm][iub[0]] * wb[pm][iub[1]]
                cN = sk.estimate_c(xp, Mb[iub], wp)
                nl.append(float(2 * np.pi * np.trapezoid(cN * P1, sk.XQ)))
            entry["p1"] = float((np.array(nl) >= entry["lam1"]).mean())
            break
        rows.append(entry)
    rows.sort(key=lambda e: (e["D"] is None,
                            -(e["D"] if e["D"] is not None
                              else 0)))

    md = ["# Astronomy corpus: book selection",
          "",
          "| # | book | author, year | ment. | coherence (z) | "
          "D (p) | lambda_1 (p, 1-sided) | kernel corr |",
          "|---|---|---|---|---|---|---|---|"]
    for i, e in enumerate(rows, 1):
        dag = "^" if e["mc"] == 3 else ""
        if e["lam1"] is None:
            l1s = cc = dcol = "--"
        else:
            l1s = f"{e['lam1']:+.2f} (p={e['p1']:.3f})"
            hot = e["corr"] >= 0.7
            cc = (f"{'**' if hot else ''}{e['corr']:+.2f}"
                  f"{'**' if hot else ''}{dag}")
            sel = "**" if (e["D"] > 0 and e["pD"] < 0.05) else ""
            dcol = f"{sel}{e['D']:+.2f} ({e['pD']:.3f}){sel}{dag}"
        md.append(f"| {i} | *{e['title']}* | {e['auth']} | {e['m']} | "
                  f"{e['coh']:.0f} ({e['z']:.1f}) | {dcol} | {l1s} | {cc} |")
    open("book_genre_table.md", "w").write("\n".join(md) + "\n")

    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{@{}l l r r r r r@{}}", r"\toprule",
           r"Book & Author, year & ment. & coher.\ ($z$) & $D$ ($p$) & $\lambda_1$ "
           r"($p$) & kernel corr \\", r"\midrule"]
    for e in rows:
        t = e["title"].replace("&", r"\&")
        if e["corr"] is not None and e["corr"] >= 0.7:
            t = r"\textbf{" + t + "}"
        dag = r"$^\dagger$" if e["mc"] == 3 else ""
        l1 = "--" if e["lam1"] is None else \
            f"{e['lam1']:+.2f} ({e['p1']:.2f})"
        cc = "--" if e["corr"] is None else f"{e['corr']:+.2f}{dag}"
        if e["D"] is None:
            dcol = "--"
        else:
            dcol = f"{e['D']:+.2f} ({e['pD']:.2f}){dag}"
            if e["D"] > 0 and e["pD"] < 0.05:
                dcol = r"\textbf{" + dcol + "}"
        tex.append(f"{t} & {e['auth']} & {e['m']} & "
                   f"{e['coh']:.0f} ({e['z']:.1f}) & {dcol} & {l1} & {cc} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\label{tab:astrobooks}", r"\end{table}"]
    open("book_genre_table.tex", "w").write("\n".join(tex))
    n_table = len(rows)
    n_D = sum(e["D"] is not None for e in rows)
    print(f"table inclusion: m >= 40 sky-object mentions "
          f"({n_table} of {len(paths)}; {n_skip_m} skipped)")
    print(f"D / lambda_1 / kernel corr: need >= 15 objects after "
          f"count floor (min_count=5, else 3); else -- "
          f"({n_D} of {n_table} books)")
    print("written -> book_genre_table.md, book_genre_table.tex")
    for e in rows[:6]:
        print(f"  {e['title'][:34]:>36}  hop {e['coh']:.0f} "
              f"(z {e['z']:.1f})  "
              f"lam1 {'--' if e['lam1'] is None else round(e['lam1'], 2)} "
              f"p {'--' if e['p1'] is None else round(e['p1'], 3)}  "
              f"D {'--' if e['D'] is None else format(e['D'], '+.2f')} "
              f"(p {'--' if e['pD'] is None else round(e['pD'], 3)})")


if __name__ == "__main__":
    main()