"""
extract_chords.py
-----------------
Extract chord-token sequences from the Bach chorales (music21's bundled
corpus, works offline). Each score is chordified and every vertical sonority
is classified; recognized chords become integer tokens (see chordlib for the
token layout), everything else is dropped.

Two vocabularies:
  --vocab triads   (default) keep only clean major/minor triads (24 tokens)
                   -> chord_sequences.json
  --vocab full     also keep diminished, augmented, and the common seventh
                   families (up to 108 tokens) -> chord_sequences_full.json

Usage:  python3 extract_chords.py [--vocab triads|full] [--n 400] [--out F]
"""
import argparse
import json
import warnings

warnings.filterwarnings("ignore")
from music21 import corpus

from chordlib import FAMILY_NAMES


def token_of(ch, full_vocab):
    """Integer token for a music21 chord, or None to drop it."""
    try:
        r = ch.root().pitchClass
        if ch.isMajorTriad():
            return r
        if ch.isMinorTriad():
            return 12 + r
        if not full_vocab:
            return None
        if ch.isDiminishedTriad():
            return 24 + r
        if ch.isAugmentedTriad():
            return 36 + r
        if ch.isDominantSeventh():
            return 48 + r
        if ch.isDiminishedSeventh():
            return 84 + r
        if ch.isHalfDiminishedSeventh():
            return 72 + r
        common = ch.commonName or ""
        if "minor seventh chord" in common:
            return 60 + r
        if "major seventh chord" in common:
            return 96 + r
    except Exception:
        return None
    return None


def extract_one(score, full_vocab):
    chords = score.chordify().recurse().getElementsByClass("Chord")
    seq = [token_of(ch, full_vocab) for ch in chords]
    return [t for t in seq if t is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", choices=["triads", "full"], default="triads")
    ap.add_argument("--n", type=int, default=400, help="max chorales")
    ap.add_argument("--out", default=None, help="output json path")
    args = ap.parse_args()
    full = args.vocab == "full"
    out = args.out or ("chord_sequences_full.json" if full
                       else "chord_sequences.json")

    try:
        items = list(corpus.chorales.Iterator())
    except Exception:
        items = [corpus.parse(p) for p in corpus.getComposer("bach")]

    sequences, family_counts = [], {}
    for item in items:
        if len(sequences) >= args.n:
            break
        try:
            score = item if hasattr(item, "chordify") else corpus.parse(item)
            seq = extract_one(score, full)
        except Exception:
            continue
        if len(seq) < 8:                     # skip near-empty extractions
            continue
        sequences.append(seq)
        for t in seq:
            family_counts[t // 12] = family_counts.get(t // 12, 0) + 1

    with open(out, "w") as f:
        json.dump(sequences, f)

    n_tok = sum(len(s) for s in sequences)
    print(f"chorales kept:      {len(sequences)}")
    print(f"total chord tokens: {n_tok}")
    for fam in sorted(family_counts):
        c = family_counts[fam]
        print(f"  {FAMILY_NAMES[fam]:>6}: {c:>6}  ({100 * c / n_tok:.1f}%)")
    print(f"mean tokens/chorale: {n_tok / len(sequences):.1f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()