"""
llm_cube_modes.py
-----------------
The cube3 "empirical eigenvectors vs Walsh characters" panel, computed
from an LLM's representations of the complete Z_2^3 cube
  man/woman/men/women/king/queen/kings/queens
(attributes: gender g, number n, royalty r).

Vector sources (--source):
  unembed   rows of the LM head / unembedding matrix (default;
            Park-style static vectors, optionally whitened)
  embed     rows of the input embedding matrix
  layer     hidden state of the target word at --layer, mean-pooled
            over a set of neutral template sentences (contextual)

The 8 vectors enter exactly as ChordBERT activations do in the paper
pipeline: doubly-centered Gram -> eigenmodes -> match each mode to its
best Walsh character chi_S (sign-gauged) -> side-by-side heatmaps plus
the per-mode flip test d1[k] d2[k] for king:queen::kings:queens.

Usage:
  python3 llm_cube_modes.py --model gpt2 --source unembed
  python3 llm_cube_modes.py --model meta-llama/Llama-3.2-1B \
      --source layer --layer 8
  python3 llm_cube_modes.py --mock   # structural self-test, no model

Requires: transformers, torch (except --mock).
"""
import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORDS = {(+1, +1, +1): "king",  (-1, +1, +1): "queen",
         (+1, -1, +1): "kings", (-1, -1, +1): "queens",
         (+1, +1, -1): "man",   (-1, +1, -1): "woman",
         (+1, -1, -1): "men",   (-1, -1, -1): "women"}
TEMPLATES = ["I saw the {} yesterday.", "The {} walked in.",
             "Everyone spoke about the {} at dinner.",
             "A story about the {} was told.",
             "They pointed at the {} quietly."]


def get_vectors(args, targets):
    rng = np.random.default_rng(0)
    if args.mock:
        pts = np.array(list(WORDS))
        X = np.zeros((8, 64))
        for k, w in enumerate([2.0, 1.2, 0.8]):      # planted degree-1
            X[:, k] = w * pts[:, k]
        X[:, 3] = 0.4 * pts[:, 0] * pts[:, 1] * pts[:, 2]  # planted gnr
        X += 0.05 * rng.standard_normal(X.shape)
        return X
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, output_hidden_states=(args.source == "layer"),
        torch_dtype=torch.float32)
    model.eval()
    if args.source in ("unembed", "embed"):
        M = (model.get_output_embeddings().weight
             if args.source == "unembed"
             else model.get_input_embeddings().weight)
        M = M.detach().cpu().numpy()
        X = []
        for w in targets:
            ids = tok(" " + w, add_special_tokens=False)["input_ids"]
            if len(ids) > 1:
                print(f"  note: '{w}' is {len(ids)} tokens; pooling")
            X.append(M[ids].mean(0))
        X = np.stack(X)
        if args.whiten:
            mu = M.mean(0)
            C = np.cov((M - mu).T) + 1e-6 * np.eye(M.shape[1])
            Wh = np.linalg.cholesky(np.linalg.inv(C)).T
            X = (X - mu) @ Wh
        return X
    # contextual layer vectors
    X = []
    with torch.no_grad():
        for w in targets:
            vecs = []
            for t in TEMPLATES:
                text = t.format(w)
                ids = tok(text, return_tensors="pt")
                hs = model(**ids).hidden_states[args.layer][0]
                wids = tok(" " + w, add_special_tokens=False)["input_ids"]
                seq = ids["input_ids"][0].tolist()
                for p in range(len(seq) - len(wids) + 1):
                    if seq[p:p + len(wids)] == wids:
                        vecs.append(hs[p:p + len(wids)].mean(0).numpy())
                        break
            X.append(np.mean(vecs, 0))
    return np.stack(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--source", default="unembed",
                    choices=["unembed", "embed", "layer"])
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="llm_cube_modes")
    args = ap.parse_args()

    order = list(WORDS)
    targets = [WORDS[p] for p in order]
    pts = np.array(order)
    X = get_vectors(args, targets)
    Xc = X - X.mean(0, keepdims=True)
    N = 8
    H = np.eye(N) - np.ones((N, N)) / N
    M = H @ (Xc @ Xc.T) @ H
    M = 0.5 * (M + M.T)
    M /= max(np.sqrt(np.mean(M ** 2)), 1e-12)

    # invariance of the Gram
    idx = {tuple(p): i for i, p in enumerate(order)}
    R = np.zeros((N, N))
    for f in itertools.product([1, -1], repeat=3):
        perm = [idx[tuple(p * np.array(f))] for p in pts]
        R += M[np.ix_(perm, perm)]
    R /= 8.0
    r2 = 1 - np.var(M - R) / np.var(M)
    print(f"Z_2^3 invariance of the LLM Gram: R^2 = {r2:.3f}")

    subsets = [S for r_ in range(1, 4)
               for S in itertools.combinations("gnr", r_)]

    def chi(S):
        i = {"g": 0, "n": 1, "r": 2}
        return np.array([np.prod([p[i[c]] for c in S])
                         for p in pts]) / np.sqrt(N)

    lam, U = np.linalg.eigh(M)
    o = np.argsort(-np.abs(lam))
    lam, U = lam[o], U[:, o]
    keepk = [k for k in range(N) if np.abs(lam[k]) > 1e-8][:7]
    E = np.zeros((N, 7))
    Th = np.zeros((N, 7))
    names, ovls = [], []
    for col, k in enumerate(keepk):
        ov = [U[:, k] @ chi(S) for S in subsets]
        j = int(np.argmax(np.abs(ov)))
        E[:, col] = np.sign(ov[j]) * U[:, k]
        Th[:, col] = chi(subsets[j])
        names.append("".join(subsets[j]))
        ovls.append(abs(ov[j]))
    print("matched characters:", names)
    print("overlaps:", [round(float(v), 3) for v in ovls])

    # per-mode flip test, king:queen::kings:queens
    Wv = U[:, keepk] * np.sqrt(np.abs(lam[keepk]))
    iw = {w: i for i, w in enumerate(targets)}
    d1 = Wv[iw["king"]] - Wv[iw["queen"]]
    d2 = Wv[iw["kings"]] - Wv[iw["queens"]]
    prods = d1 * d2
    print("flip test d1[k]*d2[k]:",
          {nm: round(float(p), 2) for nm, p in zip(names, prods)})

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4),
                           gridspec_kw={"width_ratios": [1, 1, 1.1]})
    vm = np.max(np.abs(E))
    for A, a, title in [(E, ax[0], "empirical eigenvectors of the\n"
                         "LLM Gram ($|\\lambda|$ order, sign-aligned)"),
                        (Th, ax[1], "theory: Walsh characters "
                         "$\\chi_S$")]:
        a.imshow(A, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
        a.set_yticks(range(N))
        a.set_yticklabels(targets, fontsize=8)
        a.set_xticks(range(7))
        if A is E:
            a.set_xticklabels([f"{nm}\n{ov:.2f}" for nm, ov
                               in zip(names, ovls)], fontsize=7.5)
        else:
            a.set_xticklabels(names, fontsize=9)
        a.set_title(title, fontsize=10)
    ax[2].bar(range(7), prods,
              color=["steelblue" if p > 0 else "crimson" for p in prods])
    ax[2].set_xticks(range(7))
    ax[2].set_xticklabels(names, fontsize=9)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_title("flip test: $d_1[k]\\,d_2[k]$\n"
                    "(king:queen::kings:queens)", fontsize=10)
    src = args.source + (f" L{args.layer}" if args.source == "layer"
                         else "")
    fig.suptitle(f"{'MOCK' if args.mock else args.model} ({src}); "
                 f"Gram invariance $R^2$={r2:.2f}", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=170, bbox_inches="tight")
    print(f"saved -> {args.out}.pdf/.png")


if __name__ == "__main__":
    main()