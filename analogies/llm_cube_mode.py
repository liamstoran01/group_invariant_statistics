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

Layer selection (--select-layer): sweep every hidden state, linearly
probe the three ±1 attributes from a training subset of the 8 words
(min-norm OLS by default), and pick the layer with the best held-out
R^2 (default: exhaustive leave-2-out). This is the cube analogue of
the sky decode-R^2 sweep, with a hold-out so d >> n cannot trivially
interpolate.

Usage:
  python3 llm_cube_mode.py --model gpt2 --source unembed
  python3 llm_cube_mode.py --model gpt2 --source layer --select-layer
  python3 llm_cube_mode.py --model meta-llama/Llama-3.2-1B \
      --source layer --layer 8
  python3 llm_cube_mode.py --mock   # structural self-test, no model

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
def mock_layer_stack(n_layers=5, dim=64, seed=0):
    """Planted cube at layer 2; other layers are noise."""
    rng = np.random.default_rng(seed)
    pts = np.array(list(WORDS), dtype=float)
    stack = rng.standard_normal((n_layers, 8, dim))
    X = np.zeros((8, dim))
    for k, w in enumerate([2.0, 1.2, 0.8]):
        X[:, k] = w * pts[:, k]
    X[:, 3] = 0.4 * pts[:, 0] * pts[:, 1] * pts[:, 2]
    stack[2] = X + 0.05 * rng.standard_normal(X.shape)
    return stack


def _locate_word(tok, seq, w):
    wids = tok(" " + w, add_special_tokens=False)["input_ids"]
    for p in range(len(seq) - len(wids) + 1):
        if seq[p:p + len(wids)] == wids:
            return p, p + len(wids)
    wids = tok(w, add_special_tokens=False)["input_ids"]
    for p in range(len(seq) - len(wids) + 1):
        if seq[p:p + len(wids)] == wids:
            return p, p + len(wids)
    return None


def collect_layer_stack(model, tok, targets, device):
    """(n_layers, 8, d) contextual states, HF index 0 = embeddings."""
    import torch
    model.eval()
    per_word = []
    with torch.no_grad():
        for w in targets:
            vecs = []
            for t in TEMPLATES:
                ids = tok(t.format(w), return_tensors="pt")
                ids = {k: v.to(device) for k, v in ids.items()}
                hs = model(**ids, output_hidden_states=True).hidden_states
                seq = ids["input_ids"][0].tolist()
                loc = _locate_word(tok, seq, w)
                if loc is None:
                    continue
                a, b = loc
                vecs.append(np.stack([
                    h[0, a:b].mean(0).detach().cpu().float().numpy()
                    for h in hs]))
            if not vecs:
                raise RuntimeError(f"could not locate '{w}' in templates")
            per_word.append(np.mean(vecs, 0))
    return np.stack(per_word, 1)


def static_vectors(model, tok, targets, source, whiten):
    M = (model.get_output_embeddings().weight
         if source == "unembed"
         else model.get_input_embeddings().weight)
    M = M.detach().cpu().float().numpy()
    X = []
    for w in targets:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(ids) > 1:
            print(f"  note: '{w}' is {len(ids)} tokens; pooling")
        X.append(M[ids].mean(0))
    X = np.stack(X)
    if whiten:
        mu = M.mean(0)
        C = np.cov((M - mu).T) + 1e-6 * np.eye(M.shape[1])
        Wh = np.linalg.cholesky(np.linalg.inv(C)).T
        X = (X - mu) @ Wh
    return X


def ridge_predict(Xtr, Ytr, Xte, lam):
    """Center on train; min-norm OLS (lam=0) or ridge, intercept free."""
    mu = Xtr.mean(0)
    Ztr = Xtr - mu
    Zte = Xte - mu
    n, d = Ztr.shape
    X1 = np.concatenate([np.ones((n, 1)), Ztr], 1)
    if lam <= 0:
        W = np.linalg.lstsq(X1, Ytr, rcond=None)[0]
    else:
        G = X1.T @ X1
        G[1:, 1:] += lam * np.eye(d)
        W = np.linalg.solve(G, X1.T @ Ytr)
    Xte1 = np.concatenate([np.ones((len(Zte), 1)), Zte], 1)
    return Xte1 @ W


def heldout_probe(X, Y, n_test=2, n_splits=0, lam=1.0, seed=0):
    """Mean held-out R^2 and sign accuracy of attribute probes.

    n_splits <= 0 uses every n_test-subset (C(8,2)=28 by default).
    """
    n = len(X)
    idx = np.arange(n)
    if n_splits <= 0:
        splits = list(itertools.combinations(idx, n_test))
    else:
        rng = np.random.default_rng(seed)
        splits = [tuple(sorted(rng.choice(n, n_test, replace=False)))
                  for _ in range(n_splits)]
    r2s, accs = [], []
    for te in splits:
        te = np.asarray(te)
        tr = np.array([i for i in idx if i not in set(te.tolist())])
        yhat = ridge_predict(X[tr], Y[tr], X[te], lam)
        ytrue = Y[te]
        mu = Y[tr].mean(0)
        num = ((ytrue - yhat) ** 2).sum(0)
        den = ((ytrue - mu) ** 2).sum(0)
        r2s.append(1.0 - num / np.maximum(den, 1e-12))
        accs.append(np.mean(np.sign(yhat) == ytrue))
    r2s = np.array(r2s)
    return dict(r2=float(np.mean(r2s)),
                r2_attr=np.mean(r2s, 0),
                acc=float(np.mean(accs)),
                n_splits=len(splits))


def select_layer(stack, Y, n_test, n_splits, lam, seed, out_stem):
    L = stack.shape[0]
    rows = []
    print(f"\nheld-out attribute probe  ({L} layers, n_test={n_test}, "
          f"{'exhaustive' if n_splits <= 0 else n_splits} splits, "
          f"ridge={lam:g})")
    print(f"{'layer':>6} {'R^2':>8}  {'g':>7} {'n':>7} {'r':>7}  {'sign':>6}")
    for l in range(L):
        s = heldout_probe(stack[l], Y, n_test, n_splits, lam, seed)
        rows.append(s)
        ra = s["r2_attr"]
        print(f"{l:>6} {s['r2']:>8.3f}  {ra[0]:>7.3f} {ra[1]:>7.3f} "
              f"{ra[2]:>7.3f}  {s['acc']:>6.2f}")
    best = int(np.argmax([s["r2"] for s in rows]))
    print(f"selected layer {best}  (held-out R^2 = {rows[best]['r2']:.3f}, "
          f"sign acc {rows[best]['acc']:.2f})")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(range(L), [s["r2"] for s in rows], "o-", ms=5, lw=1.5,
            label="held-out $R^2$")
    ax.axvline(best, color="k", ls=":", lw=0.9)
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out probe $R^2$")
    ax.set_title("layer selected by held-out attribute decode")
    ax.legend(fontsize=9)
    fig.tight_layout()
    sweep = f"{out_stem}_layer_probe"
    fig.savefig(f"{sweep}.pdf", bbox_inches="tight")
    fig.savefig(f"{sweep}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {sweep}.pdf/.png")
    return best, rows[best]


def get_vectors(args, targets):
    if args.mock:
        stack = mock_layer_stack()
        if args.select_layer:
            return stack, None
        layer = args.layer if args.source == "layer" else 2
        layer = int(np.clip(layer, 0, stack.shape[0] - 1))
        return stack[layer], None
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    if args.source in ("unembed", "embed") and not args.select_layer:
        return static_vectors(model, tok, targets, args.source,
                              args.whiten), None
    print(f"torch device: {device}")
    stack = collect_layer_stack(model, tok, targets, device)
    if args.select_layer:
        return stack, None
    L = stack.shape[0]
    if not 0 <= args.layer < L:
        raise SystemExit(f"--layer {args.layer} out of range 0..{L - 1}")
    return stack[args.layer], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--source", default="unembed",
                    choices=["unembed", "embed", "layer"])
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--select-layer", action="store_true",
                    help="pick layer by held-out linear probe of (g,n,r)")
    ap.add_argument("--n-test", type=int, default=2,
                    help="held-out words per split (default 2 of 8)")
    ap.add_argument("--n-splits", type=int, default=0,
                    help="random splits; 0 = all C(8, n_test) subsets")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="ridge penalty (0 = min-norm OLS; no feature scaling)")
    ap.add_argument("--probe-seed", type=int, default=0)
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="llm_cube_modes")
    args = ap.parse_args()

    order = list(WORDS)
    targets = [WORDS[p] for p in order]
    pts = np.array(order)
    Y = pts.astype(float)
    raw, _ = get_vectors(args, targets)
    probe = None
    if args.select_layer:
        if raw.ndim != 3:
            raise SystemExit("--select-layer needs contextual layer states")
        args.source = "layer"
        args.layer, probe = select_layer(
            raw, Y, args.n_test, args.n_splits, args.ridge,
            args.probe_seed, args.out)
        X = raw[args.layer]
    else:
        X = raw
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
    extra = ""
    if probe is not None:
        extra = f"; probe $R^2$={probe['r2']:.2f}"
    fig.suptitle(f"{'MOCK' if args.mock else args.model} ({src}); "
                 f"Gram invariance $R^2$={r2:.2f}{extra}",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=170, bbox_inches="tight")
    print(f"saved -> {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
