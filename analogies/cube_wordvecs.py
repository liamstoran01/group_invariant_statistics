"""
cube_wordvecs.py
----------------
Train Word2Vec and GloVe from scratch on the cube corpus (text8 by
default), then make the Boolean-irrep overlay of cube3_corpus_llm.py
panel (b) for each model: empirical Gram eigenvectors (fill) vs matched
Walsh characters (frame).

Word2Vec: skip-gram + negative sampling in PyTorch (GPU if CUDA),
streaming window pairs so text8 does not have to sit on device.
GloVe: AdaGrad factorization of windowed co-occurrence (same recipe as
celestial/sky_wordvecs.py).

Writes cube_wordvecs.pdf/.png (Word2Vec | GloVe).

Caches encoded text8, GloVe co-occurrence, and embeddings under
cube_wordvecs_cache/ so later runs skip tokenization / pair counting /
converged Word2Vec. GloVe stops when relative SSE plateaus (or hits
--glove-max-iter); a checkpoint is written every 10 steps.

Usage:
  python3 cube_wordvecs.py
  python3 cube_wordvecs.py --text ../months/text8.txt --window 10
  python3 cube_wordvecs.py --pretrained
"""
import argparse
import gzip
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm

from cube3_corpus_llm import (
    _DEFAULT_TEXT8, draw_overlay, gram_from_vectors, match_walsh, reynolds,
    subset_label,
)
from cube_case import WORDS

_CACHE = Path(__file__).resolve().parent / "cube_wordvecs_cache"

# Released models: Mikolov Google News skip-gram, Pennington GloVe 6B.
_W2V_URL = ("https://huggingface.co/LoganKilpatrick/"
            "GoogleNews-vectors-negative300/resolve/main/"
            "GoogleNews-vectors-negative300.bin.gz")
_GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
_GLOVE_MEMBER = "glove.6B.300d.txt"


def _corpus_tag(text_path):
    stem = Path(text_path).stem.lower()
    return "" if stem == "text8" else f"_{stem}"


def _cache_paths(min_count, window, dim, seed, subsystem=False, corpus=""):
    _CACHE.mkdir(exist_ok=True)
    tag = "_sub" if subsystem else ""
    return dict(
        corpus=_CACHE / f"corpus{corpus}_mc{min_count}.npz",
        cooc=_CACHE / f"glove_cooc{tag}{corpus}_w{window}_mc{min_count}.npz",
        w2v=_CACHE / f"w2v{tag}{corpus}_d{dim}_w{window}_s{seed}.npy",
        glove=_CACHE / f"glove{tag}{corpus}_d{dim}_w{window}_s{seed}.npy",
        glove_ckpt=_CACHE / f"glove{tag}{corpus}_ckpt_d{dim}_w{window}_s{seed}.pt",
    )


def _download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"using {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"downloading {url}")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        pbar = tqdm(total=total or None, unit="B", unit_scale=True,
                    desc=dest.name, mininterval=1.0, dynamic_ncols=True)
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            pbar.update(len(chunk))
        pbar.close()
    tmp.rename(dest)
    print(f"  saved {dest.name} ({dest.stat().st_size:,} bytes)")
    return dest


def _stack_keep(found, keep, name):
    missing = [w for w in keep if w not in found]
    if missing:
        raise SystemExit(f"{name} missing vectors for {missing}")
    X = np.stack([found[w] for w in keep], 0).astype(np.float64)
    print(f"{name}: {X.shape[1]}d, {len(keep)} cube words")
    return X


def load_pretrained_word2vec(keep, cache_dir=_CACHE):
    npy = cache_dir / "pretrained_w2v_google300_cube8.npy"
    if npy.is_file():
        X = np.load(npy)
        print(f"loaded {npy.name}  shape {X.shape}")
        return X
    gz = _download(_W2V_URL, cache_dir / "GoogleNews-vectors-negative300.bin.gz")
    want = set(keep)
    found, fallback = {}, {}
    with gzip.open(gz, "rb") as f:
        header = f.readline().decode("utf-8").split()
        n, dim = int(header[0]), int(header[1])
        print(f"Word2Vec Google News: {n:,} vectors, dim {dim}")
        word_buf = bytearray()
        pbar = tqdm(total=n, desc="scanning Word2Vec", unit="tok",
                    mininterval=1.0, dynamic_ncols=True)
        for i in range(n):
            word_buf.clear()
            while True:
                ch = f.read(1)
                if not ch or ch == b" ":
                    break
                if ch != b"\n":
                    word_buf.extend(ch)
            vec = np.frombuffer(f.read(4 * dim), dtype="<f4").copy()
            word = word_buf.decode("latin-1")
            if word in want and word not in found:
                found[word] = vec
            else:
                key = word.lower()
                if key in want and key not in found and key not in fallback:
                    fallback[key] = vec
            pbar.update(1)
            if len(found) == len(want):
                pbar.close()
                print(f"  found all {len(want)} words at rank {i + 1:,}")
                break
        else:
            pbar.close()
    for w in keep:
        if w not in found and w in fallback:
            print(f"  {w}: using case-folded Google News entry")
            found[w] = fallback[w]
    X = _stack_keep(found, keep, "Word2Vec Google News")
    np.save(npy, X)
    print(f"cached {npy.name}")
    return X


def load_pretrained_glove(keep, cache_dir=_CACHE):
    npy = cache_dir / "pretrained_glove6b300_cube8.npy"
    if npy.is_file():
        X = np.load(npy)
        print(f"loaded {npy.name}  shape {X.shape}")
        return X
    zpath = _download(_GLOVE_URL, cache_dir / "glove.6B.zip")
    want = set(keep)
    found = {}
    with zipfile.ZipFile(zpath) as zf:
        with zf.open(_GLOVE_MEMBER) as raw:
            pbar = tqdm(desc="scanning GloVe 6B 300d", unit="tok",
                        mininterval=1.0, dynamic_ncols=True)
            for line in raw:
                tok, *rest = line.decode("utf-8").split()
                pbar.update(1)
                if tok not in want:
                    continue
                found[tok] = np.fromiter(rest, dtype=np.float32)
                if len(found) == len(want):
                    break
            pbar.close()
    X = _stack_keep(found, keep, "GloVe 6B 300d")
    np.save(npy, X)
    print(f"cached {npy.name}")
    return X


def encode_corpus(path, keep, min_count=5, cache=None):
    if cache is not None and cache.is_file():
        t0 = time.perf_counter()
        z = np.load(cache, allow_pickle=False)
        ids = z["ids"]
        words = [str(w) for w in z["words"]]
        vocab = {w: i for i, w in enumerate(words)}
        print(f"loaded encoded corpus {cache.name}: {len(ids):,} tokens, "
              f"vocab {len(vocab):,}  ({time.perf_counter() - t0:.1f}s)")
        return vocab, ids
    t0 = time.perf_counter()
    print(f"reading {path} ({Path(path).stat().st_size:,} bytes)...")
    toks = open(path).read().split()
    print(f"  {len(toks):,} tokens, counting...")
    cnt = Counter(toks)
    vocab = {}
    for w in keep:
        vocab[w] = len(vocab)
    vocab["<unk>"] = len(vocab)
    unk = vocab["<unk>"]
    for w, c in cnt.items():
        if c >= min_count and w not in vocab:
            vocab[w] = len(vocab)
    ids = np.fromiter(
        (vocab.get(t, unk) for t in tqdm(
            toks, desc="encoding ids", unit="tok", mininterval=1.0,
            dynamic_ncols=True)),
        dtype=np.int32, count=len(toks))
    words = [None] * len(vocab)
    for w, i in vocab.items():
        words[i] = w
    if cache is not None:
        np.savez_compressed(cache, ids=ids, words=np.array(words))
        print(f"cached encoded corpus -> {cache.name}")
    print(f"corpus {path}: {len(ids):,} tokens, vocab {len(vocab):,} "
          f"(min_count={min_count})  ({time.perf_counter() - t0:.1f}s)")
    for w in keep:
        print(f"  {w}: {cnt[w]}  id={vocab[w]}")
    return vocab, ids


def cube_rows(vocab, X, targets):
    missing = [w for w in targets if w not in vocab]
    if missing:
        raise SystemExit(f"missing embeddings for {missing}")
    return np.stack([X[vocab[w]] for w in targets], 0).astype(np.float64)


def train_word2vec(ids, V, window, dim, epochs, seed, n_neg=5,
                   batch=65536, lr=0.01):
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Word2Vec skip-gram (torch {device}): {len(ids):,} tokens, "
          f"V={V}, dim {dim}, window {window}, {epochs} epochs, seed {seed}")
    torch.manual_seed(seed)
    counts = np.bincount(ids, minlength=V).astype(np.float64)
    noise = np.power(np.maximum(counts, 1.0), 0.75)
    noise /= noise.sum()
    noise_t = torch.tensor(noise, device=device, dtype=torch.float32)
    in_emb = torch.nn.Embedding(V, dim).to(device)
    out_emb = torch.nn.Embedding(V, dim).to(device)
    scale = 0.5 / dim
    torch.nn.init.uniform_(in_emb.weight, -scale, scale)
    torch.nn.init.uniform_(out_emb.weight, -scale, scale)
    opt = torch.optim.Adam(
        list(in_emb.parameters()) + list(out_emb.parameters()), lr=lr)
    # int32 ids + token chunks so fil9 (~124M) never materializes all pairs
    ids_t = torch.tensor(ids, device=device, dtype=torch.int32)
    ntok = ids_t.shape[0]
    chunk = 1 << 21
    est_pairs = 2 * sum(ntok - d for d in range(1, window + 1))
    pbar = tqdm(total=epochs * ((est_pairs + batch - 1) // batch),
                desc="Word2Vec", unit="batch", mininterval=1.0,
                dynamic_ncols=True)
    for ep in range(epochs):
        total, nb = 0.0, 0
        for d in range(1, window + 1):
            for s in range(0, ntok - d, chunk):
                e = min(s + chunk, ntok - d)
                a = ids_t[s:e].long()
                b = ids_t[s + d:e + d].long()
                m = a != b
                a, b = a[m], b[m]
                ctr = torch.cat([a, b])
                ctx = torch.cat([b, a])
                n = ctr.shape[0]
                if n == 0:
                    continue
                perm = torch.randperm(n, device=device)
                for start in range(0, n, batch):
                    ix = perm[start:start + batch]
                    c, o = ctr[ix], ctx[ix]
                    B = c.shape[0]
                    v_c = in_emb(c)
                    pos = (v_c * out_emb(o)).sum(1)
                    neg = torch.multinomial(noise_t, B * n_neg,
                                            replacement=True)
                    v_n = out_emb(neg.view(B, n_neg))
                    neg_s = (v_c.unsqueeze(1) * v_n).sum(2)
                    loss = (-F.logsigmoid(pos).mean()
                            - F.logsigmoid(-neg_s).mean())
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total += float(loss.detach())
                    nb += 1
                    pbar.set_postfix(ep=f"{ep + 1}/{epochs}", d=d,
                                     loss=f"{total / max(nb, 1):.3g}",
                                     refresh=False)
                    pbar.update(1)
                del a, b, m, ctr, ctx, perm
        tqdm.write(f"  epoch {ep + 1}/{epochs}  loss {total / max(nb, 1):.4g}  "
                   f"(streamed {ntok:,} tokens)")
    pbar.close()
    return in_emb.weight.detach().float().cpu().numpy()


def subsystem_pairs(ids, vocab, targets, window):
    """Windowed pairs among the 8 cube words in original token positions."""
    inv = {vocab[w]: k for k, w in enumerate(targets)}
    keep = np.array(sorted(inv), dtype=np.int32)
    pos = np.flatnonzero(np.isin(ids, keep)).astype(np.int32)
    sub = np.array([inv[int(ids[i])] for i in pos], dtype=np.int32)
    left, right = [], []
    n = len(pos)
    for i in tqdm(range(n), desc="subsystem pairs", unit="mention",
                   mininterval=1.0, dynamic_ncols=True):
        j = i + 1
        while j < n and pos[j] - pos[i] <= window:
            a, b = int(sub[i]), int(sub[j])
            if a != b:
                left.append(a)
                right.append(b)
                left.append(b)
                right.append(a)
            j += 1
    ctr = np.asarray(left, dtype=np.int32)
    ctx = np.asarray(right, dtype=np.int32)
    print(f"subsystem: {len(pos):,} mentions of 8 words, "
          f"{len(ctr):,} directed window pairs")
    return sub, ctr, ctx


def train_word2vec_pairs(ctr, ctx, counts, V, dim, epochs, seed, n_neg=5,
                         batch=8192, lr=0.01):
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_neg = min(n_neg, max(V - 1, 1))
    print(f"Word2Vec skip-gram subsystem (torch {device}): {len(ctr):,} pairs, "
          f"V={V}, dim {dim}, {epochs} epochs, seed {seed}")
    torch.manual_seed(seed)
    noise = np.power(np.maximum(counts, 1.0), 0.75)
    noise /= noise.sum()
    noise_t = torch.tensor(noise, device=device, dtype=torch.float32)
    in_emb = torch.nn.Embedding(V, dim).to(device)
    out_emb = torch.nn.Embedding(V, dim).to(device)
    scale = 0.5 / dim
    torch.nn.init.uniform_(in_emb.weight, -scale, scale)
    torch.nn.init.uniform_(out_emb.weight, -scale, scale)
    opt = torch.optim.Adam(
        list(in_emb.parameters()) + list(out_emb.parameters()), lr=lr)
    ctr_t = torch.tensor(ctr, device=device, dtype=torch.long)
    ctx_t = torch.tensor(ctx, device=device, dtype=torch.long)
    n = ctr_t.shape[0]
    batches_ep = (n + batch - 1) // batch
    pbar = tqdm(total=epochs * batches_ep, desc="Word2Vec", unit="batch",
                mininterval=1.0, dynamic_ncols=True)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total, nb = 0.0, 0
        for start in range(0, n, batch):
            ix = perm[start:start + batch]
            c, o = ctr_t[ix], ctx_t[ix]
            B = c.shape[0]
            v_c = in_emb(c)
            pos = (v_c * out_emb(o)).sum(1)
            neg = torch.multinomial(noise_t, B * n_neg, replacement=True)
            v_n = out_emb(neg.view(B, n_neg))
            neg_s = (v_c.unsqueeze(1) * v_n).sum(2)
            loss = (-F.logsigmoid(pos).mean()
                    - F.logsigmoid(-neg_s).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
            pbar.set_postfix(ep=f"{ep + 1}/{epochs}",
                             loss=f"{total / max(nb, 1):.3g}",
                             refresh=False)
            pbar.update(1)
        tqdm.write(f"  epoch {ep + 1}/{epochs}  loss {total / max(nb, 1):.4g}")
    pbar.close()
    return in_emb.weight.detach().float().cpu().numpy()


def build_glove_cooc(ids, V, window, cache=None):
    if cache is not None and cache.is_file():
        t0 = time.perf_counter()
        z = np.load(cache)
        print(f"loaded GloVe co-occurrence {cache.name}: "
              f"{len(z['ii']):,} directed pairs  "
              f"({time.perf_counter() - t0:.1f}s)")
        return z["ii"], z["jj"], z["x"]
    t0 = time.perf_counter()
    ii_p, jj_p, xx_p = [], [], []
    for d in tqdm(range(1, window + 1), desc="GloVe co-occurrence",
                   unit="offset", mininterval=0.2, dynamic_ncols=True):
        a, b = ids[:-d], ids[d:]
        m = a != b
        key = a[m].astype(np.int64) * V + b[m].astype(np.int64)
        u, c = np.unique(key, return_counts=True)
        ii_p.append((u // V).astype(np.int64))
        jj_p.append((u % V).astype(np.int64))
        xx_p.append(c.astype(np.float32))
    ii_u = np.concatenate(ii_p)
    jj_u = np.concatenate(jj_p)
    x_u = np.concatenate(xx_p)
    key = ii_u * V + jj_u
    uniq, inv = np.unique(key, return_inverse=True)
    x_one = np.zeros(len(uniq), dtype=np.float32)
    np.add.at(x_one, inv, x_u)
    ii = np.concatenate([uniq // V, uniq % V]).astype(np.int64)
    jj = np.concatenate([uniq % V, uniq // V]).astype(np.int64)
    x = np.concatenate([x_one, x_one])
    if cache is not None:
        np.savez(cache, ii=ii, jj=jj, x=x)
        print(f"cached GloVe co-occurrence -> {cache.name}")
    print(f"GloVe co-occurrence: {len(ii):,} directed pairs  "
          f"({time.perf_counter() - t0:.1f}s)")
    return ii, jj, x


def train_glove(ii_np, jj_np, x_np, V, dim, seed, xmax=100.0, alpha=0.75,
                lr=0.05, max_iter=200, tol=1e-3, patience=5, ckpt=None):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GloVe (torch {device}): V={V}, {len(ii_np):,} directed pairs, "
          f"dim {dim}, max_iter {max_iter}, tol {tol:g}, seed {seed}")
    torch.manual_seed(seed)
    ii = torch.as_tensor(ii_np, dtype=torch.long)
    jj = torch.as_tensor(jj_np, dtype=torch.long)
    x_cpu = torch.as_tensor(x_np)
    f = torch.clamp((x_cpu / xmax) ** alpha, max=1.0)
    logx = torch.log(x_cpu)
    scale = 0.5 / dim
    start = 0
    if ckpt is not None and ckpt.is_file():
        state = torch.load(ckpt, map_location=device)
        W, Wt = state["W"], state["Wt"]
        b, bt = state["b"], state["bt"]
        gW, gWt = state["gW"], state["gWt"]
        gb, gbt = state["gb"], state["gbt"]
        start = int(state["iter"])
        print(f"  resumed GloVe checkpoint at iter {start}")
    else:
        W = torch.empty(V, dim, device=device).uniform_(-scale, scale)
        Wt = torch.empty(V, dim, device=device).uniform_(-scale, scale)
        b = torch.zeros(V, device=device)
        bt = torch.zeros(V, device=device)
        gW = torch.ones_like(W)
        gWt = torch.ones_like(Wt)
        gb = torch.ones_like(b)
        gbt = torch.ones_like(bt)
    pair_batch = 1 << 20
    P = ii.shape[0]
    sse_prev = None
    flat = 0
    t0 = time.perf_counter()
    last = start
    pbar = tqdm(range(start, max_iter), desc="GloVe", unit="iter",
                initial=start, total=max_iter, mininterval=0.5,
                dynamic_ncols=True)
    for it in pbar:
        dW = torch.zeros_like(W)
        dWt = torch.zeros_like(Wt)
        db = torch.zeros_like(b)
        dbt = torch.zeros_like(bt)
        sse = 0.0
        for start_p in range(0, P, pair_batch):
            sl = slice(start_p, min(start_p + pair_batch, P))
            iib = ii[sl].to(device, non_blocking=True)
            jjb = jj[sl].to(device, non_blocking=True)
            fb = f[sl].to(device, non_blocking=True)
            logxb = logx[sl].to(device, non_blocking=True)
            pred = (W[iib] * Wt[jjb]).sum(1) + b[iib] + bt[jjb]
            diff = pred - logxb
            coef = 2.0 * fb * diff
            dW.index_add_(0, iib, coef[:, None] * Wt[jjb])
            dWt.index_add_(0, jjb, coef[:, None] * W[iib])
            db.index_add_(0, iib, coef)
            dbt.index_add_(0, jjb, coef)
            sse += float((fb * diff.square()).sum())
        gW += dW.square()
        gWt += dWt.square()
        gb += db.square()
        gbt += dbt.square()
        W -= lr * dW / gW.sqrt()
        Wt -= lr * dWt / gWt.sqrt()
        b -= lr * db / gb.sqrt()
        bt -= lr * dbt / gbt.sqrt()
        rel = (abs(sse_prev - sse) / max(sse_prev, 1e-12)
               if sse_prev is not None else 1.0)
        sse_prev = sse
        last = it + 1
        pbar.set_postfix(sse=f"{sse:.4g}", rel=f"{rel:.3g}", refresh=False)
        if last <= 5 or last % 5 == 0:
            tqdm.write(f"  iter {last}/{max_iter}  weighted SSE {sse:.4g}  "
                       f"rel {rel:.3g}")
        if ckpt is not None and last % 10 == 0:
            torch.save(dict(W=W, Wt=Wt, b=b, bt=bt, gW=gW, gWt=gWt,
                            gb=gb, gbt=gbt, iter=last), ckpt)
        if rel < tol:
            flat += 1
            if flat >= patience:
                tqdm.write(f"  GloVe converged at iter {last}  "
                           f"(rel SSE drop < {tol:g} for {patience} steps)")
                break
        else:
            flat = 0
    else:
        tqdm.write(f"  GloVe hit max_iter={max_iter}  (last rel {rel:.3g})")
    pbar.close()
    if ckpt is not None:
        torch.save(dict(W=W, Wt=Wt, b=b, bt=bt, gW=gW, gWt=gWt,
                        gb=gb, gbt=gbt, iter=last), ckpt)
    print(f"  GloVe wall {time.perf_counter() - t0:.1f}s")
    return (W + Wt).detach().float().cpu().numpy()


def overlay_pack(name, X, pts):
    M = gram_from_vectors(X)
    _, r2 = reynolds(M, pts)
    E, Th, names, ovls = match_walsh(M, pts)
    print(f"{name} invariance R^2 = {r2:.3f}")
    print(f"{name} matched characters:", names)
    print(f"{name} overlaps:", [round(float(v), 3) for v in ovls])
    return E, Th, names, ovls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=_DEFAULT_TEXT8)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=10,
                    help="Word2Vec skip-gram epochs")
    ap.add_argument("--glove-max-iter", type=int, default=200,
                    help="GloVe AdaGrad cap; stops early on SSE plateau")
    ap.add_argument("--glove-tol", type=float, default=1e-3,
                    help="relative SSE drop for GloVe convergence")
    ap.add_argument("--glove-patience", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-word2vec", action="store_true",
                    help="reuse cached Word2Vec vectors (error if missing)")
    ap.add_argument("--subsystem-only", action="store_true",
                    help="train 8 cube-word embeddings only (windowed "
                         "pairs among those words, original token positions)")
    ap.add_argument("--pretrained", action="store_true",
                    help="use released Word2Vec (Google News 300d) and "
                         "GloVe (Wikipedia+Gigaword 6B 300d)")
    ap.add_argument("--out", default="cube_wordvecs")
    args = ap.parse_args()
    ctag = _corpus_tag(args.text)
    if args.out == "cube_wordvecs":
        if args.pretrained:
            args.out = "cube_wordvecs_pretrained"
        elif args.subsystem_only:
            args.out = "cube_wordvecs_subsystem" + ctag
        elif ctag:
            args.out = "cube_wordvecs" + ctag

    order = list(WORDS)
    targets = [WORDS[p] for p in order]
    pts = np.array(order)
    if args.pretrained:
        Xw8 = load_pretrained_word2vec(targets)
        Xg8 = load_pretrained_glove(targets)
    else:
        paths = _cache_paths(args.min_count, args.window, args.dim, args.seed,
                             subsystem=args.subsystem_only, corpus=ctag)
        vocab, ids = encode_corpus(args.text, targets, args.min_count,
                                   cache=paths["corpus"])
        if args.subsystem_only:
            sub, ctr, ctx = subsystem_pairs(ids, vocab, targets, args.window)
            V = 8
            counts = np.bincount(sub, minlength=V).astype(np.float64)
            if args.skip_word2vec or paths["w2v"].is_file():
                Xw = np.load(paths["w2v"])
                print(f"loaded Word2Vec {paths['w2v'].name}  shape {Xw.shape}")
            else:
                Xw = train_word2vec_pairs(ctr, ctx, counts, V, args.dim,
                                          args.epochs, args.seed)
                np.save(paths["w2v"], Xw)
                print(f"cached Word2Vec -> {paths['w2v'].name}")
            if paths["glove"].is_file():
                Xg = np.load(paths["glove"])
                print(f"loaded GloVe {paths['glove'].name}  shape {Xg.shape}")
            else:
                key = ctr.astype(np.int64) * V + ctx.astype(np.int64)
                uniq, cnt = np.unique(key, return_counts=True)
                ii = (uniq // V).astype(np.int64)
                jj = (uniq % V).astype(np.int64)
                x = cnt.astype(np.float32)
                Xg = train_glove(ii, jj, x, V, args.dim, args.seed,
                                 max_iter=args.glove_max_iter,
                                 tol=args.glove_tol,
                                 patience=args.glove_patience,
                                 ckpt=paths["glove_ckpt"])
                np.save(paths["glove"], Xg)
                print(f"cached GloVe -> {paths['glove'].name}")
            Xw8, Xg8 = Xw, Xg
        else:
            V = len(vocab)
            if args.skip_word2vec or paths["w2v"].is_file():
                if not paths["w2v"].is_file():
                    raise SystemExit(f"no cached Word2Vec at {paths['w2v']}")
                Xw = np.load(paths["w2v"])
                print(f"loaded Word2Vec {paths['w2v'].name}  shape {Xw.shape}")
            else:
                t0 = time.perf_counter()
                Xw = train_word2vec(ids, V, args.window, args.dim, args.epochs,
                                    args.seed)
                np.save(paths["w2v"], Xw)
                print(f"cached Word2Vec -> {paths['w2v'].name}  "
                      f"({time.perf_counter() - t0:.1f}s)")
            if paths["glove"].is_file():
                Xg = np.load(paths["glove"])
                print(f"loaded GloVe {paths['glove'].name}  shape {Xg.shape}")
            else:
                ii, jj, x = build_glove_cooc(ids, V, args.window,
                                             cache=paths["cooc"])
                Xg = train_glove(ii, jj, x, V, args.dim, args.seed,
                                 max_iter=args.glove_max_iter,
                                 tol=args.glove_tol,
                                 patience=args.glove_patience,
                                 ckpt=paths["glove_ckpt"])
                np.save(paths["glove"], Xg)
                print(f"cached GloVe -> {paths['glove'].name}")
            Xw8, Xg8 = (cube_rows(vocab, Xw, targets),
                        cube_rows(vocab, Xg, targets))

    Ew, Thw, nw, ow = overlay_pack("Word2Vec", Xw8, pts)
    Eg, Thg, ng, og = overlay_pack("GloVe", Xg8, pts)
    nw_s = [subset_label(nm) for nm in nw]
    ng_s = [subset_label(nm) for nm in ng]

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "font.size": 20, "axes.titlesize": 24, "axes.labelsize": 22,
        "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
    })
    tick_fs, title_fs = 20, 24
    cmap = plt.cm.RdBu_r
    vm = max(float(np.max(np.abs(np.stack([Ew, Thw, Eg, Thg])))), 1e-12)
    norm = Normalize(vmin=-vm, vmax=vm)
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 6.4))
    if args.subsystem_only and not args.pretrained:
        t_w2v, t_glove = "Word2Vec (8 words)", "GloVe (8 words)"
    else:
        t_w2v, t_glove = "Word2Vec", "GloVe"
    draw_overlay(ax[0], Ew, Thw, nw_s, ow, targets, norm, cmap, t_w2v,
                 xlabel=False, show_overlap=False,
                 tick_fs=tick_fs, title_fs=title_fs, xtick_fs=20, xrot=45)
    draw_overlay(ax[1], Eg, Thg, ng_s, og, targets, norm, cmap, t_glove,
                 ytick=False, xlabel=False, show_overlap=False,
                 tick_fs=tick_fs, title_fs=title_fs, xtick_fs=20, xrot=45)
    fig.subplots_adjust(left=0.12, right=0.84, top=0.90, bottom=0.28,
                        wspace=0.08)
    fig.canvas.draw()
    p0, p1 = ax[0].get_position(), ax[1].get_position()
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([p1.x1 + 0.02, p1.y0, 0.02, p1.height])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=20)
    h_emp = Line2D([0], [0], marker="s", color="none",
                   markerfacecolor="0.35", markeredgecolor="none", ms=16)
    h_th = Line2D([0], [0], marker="s", color="none",
                  markerfacecolor="none", markeredgecolor="0.35",
                  markeredgewidth=2.6, ms=18)
    fig.legend([h_emp, h_th], ["empirical", "theory"],
               loc="upper center", ncol=2, fontsize=22, frameon=False,
               bbox_to_anchor=(0.5 * (p0.x0 + p1.x1), p0.y0 - 0.14),
               bbox_transform=fig.transFigure)
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=170, bbox_inches="tight")
    print(f"saved -> {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
