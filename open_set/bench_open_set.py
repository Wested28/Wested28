"""
bench_open_set.py -- synthetic test of the open-set rejection methods in
open_set.py against a plain absolute distance threshold.

Synthetic world (embeddings on a unit sphere, distance = (1 - cos) / 2,
SMALL = same character):
  references : 17 classes, sizes 10..300 (~2,200 refs), each class with its
               own spread; 2% of references are OFF-MODEL (placed at random,
               still carrying the class label) -- as in the real roster.
  in-set     : queries from the 17 classes; 10% drawn at 1.8x the class
               spread (wide shots, odd angles -- far from their own class).
  out of set :
    unseen     characters not in the reference set (new centres)
    lookalike  unseen characters whose centre sits near a known class
    generic    "no character" frames: weakly aligned with the mean of all
               class centres (hub-like), the backgrounds/UI case
    pattern    random directions (kaleidoscopes)
    near_odd   a query placed right next to one OFF-MODEL reference -- close
               to ONE ref, far from that ref's class (the stated hard case)

Two comparisons:
  A. calibrated from references only (calibrate(), the realistic path) at
     target_accept 0.80 / 0.90 / 0.95 / 0.99
  B. matched: each method's threshold set (with the answers) so the in-set
     false-reject rate is exactly 5 / 10 / 20 %, then OOS recall compared --
     pure separability, no calibration error
Five seeds; mean +- sd.

usage: python bench_open_set.py [n_seeds=5] [noise_scale=2.6]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import open_set as OS  # noqa: E402

DIM = 48
K = 5  # k for knn / recip; see sensitivity.py
ODD = 0.02  # share of references that are off-model
NS = 2.6  # noise scale; tuned so in-set NN accuracy is ~89%, as measured (88.8%)
METHODS = ["abs", "knn", "ratio", "class", "recip", "abs+ratio", "knn+ratio", "knn+class"]
OOS_KINDS = ["unseen", "lookalike", "generic", "pattern", "near_odd"]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def world(seed):
    rng = np.random.default_rng(seed)
    C = 17
    centres = unit(rng.normal(size=(C, DIM)))
    spread = rng.uniform(0.35, 0.8, C)                # per-character variability
    sizes = np.clip(np.round(rng.lognormal(4.6, 0.8, C)), 10, 300).astype(int)
    R, rl, odd = [], [], []
    for c in range(C):
        pts = unit(centres[c] + spread[c] * rng.normal(size=(sizes[c], DIM)) / np.sqrt(DIM) * NS)
        n_odd = max(0, int(round(ODD * sizes[c])))
        if n_odd:
            j = rng.choice(sizes[c], n_odd, replace=False)
            pts[j] = unit(rng.normal(size=(n_odd, DIM)))
            odd += [len(R) + jj for jj in j]
        R += list(pts); rl += [c] * sizes[c]
    R = np.array(R); rl = np.array(rl)

    def draw(c, s, n):
        return unit(centres[c] + s * rng.normal(size=(n, DIM)) / np.sqrt(DIM) * NS)

    Q, ql, kind = [], [], []
    for _ in range(1500):                              # in-set
        c = rng.choice(C, p=sizes / sizes.sum())
        hard = rng.random() < 0.10
        Q.append(draw(c, spread[c] * (1.8 if hard else 1.0), 1)[0]); ql.append(c)
        kind.append("in_hard" if hard else "in")
    newc = unit(rng.normal(size=(25, DIM)))
    look = unit(centres[rng.integers(0, C, 8)] + 0.55 * rng.normal(size=(8, DIM)) / np.sqrt(DIM) * NS)
    hub = unit(centres.mean(0))
    for k, n in zip(OOS_KINDS, [400, 300, 400, 200, 200]):
        for _ in range(n):
            if k == "unseen":
                v = draw_c(newc[rng.integers(25)], rng.uniform(0.35, 0.8), rng)
            elif k == "lookalike":
                v = draw_c(look[rng.integers(8)], rng.uniform(0.35, 0.8), rng)
            elif k == "generic":
                v = unit(0.9 * hub + rng.normal(size=DIM) / np.sqrt(DIM) * 1.3)
            elif k == "pattern":
                v = unit(rng.normal(size=DIM))
            else:
                v = unit(R[rng.choice(odd) if odd else rng.integers(len(R))] + 0.25 * rng.normal(size=DIM) / np.sqrt(DIM) * NS)
            Q.append(v); ql.append(-1); kind.append(k)
    Q = np.array(Q)
    dist = lambda A, B: ((1 - A @ B.T) / 2).astype(np.float32)
    names = np.array([f"C{c:02d}" for c in range(C)])
    truth = np.array([names[c] if c >= 0 else None for c in ql], dtype=object)
    return dist(Q, R), dist(R, R), names[rl], truth, np.array(kind)


def draw_c(centre, s, rng):
    return unit(centre + s * rng.normal(size=DIM) / np.sqrt(DIM) * NS)


def method_scores(D_qr, D_rr, rl, k=K):
    """Raw oddness per method (thresholds from a 0.95 calibration only used
    to normalise combinations)."""
    p = OS.calibrate(D_rr, rl, target_accept=0.95, k=k)
    out = {}
    for m in METHODS:
        _, s, _ = OS.classify_open_set(D_qr, rl, **{**p, "method": m})
        out[m] = s
    return out, p


def auroc(score, positive):
    """P(score of a random positive > random negative); ties count half."""
    r = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort").astype(float)
    # average ranks for ties
    s_sorted = np.sort(score)
    _, inv, cnt = np.unique(s_sorted, return_inverse=True, return_counts=True)
    starts = np.cumsum(cnt) - cnt
    avg = starts + (cnt - 1) / 2
    ranks = avg[np.searchsorted(np.unique(s_sorted), score)]
    n1, n0 = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def run(seed):
    D_qr, D_rr, rl, truth, kind = world(seed)
    oos = np.array([t is None for t in truth])
    res = {"A": {}, "B": {}, "kinds": {}, "auroc": {}, "nn_acc": None}
    # A: reference-only calibration
    for a in (0.80, 0.90, 0.95, 0.99):
        p = OS.calibrate(D_rr, rl, target_accept=a, k=K)
        for m in METHODS:
            names, _, rej = OS.classify_open_set(D_qr, rl, **{**p, "method": m})
            res["A"][(a, m)] = (
                (rej & oos).sum() / max(rej.sum(), 1),               # reject precision
                (rej & oos).sum() / oos.sum(),                       # reject recall
                (rej & ~oos).sum() / (~oos).sum(),                   # in-set false reject
                ((~rej) & ~oos & (names == truth)).sum() / max((~rej).sum(), 1),  # accepted precision
            )
    # B: matched in-set false-reject (oracle thresholds)
    sc, p95 = method_scores(D_qr, D_rr, rl)
    names_all, _, _ = OS.classify_open_set(D_qr, rl, **{**p95, "method": "abs"})
    res["nn_acc"] = (names_all[~oos] == truth[~oos]).mean()
    for m, s in sc.items():
        res["auroc"][m] = auroc(s, oos)
        for fr in (0.05, 0.10, 0.20):
            t = np.quantile(s[~oos], 1 - fr)
            rej = s > t
            res["B"][(fr, m)] = ((rej & oos).sum() / oos.sum(),
                                 ((~rej) & ~oos & (names_all == truth)).sum() / max((~rej).sum(), 1))
            if fr == 0.10:
                for k in OOS_KINDS + ["in_hard"]:
                    mk = kind == k
                    res["kinds"][(m, k)] = rej[mk].mean()
    return res


def main():
    global NS
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    if len(sys.argv) > 2:
        NS = float(sys.argv[2])
    runs = [run(s) for s in range(seeds)]
    ms = lambda vals: f"{100 * np.mean(vals):5.1f} ±{100 * np.std(vals):4.1f}"
    print(f"# Open-set rejection benchmark ({seeds} seeds; 1,500 in-set + 1,500 out-of-set queries "
          f"vs ~2,200 refs / 17 classes; noise scale {NS})\n")
    print(f"plain nearest-neighbour naming accuracy on in-set queries: "
          f"{ms([r['nn_acc'] for r in runs])}%\n")
    print("## B. Separability at MATCHED in-set false-reject (oracle thresholds)\n")
    print("OOS rejected (%) when each method is allowed to wrongly reject the same share of in-set clips.\n")
    print("| method | AUROC | 5% FR | 10% FR | 20% FR |")
    print("|---|---:|---:|---:|---:|")
    for m in METHODS:
        print(f"| {m} | {np.mean([r['auroc'][m] for r in runs]):.3f} ±{np.std([r['auroc'][m] for r in runs]):.3f} | "
              + " | ".join(ms([r['B'][(fr, m)][0] for r in runs]) for fr in (0.05, 0.10, 0.20)) + " |")
    print("\n### Which out-of-set kinds each method catches (at 10% in-set false-reject)\n")
    print("| method | " + " | ".join(OOS_KINDS) + " | in_hard (false reject) |")
    print("|---|" + "---:|" * (len(OOS_KINDS) + 1))
    for m in METHODS:
        print(f"| {m} | " + " | ".join(ms([r['kinds'][(m, k)] for r in runs]) for k in OOS_KINDS + ["in_hard"]) + " |")
    print("\n## A. Calibrated from references only (the realistic path)\n")
    print("Thresholds from leave-one-out on the references at each target accept rate. "
          "Columns: reject precision / reject recall / in-set false-reject / precision of the names still proposed. "
          "Prevalence here is 50% out-of-set; reject precision scales with your real prevalence.\n")
    print("| target accept | method | rej. precision | rej. recall | in-set false rej. | accepted precision |")
    print("|---:|---|---:|---:|---:|---:|")
    for a in (0.80, 0.90, 0.95, 0.99):
        for m in METHODS:
            v = [r["A"][(a, m)] for r in runs]
            print(f"| {a:.2f} | {m} | " + " | ".join(ms([x[i] for x in v]) for i in range(4)) + " |")


if __name__ == "__main__":
    main()
