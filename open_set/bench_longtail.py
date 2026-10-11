"""
bench_longtail.py -- does any abstention method beat a tuned distance
threshold when most characters have 1-9 references? Synthetic world built to
the measured shape of the real pile.

World (per seed):
  reference set : 17 populated characters (10-200 clips) + 75 tail characters
                  (1-9 clips), ~1,033 clips x 3 frames
  query pile    : 2,875 clips x 3 frames: 50% populated characters, 20% tail
                  characters, 15% characters in NO reference set, 15% no
                  character at all
  look-alikes   : half the tail and half the unseen characters sit near a
                  populated character -- the measured failure mode
  distances     : (1-cos)/2 rescaled so the median query frame -> nearest
                  reference distance is 0.067, as measured

Conditions:
  A  reference set = the 17 populated characters only (what produced the
     99.3% -> 92.9% gap: tail characters are invisible to it)
  B  reference set = all 92 characters, tail included (1-clip ones too)

A clip is auto-accepted when its 3 frames agree on one name and its worst
frame passes; clips are ranked by worst-frame score, and we accept the most
confident 5 / 10 / 20 / 35 % of the pile. Precision = accepted clips whose
name is right. Baseline: the plain distance threshold (abs).

usage: python bench_longtail.py [n_seeds=5] [preset=shape|rule]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import open_set_v2 as OS  # noqa: E402

DIM, NS, FN = 48, 0.85, 0.9    # tuned: see the distance table at the top of the output
LA_TWIN, LA_LOOK, P_TWIN, P_LOOK = 0.08, 0.55, 0.45, 0.3   # look-alike structure of tail/unseen characters
SPREAD_SD = 0.65      # per-clip spread multiplier, lognormal sd -- widens the distance distribution
P_DUP = 0.05          # share of populated-character queries that are near-copies of a reference shot
DUP_NOISE = 0.5
METHODS = ["abs", "ratio", "local", "knn", "abs+vote", "knn+vote", "ratio+vote", "abs+ratio"]
COVER = (0.044, 0.10, 0.20)    # 0.044 = 127 of 2,875: the measured operating point
PRESETS = {
    # distances match the measured quantiles (p1..p99); harsher than the real pile
    "shape": dict(FN=0.9, P_TWIN=0.45),
    # the real proposal rule (3/3, d<=0.05) lands near the measured 92.9% precision;
    # the distance tail is shorter than measured (p99 ~0.12 vs 0.171)
    "rule": dict(FN=1.4, P_TWIN=0.2),
}


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def world(seed):
    rng = np.random.default_rng(seed)
    noise = lambda n=1: rng.normal(size=(n, DIM)) / np.sqrt(DIM)
    pop_n = np.clip(np.round(rng.lognormal(3.5, 0.75, 17)), 10, 200).astype(int)
    pop_n = np.maximum(10, np.round(pop_n * 800 / pop_n.sum())).astype(int)
    tail_n = np.clip(rng.geometric(0.33, 75), 1, 9)
    pop_c = unit(rng.normal(size=(17, DIM)))
    def near_or_new(n):
        out = []
        for _ in range(n):
            r = rng.random()
            if r < P_TWIN:
                out.append(unit(pop_c[rng.integers(17)] + LA_TWIN * NS * noise()[0]))   # same helmet, other wearer
            elif r < P_TWIN + P_LOOK:
                out.append(unit(pop_c[rng.integers(17)] + LA_LOOK * NS * noise()[0]))
            else:
                out.append(unit(rng.normal(size=DIM)))
        return np.array(out)
    tail_c, unseen_c = near_or_new(75), near_or_new(60)
    centres = np.vstack([pop_c, tail_c, unseen_c])          # 0-16 pop, 17-91 tail, 92-151 unseen
    spread = rng.uniform(0.35, 0.8, len(centres))

    def clips(c, n):
        m = rng.lognormal(0, SPREAD_SD, (n, 1))
        v = unit(centres[c] + spread[c] * m * NS * noise(n))
        f = unit(np.repeat(v, 3, 0) + FN * spread[c] * noise(3 * n))
        return f

    R, rl = [], []
    for c, n in list(enumerate(pop_n)) + [(17 + t, n) for t, n in enumerate(tail_n)]:
        R.append(clips(c, n)); rl += [c] * (3 * n)
    R = np.vstack(R); rl = np.array(rl)
    Q, qt, kind = [], [], []
    hub = unit(pop_c.mean(0))
    for _ in range(2875):
        r = rng.random()
        if r < 0.50:
            c = rng.choice(17, p=pop_n / pop_n.sum())
            if rng.random() < P_DUP:                       # near-copy of an existing reference shot
                j = rng.choice(np.where(rl == c)[0])
                Q.append(unit(np.repeat(R[j][None], 3, 0) + DUP_NOISE * spread[c] * noise(3)))
            else:
                Q.append(clips(c, 1))
            qt.append(c); kind.append("pop")
        elif r < 0.70:
            c = 17 + rng.integers(75); Q.append(clips(c, 1)); qt.append(c); kind.append("tail")
        elif r < 0.85:
            c = 92 + rng.integers(60); Q.append(clips(c, 1)); qt.append(-1); kind.append("unseen")
        else:
            v = unit(hub * 0.8 + 1.4 * NS * noise()[0]) if rng.random() < 0.6 else unit(rng.normal(size=DIM))
            Q.append(unit(np.repeat(v[None], 3, 0) + FN * 0.6 * noise(3))); qt.append(-1); kind.append("nochar")
    Q = np.vstack(Q)
    return R, rl, Q, np.array(qt), np.array(kind), pop_n, tail_n


def dist(A, B, scale):
    return ((1 - A @ B.T) / 2 * scale).astype(np.float32)


def evaluate(seed):
    R, rl, Q, qt, kind, pop_n, tail_n = world(seed)
    nq = len(qt)
    clip_of = np.repeat(np.arange(nq), 3)
    # scale so the median query-frame -> nearest-ref distance (all 92) is 0.067
    raw = dist(Q, R, 1.0)
    scale = 0.067 / np.median(raw.min(1))
    D_all = raw * scale
    R_dist = dist(R, R, scale)
    res = {"quantiles": np.quantile(D_all.min(1), [.01, .05, .10, .25, .50, .75, .90, .95, .99]),
           "max": float(D_all.min(1).max())}
    for cond, keep in (("A", rl < 17), ("B", np.ones(len(rl), bool))):
        Dq, Drr, lab = D_all[:, keep], R_dist[np.ix_(keep, keep)], rl[keep]
        p = OS.calibrate(Drr, lab, 0.95, method="abs+vote")
        for m in METHODS:
            p2 = OS.calibrate(Drr, lab, 0.95, method=m) if "+" in m else p
            names, sc, _ = OS.classify_open_set(Dq, lab, **{**p2, "method": m})
            names = names.reshape(nq, 3); sc = sc.reshape(nq, 3)
            agree = (names == names[:, :1]).all(1)
            worst = sc.max(1)
            correct = agree & (names[:, 0] == qt)
            ok_rank = np.where(agree, worst, np.inf)
            order = np.argsort(ok_rank, kind="stable")
            for cv in COVER:
                n = int(round(cv * nq))
                acc = order[:n]
                acc = acc[np.isfinite(ok_rank[acc])]
                res[(cond, m, cv)] = (correct[acc].mean() if len(acc) else np.nan,
                                      np.isin(kind[acc], ["tail", "unseen", "nochar"]).mean() if cond == "A"
                                      else np.isin(kind[acc], ["unseen", "nochar"]).mean())
        # the real proposal rule: 3/3 agreement and worst-frame distance <= 0.05
        nm_r, sc_r, _ = OS.classify_open_set(Dq, lab, **{**p, "method": "abs"})
        d1_r = (sc_r * p["thresholds"]["abs"]).reshape(nq, 3).max(1)
        nm_r = nm_r.reshape(nq, 3); ag_r = (nm_r == nm_r[:, :1]).all(1)
        acc_r = ag_r & (d1_r <= 0.05)
        res[(cond, "rule")] = (acc_r.mean(), (acc_r & (nm_r[:, 0] == qt)).sum() / max(acc_r.sum(), 1))
        # reproduce the measured gap with the plain threshold in condition A:
        # pick the threshold that gives >= 99.3% precision on LABELLED (pop) clips only
        names, sc, _ = OS.classify_open_set(Dq, lab, **{**p, "method": "abs"})
        d1 = (sc * p["thresholds"]["abs"]).reshape(nq, 3).max(1)
        names = names.reshape(nq, 3); agree = (names == names[:, :1]).all(1)
        correct = agree & (names[:, 0] == qt)
        best_t = None
        for t in np.quantile(d1, np.linspace(0.01, 0.6, 120)):
            m_lab = agree & (d1 <= t) & (kind == "pop")
            if m_lab.sum() >= 30 and correct[m_lab].mean() >= 0.993:
                best_t = t
        if best_t is not None:
            m_lab = agree & (d1 <= best_t) & (kind == "pop")
            m_all = agree & (d1 <= best_t)
            res[(cond, "gap")] = (best_t, correct[m_lab].mean(), correct[m_all].mean(), int(m_all.sum()))
    return res


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    preset = sys.argv[2] if len(sys.argv) > 2 else "shape"
    globals().update(PRESETS[preset])
    runs = [evaluate(s) for s in range(seeds)]
    f = lambda v: f"{100 * np.nanmean(v):5.1f} ±{100 * np.nanstd(v):4.1f}"
    q = np.mean([r["quantiles"] for r in runs], 0)
    print(f"# Long-tail open-set benchmark ({seeds} seeds, world preset '{preset}': {PRESETS[preset]})\n")
    print("Synthetic query-frame -> nearest-reference distance (target in brackets):\n")
    print("| p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |\n|---|---|---|---|---|---|---|---|---|---|")
    tgt = [.014, .025, .032, .048, .067, .094, .126, .144, .171]
    print("| " + " | ".join(f"{a:.3f} ({b:.3f})" for a, b in zip(q, tgt))
          + f" | {np.mean([r['max'] for r in runs]):.2f} (~0.30) |\n")
    for cond in ("A", "B"):
        g = [r.get((cond, "gap")) for r in runs if r.get((cond, "gap"))]
        if g:
            print(f"**Condition {cond}, plain threshold tuned to ≥99.3% on labelled (populated-only) clips:** "
                  f"threshold {np.mean([x[0] for x in g]):.3f}, labelled precision {f([x[1] for x in g])}%, "
                  f"**whole-pile precision {f([x[2] for x in g])}%** on {np.mean([x[3] for x in g]):.0f} accepted clips\n")
    for cond in ("A", "B"):
        rr = [r[(cond, "rule")] for r in runs]
        print(f"**Condition {cond}, the actual proposal rule (3/3 agree, d ≤ 0.05):** accepts "
              f"{100 * np.mean([x[0] for x in rr]):.1f}% of clips at **{f([x[1] for x in rr])}% precision**\n")
    for cond, title in (("A", "A — reference set = 17 populated characters (current)"),
                        ("B", "B — reference set = all 92 characters, tail included")):
        print(f"## {title}\n")
        print("Precision of auto-accepted clips (and share of them that are out-of-set) when accepting the most "
              "confident N% of the pile.\n")
        print("| method | " + " | ".join(f"{int(c * 100)}% accepted" for c in COVER) + " |")
        print("|---|" + "---:|" * len(COVER))
        for m in METHODS:
            print(f"| {m} | " + " | ".join(
                f"{f([r[(cond, m, c)][0] for r in runs])} (oos {100 * np.nanmean([r[(cond, m, c)][1] for r in runs]):.1f})"
                for c in COVER) + " |")
        print()


if __name__ == "__main__":
    main()
