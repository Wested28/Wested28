"""
open_set_v2.py -- abstention for nearest-neighbour character naming when most
characters have very few references (long tail). numpy + stdlib only.

D_query_ref : (n_query, n_ref) CCIP distances, SMALL = same character.
              Rows can be frames; aggregate_clips() rolls them up to clips
              with the same 3/3 agreement rule the proposal run uses.

Scores (each "doubt": higher = less trustworthy). None of them needs a
populated within-class distribution, which is the point -- 75 of 92
characters have under 10 references and some have one.

  abs    d1                                  plain distance to the nearest ref
  ratio  d1 / d2   d2 = nearest ref of a DIFFERENT character (Lowe)
  local  d1 / sigma(j*)   sigma(j) = distance from the nearest ref j* to its
         k-th nearest OTHER ref of ANY character: "is the query inside that
         reference's own neighbourhood". Works for a 1-clip character.
  vote   a GATE, not a ranking: the k nearest refs must all carry the
         predicted name (share >= vote_min, default 1.0 = unanimous). A
         1- or 2-clip character can never be unanimous at k=3, so it always
         goes to a human -- deliberately. As a ranking it is useless (three
         possible values, all ties), so "vote" only ever appears as "+vote".
  knn    mean of the k nearest refs of the predicted character, k capped at
         the character's size. Included because it was the earlier
         recommendation: on a long tail its meaning changes per character.

Combinations: "abs+ratio" rejects when either part does, with a joint
threshold calibrated so the combination hits target_accept. "+vote" adds the
unanimity gate on top of whatever it is combined with; the gate is
structural (k_vote, vote_min), not calibrated.

    p = calibrate(D_ref_ref, ref_labels, target_accept=0.95, method="abs+vote")
    names, scores, reject = classify_open_set(D_query_ref, ref_labels, **p)
"""
import numpy as np

SINGLE = ("abs", "ratio", "local", "knn")   # rankings; "vote" is a gate added with "+vote"


def _classes(ref_labels):
    labels = np.asarray(ref_labels)
    names, inv = np.unique(labels, return_inverse=True)
    return labels, names, inv


def _sigma(D_rr, k):
    """distance from each ref to its k-th nearest OTHER ref, any character"""
    D = np.array(D_rr, dtype=np.float32, copy=True)
    np.fill_diagonal(D, np.inf)
    kk = min(k, D.shape[1] - 1)
    return np.partition(D, kk - 1, axis=1)[:, kk - 1]


def _loo_sigma_rows(D_loo, k):
    """Leave-one-out sigma for each reference-as-query: the k-th nearest
    neighbour of its nearest ref j*, NOT counting the held-out ref itself.
    Using plain sigma here leaks the held-out clip into its own yardstick
    (it is often one of j*'s nearest), which piles scores up at exactly 1.0
    and makes calibration accept more than it says."""
    n = D_loo.shape[0]
    kk = min(k + 1, n - 1)
    idx = np.argpartition(D_loo, kk - 1, axis=1)[:, :kk]
    dd = np.take_along_axis(D_loo, idx, 1)
    o = np.argsort(dd, 1)
    idx, dd = np.take_along_axis(idx, o, 1), np.take_along_axis(dd, o, 1)
    j = D_loo.argmin(1)
    nb, nd = idx[j], dd[j]
    kq = min(k, kk)
    holds_me = (nb[:, :kq] == np.arange(n)[:, None]).any(1)
    return np.where(holds_me, nd[:, min(kq, kk - 1)], nd[:, kq - 1])


def raw_scores(D_query_ref, ref_labels, k_vote=3, k_local=3, k_knn=5, sigma=None, sigma_rows=None):
    """All doubt scores. Returns (pred_class_index, class_names, dict)."""
    D = np.asarray(D_query_ref, dtype=np.float32)
    labels, names, inv = _classes(ref_labels)
    n, C = D.shape[0], len(names)
    rows = np.arange(n)
    # per-class min distance and argmin -- loop over classes (92), vectorised over rows
    mn = np.full((n, C), np.inf, np.float32)
    arg = np.zeros((n, C), np.int64)
    kn = np.full((n, C), np.inf, np.float32)
    for c in range(C):
        cols = np.where(inv == c)[0]
        sub = D[:, cols]
        a = sub.argmin(axis=1)
        arg[:, c] = cols[a]
        mn[:, c] = sub[rows, a]
        kk = min(k_knn, len(cols))
        part = np.partition(sub, kk - 1, axis=1)[:, :kk] if kk < len(cols) else sub
        fin = np.isfinite(part)
        kn[:, c] = np.where(fin.any(1), np.where(fin, part, 0).sum(1) / np.maximum(fin.sum(1), 1), np.inf)
    two = np.argpartition(mn, 1, axis=1)[:, :2] if C > 1 else np.zeros((n, 2), np.int64)
    first = np.where(mn[rows, two[:, 0]] <= mn[rows, two[:, 1]], two[:, 0], two[:, 1])
    second = np.where(first == two[:, 0], two[:, 1], two[:, 0])
    d1 = mn[rows, first]
    d2 = mn[rows, second] if C > 1 else np.full(n, np.inf, np.float32)
    # top-k vote over individual references
    kv = min(k_vote, D.shape[1])
    top = np.argpartition(D, kv - 1, axis=1)[:, :kv]
    vote = (inv[top] == first[:, None]).mean(axis=1)
    s = {"abs": d1, "ratio": d1 / np.maximum(d2, 1e-9), "vote_share": vote,
         "knn": kn[rows, first]}
    if sigma_rows is not None:
        s["local"] = d1 / np.maximum(np.asarray(sigma_rows, np.float32), 1e-9)
    elif sigma is not None:
        s["local"] = d1 / np.maximum(np.asarray(sigma, np.float32)[arg[rows, first]], 1e-9)
    return first, names, s


def _combine(s, thr, parts, vote_min=1.0):
    rank = [p for p in parts if p != "vote"]
    if not rank:
        raise ValueError('"vote" is a gate: combine it, e.g. "abs+vote"')
    score = np.max(np.stack([s[p] / max(thr[p], 1e-12) for p in rank]), axis=0)
    if "vote" in parts:
        score = np.where(s["vote_share"] + 1e-9 >= vote_min, score, np.inf)
    return score


def calibrate(D_ref_ref, ref_labels, target_accept=0.95, method="abs+vote",
              k_vote=3, k_local=3, k_knn=5, vote_min=1.0):
    """Leave-one-out on the references: every reference is scored against all
    the OTHER references, and each threshold is put where `target_accept` of
    them would be accepted. That fixes the false-reject side only."""
    D = np.array(D_ref_ref, dtype=np.float32, copy=True)
    np.fill_diagonal(D, np.inf)
    labels, names, inv = _classes(ref_labels)
    sigma = _sigma(D_ref_ref, k_local)
    _, _, s = raw_scores(D, labels, k_vote, k_local, k_knn, sigma, sigma_rows=_loo_sigma_rows(D, k_local))
    thr = {m: float(np.quantile(v[np.isfinite(v)], target_accept)) for m, v in s.items() if m != "vote_share"}
    rank = [p for p in method.split("+") if p != "vote"]
    if len(rank) > 1:
        comb = _combine(s, thr, rank)
        thr[method] = float(np.quantile(comb[np.isfinite(comb)], target_accept))
    gate_pass = float((s["vote_share"] + 1e-9 >= vote_min).mean())
    return {"method": method, "thresholds": thr, "sigma": sigma, "k_vote": k_vote,
            "k_local": k_local, "k_knn": k_knn, "vote_min": vote_min,
            "target_accept": target_accept, "loo_vote_gate_pass": gate_pass}


def classify_open_set(D_query_ref, ref_labels, method="abs+vote", thresholds=None, sigma=None,
                      k_vote=3, k_local=3, k_knn=5, vote_min=1.0, **_ignored):
    """D_query_ref: (n_query, n_ref) distances, SMALL = same character.
    Returns (names, scores, reject_mask); reject_mask True means 'none of
    these reference characters' -- in practice: send to a human.
    scores <= 1 accepted, > 1 rejected (each part divided by its threshold).
    Pass the dict from calibrate() as **params."""
    if thresholds is None:
        raise ValueError("run calibrate() on the reference set first")
    first, names, s = raw_scores(D_query_ref, ref_labels, k_vote, k_local, k_knn, sigma)
    parts = method.split("+")
    for p in parts:
        if p != "vote" and p not in s:
            raise ValueError(f"unknown or uncalibrated method part {p!r}")
    score = _combine(s, thresholds, parts, vote_min)
    rank = [p for p in parts if p != "vote"]
    if len(rank) > 1:
        if method not in thresholds:
            raise ValueError(f"pass method={method!r} to calibrate()")
        score = score / max(thresholds[method], 1e-12)
    return names[first], score.astype(np.float32), score > 1.0


def aggregate_clips(clip_ids, names, reject, scores=None, need=3):
    """Frame rows -> clips. A clip is accepted only when at least `need` of
    its frames are accepted AND all accepted frames agree on the name (the
    proposal run's 3/3 rule with need=3). Returns dict clip -> (name or None,
    n_agree, worst_score)."""
    clip_ids = np.asarray(clip_ids)
    out = {}
    order = np.argsort(clip_ids, kind="stable")
    ids_sorted = clip_ids[order]
    starts = np.r_[0, np.flatnonzero(ids_sorted[1:] != ids_sorted[:-1]) + 1, len(order)]
    for a, b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]
        ok = ~np.asarray(reject)[idx]
        nm = np.asarray(names)[idx]
        worst = float(np.max(np.asarray(scores)[idx])) if scores is not None else np.nan
        if ok.sum() >= need and len(set(nm[ok].tolist())) == 1 and ok.all():
            out[ids_sorted[a]] = (nm[ok][0], int(ok.sum()), worst)
        else:
            vals, cnt = np.unique(nm, return_counts=True)
            out[ids_sorted[a]] = (None, int(cnt.max()), worst)
    return out


def tune_threshold(clip_scores, correct, target_precision=0.99, min_accepted=20):
    """The one setting references alone cannot give you: where to put the
    threshold on the REAL pile. Feed it clips a human has ruled -- e.g. the
    127 from the 3/3, d<=0.05 run -- as their worst-frame score (distance for
    "abs") and whether the proposed name was right. Returns the largest
    threshold whose accepted set still meets target_precision, with the
    precision and count it gives, or None if no threshold does.
    With 127 rulings and 9 errors, expect a coarse answer: re-run it as the
    ruled set grows."""
    s = np.asarray(clip_scores, dtype=np.float64)
    c = np.asarray(correct, dtype=bool)
    order = np.argsort(s, kind="stable")
    s, c = s[order], c[order]
    prec = np.cumsum(c) / np.arange(1, len(c) + 1)
    best = None
    for k in range(min_accepted - 1, len(s)):
        if k + 1 < len(s) and s[k + 1] == s[k]:
            continue                                      # only cut between distinct scores
        if prec[k] >= target_precision:
            best = (float(s[k]), float(prec[k]), k + 1)
    return best
