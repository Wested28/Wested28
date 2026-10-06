"""
open_set.py -- "none of these characters" for nearest-neighbour naming over a
precomputed CCIP distance matrix (SMALL = same character). numpy + stdlib.

Typical use
    params = calibrate(D_ref_ref, ref_labels, target_accept=0.95)
    names, scores, reject = classify_open_set(D_query_ref, ref_labels, **params)

calibrate() needs only the reference set: it scores every reference clip
leave-one-out against the others and puts each threshold where
`target_accept` of the references would still be accepted. That fixes the
FALSE-REJECT side. It cannot tell you how many out-of-set clips get through:
that needs labelled negatives (clips ruled "no character", or named
characters outside the reference set). evaluate() measures it once you have
them.

Methods (each yields an "oddness" score, higher = more likely out of set)
    "abs"      d1                        absolute distance to the nearest ref
    "knn"      mean of the k nearest refs of the matched class
    "ratio"    d1 / d2,  d2 = nearest ref of a DIFFERENT class   (Lowe)
    "class"    d1 / tau_c,  tau_c = matched class's own LOO quantile
    "recip"    d1 / r_k(j*),  r_k = nearest ref j*'s distance to its k-th
               same-class neighbour (is the query inside j*'s neighbourhood?)
    "a+b"      combination: rejected if EITHER part rejects; score is the
               larger of the two calibrated ratios
"""
import numpy as np

SINGLE = ("abs", "knn", "ratio", "class", "recip")
COMBOS = ("abs+ratio", "knn+ratio", "knn+class")


def _classes(ref_labels):
    labels = np.asarray(ref_labels)
    names, inv = np.unique(labels, return_inverse=True)
    return labels, names, inv


def _per_class(D, inv, n_cls, k):
    """Per class: min distance (n, C) and mean of the k smallest (n, C).
    Vectorised over queries; loops over classes only (C is ~17)."""
    n = D.shape[0]
    mn = np.full((n, n_cls), np.inf, np.float32)
    kn = np.full((n, n_cls), np.inf, np.float32)
    arg = np.zeros((n, n_cls), np.int64)
    for c in range(n_cls):
        cols = np.where(inv == c)[0]
        sub = D[:, cols]
        a = sub.argmin(axis=1)
        arg[:, c] = cols[a]
        mn[:, c] = sub[np.arange(n), a]
        kk = min(k, len(cols))
        part = np.partition(sub, kk - 1, axis=1)[:, :kk] if kk < len(cols) else sub
        # mean of the finite ones (LOO puts inf on the diagonal); all-inf -> inf
        fin = np.isfinite(part)
        cnt = fin.sum(axis=1)
        tot = np.where(fin, part, 0).sum(axis=1)
        kn[:, c] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.inf)
    return mn, kn, arg


def _ref_radius(D_rr, inv, k):
    """r_k(j): distance from ref j to its k-th nearest OTHER ref of its own
    class (inf for classes smaller than k+1)."""
    D = np.array(D_rr, dtype=np.float32, copy=True)
    np.fill_diagonal(D, np.inf)
    r = np.full(D.shape[0], np.inf, np.float32)
    for c in np.unique(inv):
        idx = np.where(inv == c)[0]
        if len(idx) <= 1:
            continue
        sub = D[np.ix_(idx, idx)]
        kk = min(k, len(idx) - 1)
        r[idx] = np.partition(sub, kk - 1, axis=1)[:, kk - 1]
    return r


def raw_scores(D_query_ref, ref_labels, k=5, tau=None, radius=None):
    """All single-method oddness scores. Returns (pred_class_index, dict)."""
    D = np.asarray(D_query_ref, dtype=np.float32)
    labels, names, inv = _classes(ref_labels)
    mn, kn, arg = _per_class(D, inv, len(names), k)
    order = np.argsort(mn, axis=1)
    c1 = order[:, 0]
    rows = np.arange(D.shape[0])
    d1 = mn[rows, c1]
    d2 = mn[rows, order[:, 1]] if len(names) > 1 else np.full_like(d1, np.inf)
    s = {
        "abs": d1,
        "knn": kn[rows, c1],
        "ratio": d1 / np.maximum(d2, 1e-9),
    }
    if tau is not None:
        s["class"] = d1 / np.maximum(np.asarray(tau, np.float32)[c1], 1e-9)
    if radius is not None:
        j = arg[rows, c1]
        s["recip"] = d1 / np.maximum(np.asarray(radius, np.float32)[j], 1e-9)
    return c1, names, s


def calibrate(D_ref_ref, ref_labels, target_accept=0.95, k=5, class_q=None,
              shrink=20, method="knn"):
    """Reference-only calibration by leave-one-out.

    Every reference is treated as a query against all OTHER references; each
    method's threshold is the `target_accept` quantile of those scores.
    Per-class thresholds (method "class") use the class's own LOO nearest-
    same-class distances, shrunk toward the global quantile for small
    classes: tau_c = (n_c * q_c + shrink * q_all) / (n_c + shrink).

    Returns a params dict for classify_open_set()."""
    D = np.array(D_ref_ref, dtype=np.float32, copy=True)
    np.fill_diagonal(D, np.inf)
    labels, names, inv = _classes(ref_labels)
    q = class_q if class_q is not None else target_accept
    # per-class tau from LOO nearest SAME-class distance
    same = np.full(len(labels), np.inf, np.float32)
    for c in range(len(names)):
        idx = np.where(inv == c)[0]
        if len(idx) > 1:
            same[idx] = D[np.ix_(idx, idx)].min(axis=1)
    fin = np.isfinite(same)
    q_all = float(np.quantile(same[fin], q)) if fin.any() else 1.0
    tau = np.empty(len(names), np.float32)
    for c in range(len(names)):
        v = same[(inv == c) & fin]
        n_c = len(v)
        q_c = float(np.quantile(v, q)) if n_c else q_all
        tau[c] = (n_c * q_c + shrink * q_all) / (n_c + shrink)
    radius = _ref_radius(D_ref_ref, inv, k)
    c1, _, s = raw_scores(D, labels, k=k, tau=tau, radius=radius)
    # LOO naming accuracy is reported, and thresholds are set on ALL refs
    # (the target is "accept this share of genuine in-set clips")
    thr = {m: float(np.quantile(v[np.isfinite(v)], target_accept)) for m, v in s.items()}
    # A combination rejects when EITHER part does, so calibrating each part
    # to `target_accept` alone over-rejects (two 5% tails make ~10%). Store a
    # joint threshold on the combined score so the combination itself hits
    # the target on the references.
    for combo in set(COMBOS) | {method}:
        parts = combo.split("+")
        if len(parts) > 1 and all(p_ in s for p_ in parts):
            comb = np.max(np.stack([s[p_] / max(thr[p_], 1e-12) for p_ in parts]), axis=0)
            thr[combo] = float(np.quantile(comb[np.isfinite(comb)], target_accept))
    loo_acc = float((names[c1] == labels).mean())
    return {"method": method, "k": k, "tau": tau, "radius": radius,
            "thresholds": thr, "target_accept": target_accept,
            "loo_naming_accuracy": loo_acc, "class_names": names}


def classify_open_set(D_query_ref, ref_labels, method="knn", k=5, tau=None,
                      radius=None, thresholds=None, **_ignored):
    """D_query_ref: (n_query, n_ref) distances, SMALL = same character.
    Returns (names, scores, reject_mask) -- reject_mask True meaning
    'none of these reference characters'.

    names  : nearest class by single-link (min) distance, for every query
             (also for rejected ones -- the guess is kept, the flag says
             not to trust it).
    scores : the method's oddness divided by its threshold. <= 1 accepted,
             > 1 rejected; larger = further outside. Comparable across
             methods.
    Pass the dict from calibrate() as **params."""
    if thresholds is None:
        raise ValueError("thresholds missing: run calibrate() on the reference set first")
    c1, names, s = raw_scores(D_query_ref, ref_labels, k=k, tau=tau, radius=radius)
    parts = method.split("+")
    for p in parts:
        if p not in SINGLE:
            raise ValueError(f"unknown method part {p!r}; use {SINGLE} joined by '+'")
        if p not in s:
            raise ValueError(f"method {p!r} needs {'tau' if p == 'class' else 'radius'} from calibrate()")
    score = np.max(np.stack([s[p] / max(thresholds[p], 1e-12) for p in parts]), axis=0)
    if len(parts) > 1:
        if method not in thresholds:
            raise ValueError(f"no joint threshold for {method!r}: pass method={method!r} to calibrate()")
        score = score / max(thresholds[method], 1e-12)
    return names[c1], score.astype(np.float32), score > 1.0


def evaluate(D_query_ref, ref_labels, true_labels, params, methods=None, accepts=(0.80, 0.90, 0.95, 0.99),
             D_ref_ref=None):
    """Measure rejection on LABELLED queries. true_labels: the character
    name, or None for "not one of the reference characters" (no character,
    or a character outside the set). Recalibrates at each accept level when
    D_ref_ref is given. Returns a list of dict rows."""
    truth = np.array([t if t is not None else "" for t in true_labels], dtype=object)
    oos = truth == ""
    rows = []
    for a in accepts:
        p = calibrate(D_ref_ref, ref_labels, target_accept=a, k=params["k"]) if D_ref_ref is not None else params
        for m in methods or ["abs", "knn", "ratio", "class", "recip", "abs+ratio", "knn+ratio"]:
            names, _, rej = classify_open_set(D_query_ref, ref_labels, **{**p, "method": m})
            tp = int((rej & oos).sum())
            rows.append({
                "accept": a, "method": m,
                "reject_precision": tp / max(int(rej.sum()), 1),
                "reject_recall": tp / max(int(oos.sum()), 1),
                "inset_false_reject": int((rej & ~oos).sum()) / max(int((~oos).sum()), 1),
                # what you actually care about: of the names it still proposes, how many are right
                "accepted_precision": int(((~rej) & ~oos & (names == truth)).sum()) / max(int((~rej).sum()), 1),
                "accepted": int((~rej).sum()),
            })
    return rows
