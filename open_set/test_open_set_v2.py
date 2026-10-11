"""test_open_set_v2.py -- checks for open_set_v2.py, plus timing at the real
size (9,000 query frames x 3,100 reference frames). usage: python test_open_set_v2.py"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import open_set_v2 as OS

n = 0
def ok(msg):
    global n; n += 1; print("  ok  " + msg)

# tiny 1-D world: big class A, singleton class S, class B
ref = np.array([0.0, 0.1, 0.2, 0.3, 2.0, 5.0, 5.1, 5.2])
lab = np.array(["A", "A", "A", "A", "S", "B", "B", "B"])
D_rr = np.abs(ref[:, None] - ref[None, :]).astype(np.float32)
q = np.array([0.15, 2.02, 3.3, 5.05])
D_qr = np.abs(q[:, None] - ref[None, :]).astype(np.float32)

first, names, s = OS.raw_scores(D_qr, lab, k_vote=3)
assert list(names[first]) == ["A", "S", "S", "B"], names[first]
ok("nearest character by single-link distance")
assert np.isclose(s["vote_share"][0], 1.0) and np.isclose(s["vote_share"][1], 1 / 3)
ok("vote share: a singleton character can never be unanimous at k=3")
p = OS.calibrate(D_rr, lab, target_accept=1.0, method="abs+vote")
nm, sc, rej = OS.classify_open_set(D_qr, lab, **p)
assert not rej[0] and rej[1], (sc, rej)
ok("abs+vote: populated-class query accepted, singleton-class query sent to a human")
for m in ("abs", "ratio", "local", "knn", "abs+vote", "knn+vote", "abs+ratio", "ratio+vote"):
    OS.classify_open_set(D_qr, lab, **OS.calibrate(D_rr, lab, 0.9, method=m))
ok("every method and combination runs")
try:
    OS.classify_open_set(D_qr, lab, **{**p, "method": "vote"}); raise AssertionError
except ValueError:
    ok("'vote' alone is refused (it is a gate, not a ranking)")

# LOO calibration hits the target on the references (ranking methods)
rng = np.random.default_rng(0)
R = rng.normal(size=(600, 16)); R /= np.linalg.norm(R, axis=1, keepdims=True)
L = rng.integers(0, 10, 600).astype(str)
D = ((1 - R @ R.T) / 2).astype(np.float32)
for m in ("abs", "ratio", "local", "knn", "abs+ratio"):
    for a in (0.8, 0.95):
        pp = OS.calibrate(D, L, a, method=m)
        Dl = D.copy(); np.fill_diagonal(Dl, np.inf)
        if m == "local":   # LOO check must use the leave-one-out yardstick, as calibrate() does
            _, _, sraw = OS.raw_scores(Dl, L, sigma_rows=OS._loo_sigma_rows(Dl, 3))
            r = sraw["local"] / pp["thresholds"]["local"] > 1
        else:
            _, _, r = OS.classify_open_set(Dl, L, **pp)
        assert abs((1 - r.mean()) - a) < 0.02, (m, a, 1 - r.mean())
ok("LOO accept rate on references matches target_accept")

# clip aggregation: 3/3 rule
ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
nmz = np.array(["A", "A", "A", "A", "B", "A", "A", "A", "A"])
rj = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], bool)
agg = OS.aggregate_clips(ids, nmz, rj)
assert agg[1][0] == "A" and agg[2][0] is None and agg[3][0] is None
ok("aggregate_clips: accepted only when all 3 frames pass and agree")

# tune_threshold on rulings shaped like the real 127 (117 right, 9 wrong, 1 unknown->wrong)
sc = np.r_[rng.uniform(0.0, 0.05, 117), rng.uniform(0.02, 0.05, 10)]
cr = np.r_[np.ones(117, bool), np.zeros(10, bool)]
t = OS.tune_threshold(sc, cr, 0.99)
assert t is not None and t[1] >= 0.99 and t[0] < 0.05
allp = cr.mean()
ok(f"tune_threshold: whole set {allp:.3f} precise; tightest-safe threshold {t[0]:.4f} keeps {t[2]} clips at {t[1]:.3f}")
assert OS.tune_threshold(sc, cr, 1.01) is None
ok("tune_threshold returns None when no threshold meets the target")

# timing at the real size
nq, nr = 9000, 3100
Rr = rng.normal(size=(nr, 32)).astype(np.float32); Rr /= np.linalg.norm(Rr, axis=1, keepdims=True)
Qq = rng.normal(size=(nq, 32)).astype(np.float32); Qq /= np.linalg.norm(Qq, axis=1, keepdims=True)
Lr = rng.integers(0, 92, nr).astype(str)
Drr = (1 - Rr @ Rr.T) / 2; Dqr = (1 - Qq @ Rr.T) / 2
t0 = time.time(); pp = OS.calibrate(Drr, Lr, method="abs+vote"); tc = time.time() - t0
t0 = time.time(); OS.classify_open_set(Dqr, Lr, **pp); tq = time.time() - t0
print(f"  calibrate 3,100 refs: {tc:.2f}s   classify 9,000 x 3,100: {tq:.2f}s")
ok("runs at the real size")
print(f"\nALL {n} CHECKS PASSED")
