"""
test_open_set.py -- correctness checks for open_set.py, plus timing at the
real size (9,000 x 2,200). usage: python test_open_set.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import open_set as OS  # noqa: E402

n = 0


def ok(name):
    global n
    n += 1
    print("  ok  " + name)


rng = np.random.default_rng(0)
# tiny hand-built world: 2 classes on a line
ref = np.array([0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 5.3])
lab = np.array(["A", "A", "A", "B", "B", "B", "B"])
D_rr = np.abs(ref[:, None] - ref[None, :]).astype(np.float32)
q = np.array([0.05, 5.15, 2.6, 0.12, 40.0])
D_qr = np.abs(q[:, None] - ref[None, :]).astype(np.float32)

print("scores")
c1, names, s = OS.raw_scores(D_qr, lab, k=2)
assert list(names[c1]) == ["A", "B", "A", "A", "B"]
ok("nearest class by min distance")
assert np.allclose(s["abs"], [0.05, 0.05, 2.4, 0.02, 34.7], atol=1e-5)
ok("abs = distance to nearest ref")
assert np.isclose(s["knn"][0], (0.05 + 0.05) / 2) and np.isclose(s["knn"][1], (0.05 + 0.05) / 2)
ok("knn = mean of the k nearest refs of the matched class")
assert np.isclose(s["ratio"][2], 2.4 / 2.4) and s["ratio"][0] < 0.02
ok("ratio = d1 / nearest other-class distance (midpoint query -> 1.0)")
_, _, s1 = OS.raw_scores(D_qr, lab, k=1)
assert np.allclose(s1["knn"], s1["abs"])
ok("knn with k=1 equals abs")

print("calibrate / classify")
p = OS.calibrate(D_rr, lab, target_accept=1.0, k=2)
names, score, rej = OS.classify_open_set(D_qr, lab, **{**p, "method": "abs"})
assert list(rej) == [False, False, True, False, True], rej
ok("midpoint and far queries rejected, in-class queries accepted (target_accept=1.0)")
assert np.all((score > 1) == rej)
ok("score > 1 exactly when rejected")
for m in ("knn", "ratio", "class", "recip", "knn+class", "knn+ratio", "abs+ratio"):
    p2 = OS.calibrate(D_rr, lab, target_accept=1.0, k=2, method=m)
    OS.classify_open_set(D_qr, lab, **p2)
ok("every method and combination runs")
try:
    OS.classify_open_set(D_qr, lab, method="abs")
    raise AssertionError("should refuse without thresholds")
except ValueError:
    ok("refuses to classify without calibration")

# calibration hits its target on the references themselves (LOO), per method
R = rng.normal(size=(400, 16)); R /= np.linalg.norm(R, axis=1, keepdims=True)
L = rng.integers(0, 8, 400).astype(str)
D = ((1 - R @ R.T) / 2).astype(np.float32)
for m in ("abs", "knn", "class", "knn+class", "knn+ratio"):
    for a in (0.8, 0.95):
        p = OS.calibrate(D, L, target_accept=a, method=m)
        Dl = D.copy(); np.fill_diagonal(Dl, np.inf)
        _, _, r = OS.classify_open_set(Dl, L, **p)
        acc = 1 - r.mean()
        assert abs(acc - a) < 0.02, (m, a, acc)
ok("LOO accept rate on the references matches target_accept (incl. combinations)")

print("timing at the real size")
nq, nr = 9000, 2200
Rr = rng.normal(size=(nr, 32)).astype(np.float32); Rr /= np.linalg.norm(Rr, axis=1, keepdims=True)
Qq = rng.normal(size=(nq, 32)).astype(np.float32); Qq /= np.linalg.norm(Qq, axis=1, keepdims=True)
Lr = rng.integers(0, 17, nr).astype(str)
Drr = (1 - Rr @ Rr.T) / 2
Dqr = (1 - Qq @ Rr.T) / 2
t = time.time(); p = OS.calibrate(Drr, Lr, method="knn+class"); tc = time.time() - t
t = time.time(); OS.classify_open_set(Dqr, Lr, **p); tq = time.time() - t
print(f"  calibrate 2,200 refs: {tc:.2f}s   classify 9,000 x 2,200: {tq:.2f}s")
ok("runs at 9,000 x 2,200")
print(f"\nALL {n} CHECKS PASSED")
