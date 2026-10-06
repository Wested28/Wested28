"""
sensitivity.py -- does the knn advantage survive changes to the synthetic
world? AUROC of abs vs knn as the share of off-model references changes,
and knn AUROC as k changes. 4 seeds each. usage: python sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_open_set as B  # noqa: E402
import open_set as OS  # noqa: E402


def aurocs(odd, k, seeds=4):
    B.ODD = odd
    a, b = [], []
    for s in range(seeds):
        D_qr, D_rr, rl, truth, _ = B.world(s)
        oos = np.array([t is None for t in truth])
        p = OS.calibrate(D_rr, rl, 0.95, k=k)
        _, sa, _ = OS.classify_open_set(D_qr, rl, **{**p, "method": "abs"})
        _, sk, _ = OS.classify_open_set(D_qr, rl, **{**p, "method": "knn"})
        a.append(B.auroc(sa, oos)); b.append(B.auroc(sk, oos))
    return np.mean(a), np.mean(b)


print("| off-model share of refs | AUROC abs | AUROC knn (k=5) |\n|---:|---:|---:|")
for odd in (0.0, 0.02, 0.05, 0.10):
    print("| %.2f | %.3f | %.3f |" % ((odd,) + aurocs(odd, 5)))
print("\n| k | AUROC knn (2% off-model) |\n|---:|---:|")
for k in (1, 2, 3, 5, 8, 15):
    print("| %d | %.3f |" % (k, aurocs(0.02, k)[1]))
