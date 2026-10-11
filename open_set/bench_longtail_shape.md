# Long-tail open-set benchmark (5 seeds, world preset 'shape': {'FN': 0.9, 'P_TWIN': 0.45})

Synthetic query-frame -> nearest-reference distance (target in brackets):

| p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| 0.009 (0.014) | 0.026 (0.025) | 0.032 (0.032) | 0.046 (0.048) | 0.067 (0.067) | 0.112 (0.094) | 0.145 (0.126) | 0.154 (0.144) | 0.168 (0.171) | 0.19 (~0.30) |

**Condition A, plain threshold tuned to ≥99.3% on labelled (populated-only) clips:** threshold 0.095, labelled precision 100.0 ± 0.0%, **whole-pile precision  69.9 ± 0.9%** on 1725 accepted clips

**Condition B, plain threshold tuned to ≥99.3% on labelled (populated-only) clips:** threshold 0.053, labelled precision  99.4 ± 0.1%, **whole-pile precision  85.5 ± 2.8%** on 638 accepted clips

**Condition A, the actual proposal rule (3/3 agree, d ≤ 0.05):** accepts 19.7% of clips at ** 81.6 ± 4.0% precision**

**Condition B, the actual proposal rule (3/3 agree, d ≤ 0.05):** accepts 18.6% of clips at ** 85.4 ± 3.6% precision**

## A — reference set = 17 populated characters (current)

Precision of auto-accepted clips (and share of them that are out-of-set) when accepting the most confident N% of the pile.

| method | 4% accepted | 10% accepted | 20% accepted |
|---|---:|---:|---:|
| abs |  95.7 ± 2.4 (oos 4.3) |  88.4 ± 6.6 (oos 11.6) |  81.7 ± 6.4 (oos 18.3) |
| ratio |  94.4 ± 3.4 (oos 5.6) |  88.5 ± 6.1 (oos 11.5) |  81.5 ± 6.1 (oos 18.5) |
| local |  71.4 ± 5.4 (oos 28.6) |  67.2 ± 6.5 (oos 32.8) |  68.8 ± 3.9 (oos 31.2) |
| knn |  87.9 ± 9.3 (oos 12.1) |  84.7 ± 7.7 (oos 15.3) |  80.3 ± 5.9 (oos 19.7) |
| abs+vote |  95.7 ± 2.4 (oos 4.3) |  88.4 ± 6.6 (oos 11.6) |  81.7 ± 6.4 (oos 18.3) |
| knn+vote |  87.9 ± 9.3 (oos 12.1) |  84.7 ± 7.7 (oos 15.3) |  80.3 ± 5.9 (oos 19.7) |
| ratio+vote |  94.4 ± 3.4 (oos 5.6) |  88.5 ± 6.1 (oos 11.5) |  81.5 ± 6.1 (oos 18.5) |
| abs+ratio |  95.4 ± 2.6 (oos 4.6) |  88.5 ± 5.9 (oos 11.5) |  81.4 ± 6.3 (oos 18.6) |

## B — reference set = all 92 characters, tail included

Precision of auto-accepted clips (and share of them that are out-of-set) when accepting the most confident N% of the pile.

| method | 4% accepted | 10% accepted | 20% accepted |
|---|---:|---:|---:|
| abs |  96.2 ± 2.4 (oos 1.4) |  89.9 ± 4.7 (oos 5.2) |  85.1 ± 5.7 (oos 7.2) |
| ratio |  96.2 ± 3.4 (oos 3.8) |  91.2 ± 4.6 (oos 8.4) |  88.6 ± 5.1 (oos 9.9) |
| local |  95.6 ± 1.8 (oos 3.2) |  87.2 ± 2.4 (oos 9.7) |  84.6 ± 2.0 (oos 10.5) |
| knn |  91.6 ± 6.0 (oos 4.6) |  87.9 ± 5.3 (oos 6.3) |  83.7 ± 5.7 (oos 8.2) |
| abs+vote |  94.4 ± 3.4 (oos 2.5) |  90.4 ± 5.0 (oos 5.4) |  86.7 ± 5.5 (oos 7.9) |
| knn+vote |  90.5 ± 6.5 (oos 5.4) |  88.7 ± 5.5 (oos 6.5) |  85.6 ± 5.3 (oos 8.6) |
| ratio+vote |  94.3 ± 4.2 (oos 5.7) |  90.4 ± 5.3 (oos 9.2) |  88.1 ± 5.3 (oos 10.2) |
| abs+ratio |  97.3 ± 3.2 (oos 2.2) |  91.7 ± 5.7 (oos 6.6) |  88.2 ± 4.6 (oos 8.1) |

