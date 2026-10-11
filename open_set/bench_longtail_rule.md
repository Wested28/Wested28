# Long-tail open-set benchmark (5 seeds, world preset 'rule': {'FN': 1.4, 'P_TWIN': 0.2})

Synthetic query-frame -> nearest-reference distance (target in brackets):

| p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| 0.005 (0.014) | 0.028 (0.025) | 0.034 (0.032) | 0.046 (0.048) | 0.067 (0.067) | 0.096 (0.094) | 0.110 (0.126) | 0.115 (0.144) | 0.123 (0.171) | 0.14 (~0.30) |

**Condition A, plain threshold tuned to ≥99.3% on labelled (populated-only) clips:** threshold 0.094, labelled precision 100.0 ± 0.0%, **whole-pile precision  77.1 ± 2.3%** on 1698 accepted clips

**Condition B, plain threshold tuned to ≥99.3% on labelled (populated-only) clips:** threshold 0.068, labelled precision  99.6 ± 0.2%, **whole-pile precision  89.9 ± 4.8%** on 1038 accepted clips

**Condition A, the actual proposal rule (3/3 agree, d ≤ 0.05):** accepts 21.4% of clips at ** 93.0 ± 3.0% precision**

**Condition B, the actual proposal rule (3/3 agree, d ≤ 0.05):** accepts 20.6% of clips at ** 94.1 ± 2.1% precision**

## A — reference set = 17 populated characters (current)

Precision of auto-accepted clips (and share of them that are out-of-set) when accepting the most confident N% of the pile.

| method | 4% accepted | 10% accepted | 20% accepted |
|---|---:|---:|---:|
| abs |  99.5 ± 0.6 (oos 0.5) |  96.7 ± 2.1 (oos 3.3) |  93.3 ± 3.4 (oos 6.7) |
| ratio |  99.4 ± 0.9 (oos 0.6) |  96.4 ± 2.1 (oos 3.6) |  92.7 ± 3.3 (oos 7.3) |
| local |  82.4 ± 7.2 (oos 17.6) |  83.1 ± 5.7 (oos 16.9) |  83.9 ± 4.8 (oos 16.1) |
| knn |  98.7 ± 1.4 (oos 1.3) |  95.8 ± 2.8 (oos 4.2) |  92.7 ± 4.0 (oos 7.3) |
| abs+vote |  99.5 ± 0.6 (oos 0.5) |  96.6 ± 2.1 (oos 3.4) |  93.1 ± 3.5 (oos 6.9) |
| knn+vote |  98.7 ± 1.4 (oos 1.3) |  95.8 ± 2.8 (oos 4.2) |  92.7 ± 4.0 (oos 7.3) |
| ratio+vote |  99.2 ± 0.9 (oos 0.8) |  96.2 ± 2.2 (oos 3.8) |  92.7 ± 3.4 (oos 7.3) |
| abs+ratio |  99.8 ± 0.3 (oos 0.2) |  96.3 ± 2.2 (oos 3.7) |  93.1 ± 3.3 (oos 6.9) |

## B — reference set = all 92 characters, tail included

Precision of auto-accepted clips (and share of them that are out-of-set) when accepting the most confident N% of the pile.

| method | 4% accepted | 10% accepted | 20% accepted |
|---|---:|---:|---:|
| abs |  99.7 ± 0.4 (oos 0.2) |  96.6 ± 2.1 (oos 1.8) |  93.9 ± 2.8 (oos 3.3) |
| ratio |  96.7 ± 3.7 (oos 3.3) |  95.7 ± 3.1 (oos 4.2) |  92.6 ± 2.6 (oos 6.5) |
| local |  97.5 ± 1.5 (oos 2.1) |  94.7 ± 0.7 (oos 4.2) |  92.9 ± 1.0 (oos 5.1) |
| knn |  98.4 ± 2.0 (oos 0.6) |  96.2 ± 2.4 (oos 1.9) |  93.7 ± 3.5 (oos 3.5) |
| abs+vote |  99.4 ± 0.6 (oos 0.3) |  96.9 ± 2.0 (oos 2.0) |  94.4 ± 2.4 (oos 3.8) |
| knn+vote |  98.6 ± 2.5 (oos 0.6) |  96.4 ± 2.3 (oos 2.4) |  94.2 ± 2.8 (oos 3.9) |
| ratio+vote |  96.0 ± 4.2 (oos 4.0) |  95.3 ± 3.1 (oos 4.7) |  92.2 ± 2.7 (oos 6.7) |
| abs+ratio |  97.5 ± 2.8 (oos 2.5) |  96.0 ± 2.8 (oos 3.9) |  93.4 ± 2.6 (oos 5.4) |

