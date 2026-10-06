# Job 4: answering "none of these characters"

Ran in a Claude Code cloud session, on synthetic data only, using numpy and the standard library.

## Bottom line

Use the **same absolute threshold you would have used, but on the mean distance to the k=5 nearest references of the predicted character** (`knn`), instead of the distance to the single nearest one. In the synthetic tests it beat a tuned plain threshold at every operating point, in both worlds and at every off-model share tried.

It is still a single number calibrated from references alone, so it is barely more complex than a threshold. Every fancier method I tried (ratio test, per-class thresholds, reciprocal neighbours, combinations) was no better than `knn`, and most were no better than the plain threshold.

**Why it works:** a single nearest reference can be a fluke, such as an off-model reference or a look-alike frame. Five agreeing references can't be one fluke.

## A. Survey

| method | assumes | fails when | cost |
|---|---|---|---|
| **absolute threshold** on d1 | one distance scale fits all characters | one odd reference sits near junk (the near-odd case: **0%** caught); varied characters vs tight ones | trivial |
| **mean of k nearest refs of the matched class** (`knn`) | each character has ≥k decent references | class has fewer than ~k good references; out-of-set clip sits inside a real cluster (true look-alikes) | sort k per class, trivial |
| **Lowe ratio** d1 / nearest different class | out-of-set queries are about equally far from several classes | out-of-set is far from everything (ratio ≈ 1 for in-set too); in-set clips between two similar characters get rejected | trivial |
| **per-class threshold** (class's own LOO distance quantile) | each class's spread is estimable | small classes (noisy quantile, needs shrinkage); loose classes swallow junk | trivial |
| **reciprocal / local density** (is the query inside its nearest ref's k-neighbourhood?) | references are dense where queries are legitimate | sparse classes; a dense outlier pair; it judges the neighbour, not the query | ref–ref k-NN, cheap |

## B. Recommendation

- **Use `knn` with k=5**, calibrated at `target_accept` 0.90–0.95 on leave-one-out references.
- **`knn+class` is the only combination worth trying.** At matched false-reject it tied `knn` (AUROC 0.894 both), so it isn't worth the extra moving part until real data says otherwise.
- **Don't use the ratio test.** It was below the plain threshold in both worlds (AUROC 0.736 vs 0.749). It loses the far-from-everything junk that the plain threshold catches, and it rejects legitimate clips that sit between similar characters (the PIG / PINK JELLYFISH kind). It was the best method only on the "generic" background case (47.8% vs 40.7%); that alone doesn't justify it.
- **Don't use reciprocal neighbours.** They were the worst in every table (AUROC 0.617).

## C. Implementation: `open_set.py`

```python
params = calibrate(D_ref_ref, ref_labels, target_accept=0.95, k=5, method="knn")
names, scores, reject = classify_open_set(D_query_ref, ref_labels, **params)
```

- **Scores:** a score of ≤1 means accepted. Larger scores are further outside, and scores are comparable across methods.
- **`names`:** the nearest class is still returned for rejected clips, so the review tool can show the guess.
- **Speed:** 0.28 s for 9,000 × 2,200, and 0.07 s to calibrate 2,200 references.
- **Joint calibration:** combinations are calibrated jointly. Calibrating each part separately to 95% would wrongly reject about twice as many clips (10.5% vs 5.8%); I found and fixed that.

**Calibratable from references only (leave-one-out):** every threshold, and the per-class quantiles (shrunk toward the global value for small classes). The tests check that the references' LOO accept rate hits `target_accept` within 2 points for every method.

**Cannot be calibrated from references:**
1. **The out-of-set rejection rate, and so the right `target_accept`.** That trade-off needs labelled negatives.
2. **k.** The synthetic sensitivity curve is flat from 5 to 15; `k=1` is the plain threshold.
3. **Whether a combination beats `knn`.**

All three need **labelled out-of-set clips**: the review tool's `N` (no character) rulings, plus clips named as characters outside the 17. `evaluate()` takes exactly that and prints the tables below for your real pile.

## D. Synthetic results

**The synthetic world:**
- 17 classes, about 2,200 references, 2% off-model references.
- Class spread tuned so plain nearest-neighbour naming is 90% accurate (you measured 88.8%).
- 1,500 in-set queries, a tenth of them deliberately far from their own class.
- 1,500 out-of-set queries in five kinds (see `bench_output.md` for definitions):

| kind | what it is |
|---|---|
| unseen | characters not in the reference set |
| lookalike | unseen characters close to a known one |
| generic | background-like frames near the centre of all classes |
| pattern | random, kaleidoscope-like frames |
| near-odd | queries sitting next to one off-model reference |

Results are over 5 seeds.

**Separability (each method allowed the same 10% in-set false reject):**

| method | AUROC | out-of-set caught | near-odd | generic | lookalike |
|---|---:|---:|---:|---:|---:|
| abs (baseline) | 0.749 ±0.025 | 45.8% | **0.0%** | 27.1% | 48.0% |
| **knn (k=5)** | **0.894 ±0.023** | **62.6%** | **70.9%** | 40.7% | 49.1% |
| ratio | 0.736 | 40.4% | 0.0% | 47.8% | 38.9% |
| class | 0.741 | 50.2% | 0.0% | 46.6% | 54.9% |
| knn+class | 0.894 | 64.1% | — | — | — |

**Calibrated from references only, at `target_accept` 0.95:**

| method | out-of-set caught | in-set false reject |
|---|---:|---:|
| abs | 33.2% | 5.8% |
| knn | 44.9% | 5.8% |

The easier world (98% naming accuracy) shows the same ordering: knn 0.937 vs abs 0.807 (`bench_output_easier_world.md`).

**Off-model share sensitivity** (`sensitivity_output.md`): with none, knn still wins (0.828 vs 0.752); at 2–10% the gap is larger (≈0.90 vs 0.75).

**Where `knn` doesn't help:**
- **"Generic" background-like frames:** 41% caught.
- **True look-alikes:** 49% caught.
- **Wide shots:** a third of the deliberately far in-set queries are rejected by every method.

For these, distance alone isn't enough. They are the cases to send to review, not to auto-reject.

**Honest limit:** these numbers come from a world I built. The *ordering* held across every variation I tried; the *percentages* won't transfer. Run `evaluate()` on your first few hundred `N` rulings before trusting any rejection rate.

## Files

| file | what |
|---|---|
| `open_set.py` | the module |
| `test_open_set.py` / `test_output.txt` | 11 checks, plus timing at real size |
| `bench_open_set.py` / `bench_output.md` | main benchmark |
| `bench_output_easier_world.md` | benchmark in the easier world |
| `sensitivity.py` / `sensitivity_output.md` | off-model and k sensitivity |
