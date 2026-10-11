# Job 4 (v2): "none of the above" with a long tail and real ground truth

Ran in a Claude Code cloud session, on synthetic data only, using numpy and the standard library. This replaces the earlier `REPORT.md` recommendation: the mean-of-k-nearest score assumed 5+ references per character and is worse than a plain threshold on your tail.

## The short answer

1. **A tuned plain distance threshold is as good as anything I tested.** No method beat it beyond seed-to-seed noise, in either synthetic world:
   - ratio test
   - vote gate
   - local scaling
   - mean of k nearest
   - every combination of these
2. **The change that helped is adding the tail characters to the reference set,** even 1-clip ones. It roughly halves the out-of-set clips that slip into accepted proposals.
3. **Set the threshold from your human rulings, not from labelled reference data.** In every world, a threshold tuned to ≥99.3% on labelled clips gave 70–90% on the whole pile. That's the same blindness you measured (99.3% → 92.9%).

**The threshold has no natural cut-off.** Your distance distribution is smooth and unimodal, so a threshold is a choice, not a discovery, and every method here inherits that. The 9 errors were accepted at d ≤ 0.05, which is below the median correct match (0.067). Distance can't separate them without also throwing away most correct matches.

## A. Survey (long-tail view)

| method | needs per-character data? | how it fails here | cost |
|---|---|---|---|
| **plain threshold** (d1) | no | an unseen near-twin sits closer than true matches; no threshold separates them | trivial |
| **ratio** d1/d2 (nearest other character) | no | out-of-set twins are near one character, so the ratio looks confident; no gain measured | trivial |
| **within-class distribution** (per-class quantile / mean of k nearest) | **yes** | undefined or noisy for 75 of 92 characters; mean-of-k changes meaning per character | trivial |
| **local scaling** d1 / ref's own neighbour radius | no | rewards sparse neighbourhoods, and the tail is sparse; worst in every table | ref–ref k-NN, cheap |
| **vote gate** (top-3 refs unanimous) | no | can't rank (three possible values); as a gate it's neutral: sends 1–2-clip characters to a human | cheap |

## B. Recommendation

1. **Reference set = all 92 characters**, including 1-clip ones.
2. **Accept = 3/3 frames agree and worst-frame distance ≤ T**, i.e. `method="abs"`. If you want 1–2-clip characters always reviewed by a person, use `"abs+vote"`; it measured the same as `abs`, within noise.
3. **Choose T with `tune_threshold()`** on clips a human has ruled (your 127, and more as they accumulate), never on the reference set.

**Don't use:**
- **mean of k nearest** (my previous recommendation): 87.9% against 95.7% for the plain threshold at the strict end, with only the 17 populated characters as references.
- **local scaling:** 71–82% when only the 17 populated characters are references.
- **anything needing a per-class distribution:** it can't be estimated for the characters that cause the errors.

## C. `open_set_v2.py`

```python
p = calibrate(D_ref_ref, ref_labels, target_accept=0.95, method="abs")      # LOO on references
names, scores, reject = classify_open_set(D_query_ref, ref_labels, **p)     # rows = frames
clips = aggregate_clips(clip_ids, names, reject, scores)                    # the 3/3 rule
T = tune_threshold(worst_frame_d1_of_ruled, ruled_correct, 0.99)            # from human rulings
```

- **Speed:** 2.4 s for 9,000 × 3,100 frames and 92 characters; 1.0 s to calibrate.
- **Calibratable from references:** the leave-one-out thresholds. That only fixes the false-reject rate on clips of known characters.
- **Not calibratable from references:**
  - **T** on the real pile: it needs human rulings. With 127 rulings and 9 errors it's coarse: precision is known to about ±2–3 points, and a 99% target turns on one or two clips.
  - **k_vote** and **including 1-clip characters:** structural choices; this benchmark is the evidence for them.
- **A calibration bug I found and fixed:** in local scaling, the held-out reference counted toward its own neighbour's radius, which over-accepts (87.7% instead of 80%). The tests now check every method's leave-one-out rate.

## D. Synthetic results (5 seeds each; `bench_longtail_*.md`)

Both worlds have 92 characters (17 populated, 75 tail, about 1,033 clips × 3 frames) and 2,875 query clips. The query mix is 50% populated, 20% tail, 15% unseen and 15% no character, with near-twin look-alikes of the populated characters.

| world | matched to | doesn't match |
|---|---|---|
| **shape** | your distance quantiles (p1 .009, p50 .067, p99 .168) | it's harsher than reality |
| **rule** | your rule's precision (93.0%) | its distance tail is short (p99 .123) |

Neither world reproduces your 4.4% acceptance; the rule accepts 13–21% there. A conclusion is reported only if it held in both.

**A = 17 populated references only (current), B = all 92.** Precision of auto-accepted clips; the out-of-set share among them is in brackets.

| | shape A | shape B | rule A | rule B |
|---|---|---|---|---|
| plain threshold, top 4.4% | 95.7 (4.3) | 96.2 (1.4) | 99.5 (0.5) | 99.7 (0.2) |
| plain threshold, top 10% | 88.4 (11.6) | 89.9 (5.2) | 96.7 (3.3) | 96.6 (1.8) |
| plain threshold, top 20% | 81.7 (18.3) | 85.1 (7.2) | 93.3 (6.7) | 93.9 (3.3) |
| best other method, top 20% | ratio 81.5 | ratio 88.6 | abs+ratio 93.1 | abs+vote 94.4 |
| threshold tuned to 99.3% on labelled clips → whole pile | **69.9** | **85.5** | **77.1** | **89.9** |

Where another method looks ahead, it's within one standard deviation (3–6 points), and it doesn't hold across worlds. `local` and `knn` are worse than the plain threshold in condition A of both worlds; full tables are in the files.

## Files

| file | what |
|---|---|
| `open_set_v2.py` | the module |
| `test_open_set_v2.py` / `test_v2_output.txt` | 10 checks, plus timing |
| `bench_longtail.py` | the benchmark |
| `bench_longtail_shape.md` / `bench_longtail_rule.md` | output for each world |
