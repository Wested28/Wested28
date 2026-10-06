# Job 1: person detector for CCIP crops

Ran in a Claude Code cloud session (`CLAUDE_CODE_REMOTE=true`).

## The blocker: no real decode was run

This environment's network policy blocks `huggingface.co` (proxy returns 403 on CONNECT). Its LFS/xet CDNs are blocked too. The Hugging Face connector can list repos and read text files, but it refuses binaries (`HF_FS_TEXT_ONLY`). None of the deepghs Spaces have an MCP endpoint, and image hosts are blocked. So:

- **I have not run the detector on a single real image.**
- **There are no annotated real images.** I'm not sending the synthetic ones, because they show boxes my fake model planted, not detections.
- **The zero-detection rate on masked figures is unmeasured.**

To unblock this, add `huggingface.co`, `cdn-lfs.huggingface.co`, `cdn-lfs.hf.co` and `cas-bridge.xethub.hf.co` to Allowed domains under the environment's Network access settings (https://code.claude.com/docs/en/cloud-environments#network-access). The other route is to run `detect_test_09.py` on alpha-6.

## Candidates

Sizes and thresholds below come from the HF connector, read from the repo listings and `threshold.json`. They are verified as metadata only.

| model | resolve URL | bytes | class(es) | conf (F1) |
|---|---|---:|---|---|
| **person_detect_v1.1_m** ← chosen | `https://huggingface.co/deepghs/anime_person_detection/resolve/main/person_detect_v1.1_m/model.onnx` | **103,459,671** | `person` | 0.348 (0.87) |
| person_detect_v1.3_s | `.../anime_person_detection/resolve/main/person_detect_v1.3_s/model.onnx` | 44,583,231 | `person` | 0.324 (0.86) |
| halfbody_detect_v1.0_s | `https://huggingface.co/deepghs/anime_halfbody_detection/resolve/main/halfbody_detect_v1.0_s/model.onnx` | 44,583,233 | `halfbody` | 0.577 (0.95) |
| head_detect_v2.0_s (fallback) | `https://huggingface.co/deepghs/anime_head_detection/resolve/main/head_detect_v2.0_s/model.onnx` | listed in repo | `head` | 0.413 (0.92) |

- **Why person v1.1_m:** it has the best published F1 in the repo, and it is the default for imgutils' `detect_person()` (IoU 0.5). v1.3_s is the cheaper option with the same F1 within 0.01.
- **halfbody:** this crops the upper body, so on a full-figure wide shot it should give a tighter, larger helmet. That makes it worth a second run, but whether it fires on armoured figures is untested.
- **head:** it's a fallback only. The library skill records a face detector failing 4 of 5 on these helmets. A head detector is not a face detector, but expect the same failure mode, and it is unmeasured here.
- **Disqualified: `deepghs/booru_yolo` (yolov8m_as03).** It runs on onnxruntime, but its 26 classes are body-part and NSFW regions (`head, bust, boob, …, hfox, hcat`), not figures.
- **Disqualified: anything needing torch, CUDA or a custom op.** None of the above do.

## Spec: verified by reading imgutils 0.19.0 source (`imgutils/generic/yolo.py`, from PyPI)

These apply to all four YOLOv8 models.

- **Input:** `images`, float32, `[1,3,H,W]`. H and W come from ONNX metadata `imgsz`, falling back to 640.
- **Preprocessing:** a plain resize, *not* a letterbox, using PIL bicubic. RGB, divided by 255, CHW. No mean/std.
- **Output:** `output0`, shape `[1, 4+nc, N]` (N = 8400 at 640). The rows are `cx, cy, w, h` in **model-input pixels** (not normalised), followed by per-class scores that are already sigmoided.
- **NMS is not baked in.** Do it in numpy, then scale boxes back by `orig/input` per axis.

The real model's `imgsz` metadata and fixed-vs-dynamic input shape are unverified. The probe prints both.

## What was verified in the sandbox (`verify_probe_09.py`, all passing)

1. **Decode against reference:** `decode()` matches imgutils' own `_nms_postprocess` on 300 random YOLOv8-shaped outputs, 1,625 boxes compared. Box integers and scores are identical. The reference code was lifted from the wheel by AST, so it is the real thing.
2. **`group_min` (min over crop pairs via `np.minimum.reduceat`):** equals a brute-force loop on 50 random cases with shuffled, non-contiguous groups.
3. **`p_at_1` (leave-one-out NN) and the shuffled control:** `p_at_1` equals a brute-force loop. The shuffled control on random data gives 0.112 against 0.10 chance.
4. **`pad_box`:** stays in-frame on 1,000 random boxes.
5. **End to end:** `main()` ran on a fake `_index` with tiny synthetic ONNX models (CPU).
   - The labelled-set filter kept the right 36 clips and excluded 6 decoys: a bad-shape int, a bad-shape dict, two names, no frame, not a ruling, and under 10 clips.
   - The report is written line-buffered.
   - The control-failed banner lands as the first line of the file.
   - `detect_test_09.py` also ran end to end.
6. **Compiles:** all three scripts pass `python -m py_compile`.

## Not verified

- The real detector's output on anime, and in particular on masked and helmeted figures. This is the central question, and it is open.
- That DirectML runs this YOLOv8 export and the CCIP model at batch 8.
- That the probe's `full` variant reproduces 88.8%. The probe checks this itself and shouts if it doesn't.
- Where the models sit on D:. The probe searches `_index\models\`, `_index\tools\models\`, `_index\tools\`, `_index\` and the HF cache. It checks exact byte sizes and lists every place it looked if a model is missing.

## Choices in the probe

- **`PAD_FRAC = 0.15` of box width/height per side.** YOLO boxes are tight, and the identity here sits at the extremities: ears, horns, fins, antennae.
- **The crop is not squared.** CCIP resizes straight to 384, which is how imgutils feeds it person crops.
- **`MAX_CROPS = 4`.**
- **The metrics head runs on CPU,** as a single 1.6 KB call with a large batch.
- **One extra variant, `cropN+full`.**
- **A risk the synthetic run surfaced:** under `cropN`, a crop that looks alike across clips (an enemy, a crowd, a background blob) wins the min and drags characters to 0%. The docstring tells the reader how to recognise this.

## Run order on alpha-6

```
& "D:\Video Library\_index\ag.cmd" detect_test_09.py      # look first: logs\detect_test_09\SHEET.jpg
& "D:\Video Library\_index\ag.cmd" embed_probe_09.py      # then logs\embed_probe_09.md
```

`detect_test_09.py` imports `embed_probe_09.py`, so put both in `_index\`. The detector ONNX goes at `_index\models\anime_person_detection\person_detect_v1.1_m\model.onnx`.
