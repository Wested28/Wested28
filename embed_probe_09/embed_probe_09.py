"""
embed_probe_09.py -- does cropping to detected figures help CCIP on wide shots?

WHY THIS EXISTS
  CCIP (ccip-caformer_b36-24) identifies characters at 88.8% precision@1 over
  739 hand-labelled clips / 17 characters (7.7x a shuffled control). It is
  given the whole 448x448 midpoint frame. On wide shots -- an army of
  ant-sized figures, a silhouette against a city -- the character is a few
  dozen pixels and CCIP has nothing to look at. One character scored 10% on
  the midpoint frame and 100% once other frames were used: that points at
  FRAMING, not the model.

  Hypothesis: detect the figures (deepghs anime_person_detection, a YOLOv8
  body detector -- NOT a face detector, because these characters wear full
  animal-mask helmets), crop to them, embed the crops. A 40 px figure
  becomes a 384 px input.

WHAT IT MEASURES (all on the _2 midpoint frame, leave-one-out NN P@1
through the CCIP metrics head, plus a 5-round shuffled-label control)
  full        whole frame. The CONTROL. Must reproduce 88.8% (+-2 pts) or
              nothing else in this report is comparable to earlier probes.
  crop1       the single highest-confidence detection, padded, embedded.
  cropN       every detection (up to MAX_CROPS) embedded; clip-to-clip
              distance = min over all crop pairs.
  content     whole frame with the letterbox bars cut off. The frames are 16:9
              video padded to 448x448; a crop removes the bars as a side
              effect, so this variant separates "bars gone" from "figure
              cropped". If content alone moves the score, credit it there.
  cropN+full  as cropN, with the whole frame added as one more "crop".
              Extra variant: it cannot do worse than its parts by much and
              it tells you whether crops ADD to the frame or REPLACE it.
  A frame with zero detections falls back to the full frame in every crop
  variant; the fallback rate is reported, overall and per character.

WHAT EACH OUTCOME WOULD MEAN
  crop variant up, wide-shot characters up most  -> framing was the
      problem. Next: run detection over all three frames and the library.
  crop1 down, cropN flat/up                      -> the top-confidence box is
      often the wrong figure (the enemy, a bystander). Subject choice, not
      cropping, is the problem; crop1 should not ship.
  everything flat, fallback rate high            -> the detector does not see
      these figures (masked/armoured/tiny). Cropping was never tested;
      the detector is the problem, not the idea. Read the fallback column.
  everything flat, fallback rate low             -> cropping does NOT help.
      At 448 px a 40 px figure upscaled to 384 is still 40 px of
      information; CCIP had already used what was there. The fix is more
      or higher-resolution frames, not crops.
  cropN DOWN while crop1 holds                    -> a crop shared across
      clips (an enemy, a crowd, a background shape the detector called a
      person) is near-identical between characters, and min-over-crops lets
      it win. Seen in the sandbox on synthetic data: one identical
      background box per frame took two characters from 100% to 0%.
  aggregate flat but per-character table moving -> READ THE TABLE. In this
      project an aggregate has already hidden a +90 and a -11 in one run.

USAGE
  ag.cmd embed_probe_09.py [folder containing model_feat.onnx]
  Without the argument it searches _index (skipping the frames folder) and the
  Hugging Face cache, size-checked.

READ-ONLY. Writes nothing to library.jsonl or anywhere in the index. The
only file it writes is its own report, logs\\embed_probe_09.md.
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

INDEX = Path(r"D:\Video Library\_index")

# ---- models ---------------------------------------------------------------
# Person detector: deepghs/anime_person_detection, person_detect_v1.1_m
#   https://huggingface.co/deepghs/anime_person_detection/resolve/main/person_detect_v1.1_m/model.onnx
#   103,459,671 bytes. YOLOv8m, one class "person". F1 0.87 @ conf 0.348
#   (threshold.json). This is also imgutils' detect_person() default.
DET_NAME = "person_detect_v1.1_m"
DET_BYTES = 103_459_671
DET_URL = ("https://huggingface.co/deepghs/anime_person_detection/resolve/main/"
           f"{DET_NAME}/model.onnx")
CCIP_FEAT_BYTES = 383_591_416
CCIP_METRICS_BYTES = 1_618

CONF_THRESHOLD = 0.348  # threshold.json for person_detect_v1.1_m (best F1)
IOU_THRESHOLD = 0.5     # imgutils detect_person() default
MAX_CROPS = 4           # cap per frame for cropN; by confidence
MIN_BOX_SIDE = 4        # px in the 448 frame; smaller boxes are dropped as degenerate

# PAD_FRAC: each side of the box grows by this fraction of the box's own
# width (left/right) and height (top/bottom), then clips to the frame.
# 0.15 because YOLO person boxes are tight and these characters' identity
# lives at the extremities -- helmet ears, horns, antennae, a fin -- which a
# tight box clips. The crop is NOT squared: CCIP preprocessing resizes
# straight to 384x384 with no aspect preservation, which is how it is fed
# person crops in imgutils, so a tall figure squashed to a square matches
# its usual input. Squaring would put more background in the crop instead.
PAD_FRAC = 0.15

# Letterbox detection: a row/column is "bar" when its greyscale std is below
# BAR_STD. Bars are only accepted when they are on BOTH sides and within
# BAR_SYM px of each other -- a white-background character sheet has a
# uniform top but not a matching bottom, and must not be cropped.
BAR_STD = 4.0
BAR_MIN = 6
BAR_SYM = 6

CONTROL_EXPECTED = 88.8
CONTROL_TOL = 2.0
MIN_CLIPS_PER_CHAR = 10
SHUFFLE_ROUNDS = 5
FEAT_BATCH = 8

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32).reshape(3, 1, 1)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32).reshape(3, 1, 1)


def paths():
    return {
        "lib": INDEX / "library.jsonl",
        "frames": INDEX / "frames",
        "report": INDEX / "logs" / "embed_probe_09.md",
    }


def model_candidates(*parts):
    """Where the models might live. First existing hit wins; all are listed
    in the report if none is found."""
    roots = [INDEX / "models", INDEX / "tools" / "models", INDEX / "tools",
             INDEX, Path.home() / ".cache" / "huggingface" / "hub"]
    return [r.joinpath(*parts) for r in roots]


def find_model(label, expected_bytes, cands):
    """First candidate that exists AND has the exact byte size wins. A file
    of the wrong size is reported, never used."""
    wrong = []
    for p in cands:
        if p.is_file():
            size = p.stat().st_size
            if size != expected_bytes:
                print(f"  {label}: {p}  {size:,} bytes  WRONG SIZE "
                      f"(expected {expected_bytes:,}) -- skipped")
                wrong.append(p)
                continue
            print(f"  {label}: {p}  {size:,} bytes  OK")
            return p
    print(f"  {label}: NOT FOUND. Looked in:")
    for p in cands:
        print(f"    {p}")
    raise SystemExit(f"{label} missing")


def ccip_candidates(argv):
    """Explicit folder first, then the fixed spots, then the Hugging Face
    cache layout (models--deepghs--ccip_onnx/snapshots/<hash>/...), then
    every model_feat.onnx under _index except the frames folder."""
    sub = ("ccip-caformer_b36-24", "model_feat.onnx")
    c = [Path(a) / "model_feat.onnx" for a in argv[1:2]]
    c += model_candidates("ccip_onnx", *sub) + model_candidates(*sub)
    hubs = [Path(os.environ[k]) / "hub" for k in ("HF_HOME",) if os.environ.get(k)]
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        hubs.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
    hubs.append(Path.home() / ".cache" / "huggingface" / "hub")
    for h in hubs:
        c += sorted(h.glob("models--deepghs--ccip_onnx/snapshots/*/" + "/".join(sub)))
    if INDEX.is_dir():
        for root, dirs, files in os.walk(INDEX):
            dirs[:] = [d for d in dirs if d.lower() not in ("frames", "_superseded", "charpass")]
            if "model_feat.onnx" in files:
                c.append(Path(root) / "model_feat.onnx")
    seen, out = set(), []
    for p in c:
        if str(p).lower() not in seen:
            seen.add(str(p).lower())
            out.append(p)
    return out


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def agsrc(r):
    """ag_source is a str on most records and a DICT on some. Match on VALUES;
    keys go stale on a rename. Returns (tag, raw); tag None = unknown shape."""
    v = r.get("ag_source")
    if v is None:
        return "", None
    if isinstance(v, str):
        return v.strip().upper(), None
    if isinstance(v, dict) and all(isinstance(x, str) for x in v.values()):
        vals = {x.strip().upper() for x in v.values()}
        return ("W" if "W" in vals else "C" if "C" in vals else ""), None
    return None, v


def build_set(lib_path, frames_dir):
    recs, bad = [], []
    shapes = Counter()
    with open(lib_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    for r in recs:
        shapes[type(r.get("ag_source")).__name__] += 1
    print(f"library records: {len(recs):,}")
    print(f"ag_source shapes: {dict(shapes)}")
    clips = []
    for r in recs:
        tag, raw = agsrc(r)
        if tag is None:
            bad.append((r.get("id"), raw))
            continue
        if not (r.get("ag_cluster") or tag in ("W", "C")):
            continue
        names = r.get("ag_characters")
        if not (isinstance(names, list) and len(names) == 1):
            continue
        fp = frames_dir / f"{r['id']}_2.jpg"
        if not fp.is_file():
            continue
        clips.append((r["id"], names[0], fp))
    if bad:
        print(f"!! {len(bad)} records with an unknown ag_source shape, EXCLUDED:")
        for i, raw in bad[:10]:
            print(f"   {i}: {raw!r}")
    n = Counter(c[1] for c in clips)
    keep = {k for k, v in n.items() if v >= MIN_CLIPS_PER_CHAR}
    clips = sorted([c for c in clips if c[1] in keep], key=lambda c: c[0])
    print(f"labelled set: {len(clips)} clips / {len(keep)} characters "
          f"(expected 739 / 17)")
    if (len(clips), len(keep)) != (739, 17):
        print("!! labelled set differs from the 739/17 of earlier probes -- "
              "the index changed since; the 88.8% baseline may not apply.")
    return clips


# ---- detector (decode follows imgutils.generic.yolo, v0.19.0) -------------
def det_preprocess(img, size_wh):
    """Plain resize (NOT letterbox) to the model size, RGB, /255, CHW, float32.
    imgutils uses PIL's default resample for Image.resize, which is BICUBIC."""
    im = img.resize(size_wh, Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return a.transpose(2, 0, 1)[None]


def nms(boxes, scores, iou):
    """imgutils _yolo_nms, verbatim maths (note the +1 pixel convention)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ov = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ov <= iou)[0] + 1]
    return keep


def decode(out, conf, iou, old_wh, new_wh):
    """out: [4+nc, N] -- rows cx, cy, w, h in MODEL-INPUT pixels, then class
    scores (already sigmoided). NMS is NOT baked in. Returns a list of
    (x0, y0, x1, y1, score) in ORIGINAL-image pixels, best first."""
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    assert out.shape[0] >= 5, f"unexpected detector output {out.shape}"
    best = out[4:, :].max(axis=0)
    sel = out[:, best > conf].T
    if not sel.size:
        return []
    xywh, sc = sel[:, :4], sel[:, 4:].max(axis=1)
    b = np.empty_like(xywh)
    b[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    b[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    b[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    b[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    keep = nms(b, sc, iou)
    sx, sy = old_wh[0] / new_wh[0], old_wh[1] / new_wh[1]
    res = []
    for k in keep:
        x0 = float(np.clip(b[k, 0] * sx, 0, old_wh[0]))
        y0 = float(np.clip(b[k, 1] * sy, 0, old_wh[1]))
        x1 = float(np.clip(b[k, 2] * sx, 0, old_wh[0]))
        y1 = float(np.clip(b[k, 3] * sy, 0, old_wh[1]))
        res.append((x0, y0, x1, y1, float(sc[k])))
    return res  # nms() returns indices in descending score order


def det_input_size(sess):
    meta = sess.get_modelmeta().custom_metadata_map or {}
    if "imgsz" in meta:
        h, w = json.loads(meta["imgsz"])  # ultralytics stores [h, w]
        return (int(w), int(h)), f"metadata imgsz={meta['imgsz']}"
    shp = sess.get_inputs()[0].shape
    if isinstance(shp[2], int) and isinstance(shp[3], int):
        return (shp[3], shp[2]), f"fixed input shape {shp}"
    return (640, 640), "default 640 (no metadata, dynamic input)"


def _bars(std):
    """Leading/trailing low-variance run lengths of a 1-D std profile, kept
    only if both exist and are near-equal (a real letterbox/pillarbox)."""
    live = np.where(std >= BAR_STD)[0]
    if not live.size:
        return 0, 0
    a, b = int(live[0]), int(len(std) - 1 - live[-1])
    if a >= BAR_MIN and b >= BAR_MIN and abs(a - b) <= BAR_SYM:
        return a, b
    return 0, 0


def content_box(img):
    """(x0, y0, x1, y1) of the picture inside letterbox/pillarbox bars; the
    whole image when there are none."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    H, W = g.shape
    t, b = _bars(g.std(axis=1))
    l, r = _bars(g.std(axis=0))
    return (l, t, W - r, H - b)


def pad_box(box, W, H, pad=PAD_FRAC, bounds=None):
    """Grow the box by `pad` of its own size per side, then clip to
    `bounds` (the content box) so padding never pulls in letterbox bars."""
    bx0, by0, bx1, by1 = bounds if bounds else (0, 0, W, H)
    x0, y0, x1, y1 = box[:4]
    x0, x1 = max(x0, bx0), min(x1, bx1)
    y0, y1 = max(y0, by0), min(y1, by1)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    x0, x1 = max(float(bx0), x0 - pad * w), min(float(bx1), x1 + pad * w)
    y0, y1 = max(float(by0), y0 - pad * h), min(float(by1), y1 + pad * h)
    return (int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1)))


# ---- CCIP -----------------------------------------------------------------
def ccip_pre(img):
    im = img.convert("RGB").resize((384, 384), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (a - CLIP_MEAN) / CLIP_STD


def embed(sess, imgs):
    name = sess.get_inputs()[0].name
    out = []
    for i in range(0, len(imgs), FEAT_BATCH):
        x = np.stack([ccip_pre(im) for im in imgs[i:i + FEAT_BATCH]]).astype(np.float32)
        out.append(sess.run(None, {name: x})[0])
    return np.concatenate(out, 0).astype(np.float32)


def group_min(D, groups):
    """D: [M, M] distances over embeddings. groups: per clip, a list of
    embedding indices. Returns [C, C] where entry (i, j) is the min of D
    over groups[i] x groups[j]. Done with reduceat on a reordered copy so
    each clip's rows/cols are contiguous."""
    order = np.concatenate([np.asarray(g, dtype=np.int64) for g in groups])
    starts = np.cumsum([0] + [len(g) for g in groups[:-1]]).astype(np.int64)
    R = D[np.ix_(order, order)]
    R = np.minimum.reduceat(R, starts, axis=0)
    return np.minimum.reduceat(R, starts, axis=1)


def p_at_1(Dc, labels):
    """Leave-one-out nearest neighbour. Returns per-clip hit (bool array)."""
    Dc = Dc.astype(np.float64, copy=True)
    np.fill_diagonal(Dc, np.inf)
    nn = Dc.argmin(axis=1)
    return labels[nn] == labels


def shuffled(Dc, labels, rounds=SHUFFLE_ROUNDS, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([p_at_1(Dc, rng.permutation(labels)).mean() for _ in range(rounds)]))


def main():
    P = paths()
    sys.stdout = Tee(str(P["report"]))
    t0 = time.time()
    print("# embed_probe_09 -- person-detector crops vs full frame for CCIP\n")
    print("READ-ONLY: writes nothing to the index. Only this report is written.\n")
    print("```")
    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__}, providers available: {ort.get_available_providers()}")
    det_p = find_model("detector", DET_BYTES,
                       model_candidates("anime_person_detection", DET_NAME, "model.onnx"))
    feat_p = find_model("ccip feat", CCIP_FEAT_BYTES, ccip_candidates(sys.argv))
    met_p = find_model("ccip metrics", CCIP_METRICS_BYTES,
                       [feat_p.parent / "model_metrics.onnx"])

    prov = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()]
    det = ort.InferenceSession(str(det_p), providers=prov)
    feat = ort.InferenceSession(str(feat_p), providers=prov)
    # metrics head is 1.6 KB and gets one big batch; CPU avoids any DML
    # batch-size surprise and costs nothing.
    met = ort.InferenceSession(str(met_p), providers=["CPUExecutionProvider"])
    print(f"detector providers: {det.get_providers()}")
    print(f"ccip feat providers: {feat.get_providers()}")
    size_wh, why = det_input_size(det)
    print(f"detector input {det.get_inputs()[0].name} {det.get_inputs()[0].shape} -> "
          f"feeding {size_wh[0]}x{size_wh[1]} ({why})")
    print(f"detector outputs: {[(o.name, o.shape) for o in det.get_outputs()]}")
    print(f"CONF {CONF_THRESHOLD}  IOU {IOU_THRESHOLD}  PAD_FRAC {PAD_FRAC}  "
          f"MAX_CROPS {MAX_CROPS}  MIN_BOX_SIDE {MIN_BOX_SIDE}")
    print("```\n")

    clips = build_set(P["lib"], P["frames"])
    labels = np.array([c[1] for c in clips])
    chars = sorted(set(labels))

    # ---- detection ------------------------------------------------------
    print("\n## Detection\n")
    det_in = det.get_inputs()[0].name
    frames, dets = [], []
    first_raw = True
    for k, (cid, _, fp) in enumerate(clips):
        img = Image.open(fp).convert("RGB")
        frames.append(img)
        raw = det.run(None, {det_in: det_preprocess(img, size_wh)})[0]
        if first_raw:
            top3 = np.argsort(raw[0][4:].max(axis=0))[::-1][:3]
            print(f"raw output shape {raw.shape} dtype {raw.dtype}; 3 highest-score "
                  f"columns (cx, cy, w, h, score) in {size_wh[0]}x{size_wh[1]} input px:")
            print("```")
            print(np.array2string(raw[0][:, top3].T, precision=3, suppress_small=True))
            print("```")
            first_raw = False
        d = decode(raw, CONF_THRESHOLD, IOU_THRESHOLD, img.size, size_wh)
        d = [b for b in d if (b[2] - b[0]) >= MIN_BOX_SIDE and (b[3] - b[1]) >= MIN_BOX_SIDE]
        dets.append(d)
        if (k + 1) % 100 == 0:
            print(f"  detected {k + 1}/{len(clips)}  ({time.time() - t0:.0f}s)")
    ndet = np.array([len(d) for d in dets])
    zero = ndet == 0
    print(f"\nframes with ZERO detections (fall back to full frame): "
          f"{zero.sum()}/{len(clips)} = {100 * zero.mean():.1f}%")
    print(f"detections per frame: {dict(sorted(Counter(np.minimum(ndet, 9).tolist()).items()))}"
          f"  (9 = 9+)")
    sides = [max(b[2] - b[0], b[3] - b[1]) for d in dets for b in d]
    if sides:
        q = np.percentile(sides, [10, 50, 90])
        print(f"box long side, px in the 448 frame: p10 {q[0]:.0f} / median {q[1]:.0f} / "
              f"p90 {q[2]:.0f}")
        top = [d[0][4] for d in dets if d]
        print(f"top-box confidence median {np.median(top):.3f}")

    # ---- embeddings -----------------------------------------------------
    print("\n## Embedding\n")
    imgs, full_idx, cont_idx, crop_idx = [], [], [], []
    boxes = [content_box(img) for img in frames]
    lb = sum(1 for bx, img in zip(boxes, frames) if bx != (0, 0) + img.size)
    print(f"letterboxed/pillarboxed frames: {lb}/{len(frames)}; most common content box: "
          f"{Counter(boxes).most_common(1)[0][0]}")
    for img, d, bx in zip(frames, dets, boxes):
        full_idx.append(len(imgs))
        imgs.append(img)
        cont_idx.append(len(imgs))
        imgs.append(img.crop(bx))
        ci = []
        for b in d[:MAX_CROPS]:
            ci.append(len(imgs))
            imgs.append(img.crop(pad_box(b, *img.size, bounds=bx)))
        crop_idx.append(ci)
    print(f"embedding {len(imgs)} images ({len(clips)} full + {len(clips)} content + "
          f"{len(imgs) - 2 * len(clips)} crops)")
    E = embed(feat, imgs)
    print(f"embeddings {E.shape}  ({time.time() - t0:.0f}s)")
    D = met.run(None, {met.get_inputs()[0].name: E})[0].astype(np.float32)
    print(f"metrics head distance matrix {D.shape}; "
          f"diag mean {np.diag(D).mean():.4f} (should be ~0), "
          f"asymmetry max {np.abs(D - D.T).max():.2e}")

    variants = {
        "full": [[f] for f in full_idx],
        "content": [[c] for c in cont_idx],
        "crop1": [c[:1] if c else [f] for f, c in zip(full_idx, crop_idx)],
        "cropN": [c if c else [f] for f, c in zip(full_idx, crop_idx)],
        "cropN+full": [[f] + c for f, c in zip(full_idx, crop_idx)],
    }
    res = {}
    for name, groups in variants.items():
        Dc = group_min(D, groups)
        hits = p_at_1(Dc, labels)
        res[name] = (hits, shuffled(Dc, labels))

    # ---- report ---------------------------------------------------------
    pf = 100 * res["full"][0].mean()
    control_failed = abs(pf - CONTROL_EXPECTED) > CONTROL_TOL
    banner = (f"# !!!!! CONTROL FAILED !!!!!\n\n"
              f"**`full` scored {pf:.1f}%, expected {CONTROL_EXPECTED}% +-{CONTROL_TOL}. "
              f"NOTHING ELSE IN THIS REPORT IS COMPARABLE to earlier probes until this "
              f"is explained (preprocessing, model file, or the labelled set changed).**\n")
    if control_failed:
        print("\n" + banner)
    else:
        print(f"\nControl OK: `full` {pf:.1f}% (expected {CONTROL_EXPECTED}% +-{CONTROL_TOL}).\n")

    print("## Headline -- do not stop here, read the per-character table\n")
    print("| variant | P@1 | shuffled | x control |")
    print("|---|---:|---:|---:|")
    for name, (hits, sh) in res.items():
        p = hits.mean()
        print(f"| {name} | {100 * p:.1f}% | {100 * sh:.1f}% | "
              f"{p / sh if sh else float('nan'):.1f}x |")
    print(f"\nfull-frame fallback in crop variants: {100 * zero.mean():.1f}% of frames")

    crops = ["crop1", "cropN", "cropN+full"]
    best = max(crops, key=lambda v: res[v][0].mean())
    print(f"\n## Per character: full vs {best} (best crop variant), sorted by |change|\n")
    print("**Read this table, not the headline.** An aggregate in this project has "
          "already hidden a +90 and a -11 in the same run.\n")
    print("| character | n | zero-det % | full | content | crop1 | cropN | cropN+full | change |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    for ch in chars:
        m = labels == ch
        pc = {v: 100 * res[v][0][m].mean() for v in res}
        rows.append((pc[best] - pc["full"], ch, int(m.sum()), 100 * zero[m].mean(), pc))
    rows.sort(key=lambda r: (-abs(r[0]), r[1]))
    for delta, ch, n, zr, pc in rows:
        print(f"| {ch} | {n} | {zr:.0f} | {pc['full']:.0f} | {pc['content']:.0f} | {pc['crop1']:.0f} | "
              f"{pc['cropN']:.0f} | {pc['cropN+full']:.0f} | {delta:+.0f} |")

    flips_up = int((~res["full"][0] & res[best][0]).sum())
    flips_dn = int((res["full"][0] & ~res[best][0]).sum())
    print(f"\nclip-level flips full -> {best}: {flips_up} fixed, {flips_dn} broken")
    print(f"\nrun time {time.time() - t0:.0f}s. Nothing was written to the index.")
    if control_failed:
        # repeat the banner as the first lines of the file, now the run is done
        tee = sys.stdout
        tee.f.close()
        body = P["report"].read_text(encoding="utf-8")
        P["report"].write_text(banner + "\n" + body, encoding="utf-8")
        tee.f = open(P["report"], "a", encoding="utf-8", buffering=1)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc(file=sys.stdout)
        print(f"\n^ CRASHED. Traceback is also in {paths()['report']}")
        raise
