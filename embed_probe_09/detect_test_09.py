"""
detect_test_09.py -- LOOK at what the person detector finds before trusting
embed_probe_09's numbers.

Runs deepghs person_detect_v1.1_m over a deterministic sample of labelled
midpoint frames (2 per character, sorted by id, plus any frame ids given on
the command line), draws every box at or above the confidence threshold
(green = kept, with score; yellow = below threshold but >= 0.15, so you can
see what the threshold is cutting), and writes:

  logs\\detect_test_09\\<id>.jpg     one annotated frame each
  logs\\detect_test_09\\SHEET.jpg    all of them on one contact sheet
  logs\\detect_test_09.md            per-frame counts and the zero-detection rate

What to look for: are the masked / helmeted figures boxed at all? Is the top
box the SUBJECT, or the enemy / crowd beside her? On wide shots, are the
tiny figures found? A frame with no green box falls back to the full frame
in the probe.

READ-ONLY for the index: reads library.jsonl and frames\\, writes only the
files listed above.

usage: & "D:\\Video Library\\_index\\ag.cmd" detect_test_09.py [frame_id ...]
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_probe_09 as P  # noqa: E402  same decode, same constants

PER_CHAR = 2
LOW_SHOW = 0.15
THUMB = 448
COLS = 6


def main():
    out_dir = P.INDEX / "logs" / "detect_test_09"
    report = P.INDEX / "logs" / "detect_test_09.md"
    sys.stdout = P.Tee(str(report))
    out_dir.mkdir(parents=True, exist_ok=True)
    print("# detect_test_09 -- person detector, annotated\n")
    print("READ-ONLY for the index. Writes only logs\\detect_test_09\\ and this report.\n")
    import onnxruntime as ort
    det_p = P.find_model("detector", P.DET_BYTES,
                         P.model_candidates("anime_person_detection", P.DET_NAME, "model.onnx"))
    prov = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()]
    det = ort.InferenceSession(str(det_p), providers=prov)
    size_wh, why = P.det_input_size(det)
    print(f"providers {det.get_providers()}; input {size_wh} ({why})\n")

    paths = P.paths()
    clips = P.build_set(paths["lib"], paths["frames"])
    by_char = defaultdict(list)
    for c in clips:
        by_char[c[1]].append(c)
    sample = [c for ch in sorted(by_char) for c in by_char[ch][:PER_CHAR]]
    for fid in sys.argv[1:]:
        fp = paths["frames"] / (fid if fid.endswith(".jpg") else f"{fid}_2.jpg")
        if fp.is_file():
            sample.append((fp.stem, "(arg)", fp))
        else:
            print(f"!! not found: {fp}")

    print("\n| frame | character | kept | below-threshold shown | top score | top box (448 px) |")
    print("|---|---|---:|---:|---:|---|")
    thumbs, zero, t0 = [], 0, time.time()
    din = det.get_inputs()[0].name
    for cid, ch, fp in sample:
        img = Image.open(fp).convert("RGB")
        raw = det.run(None, {din: P.det_preprocess(img, size_wh)})[0]
        allb = P.decode(raw, LOW_SHOW, P.IOU_THRESHOLD, img.size, size_wh)
        kept = [b for b in allb if b[4] > P.CONF_THRESHOLD]
        low = [b for b in allb if b[4] <= P.CONF_THRESHOLD]
        zero += not kept
        dr = ImageDraw.Draw(img)
        for b in low:
            dr.rectangle(b[:4], outline=(255, 220, 0), width=1)
        for b in kept:
            dr.rectangle(b[:4], outline=(0, 255, 0), width=2)
            dr.text((b[0] + 2, b[1] + 1), f"{b[4]:.2f}", fill=(0, 255, 0))
            pb = P.pad_box(b, *img.size)
            dr.rectangle(pb, outline=(0, 160, 255), width=1)
        dr.text((4, 4), f"{ch}  {cid}", fill=(255, 255, 255))
        img.save(out_dir / f"{cid}.jpg", quality=90)
        thumbs.append(img.resize((THUMB, THUMB)))
        top = kept[0] if kept else None
        print(f"| {cid} | {ch} | {len(kept)} | {len(low)} | "
              f"{top[4]:.3f} | {tuple(int(v) for v in top[:4])} |" if top else
              f"| {cid} | {ch} | 0 | {len(low)} | - | - |")
    n = len(sample)
    print(f"\nframes with ZERO kept detections: {zero}/{n} = {100 * zero / max(n, 1):.0f}%")
    print("green = kept (score shown), blue = the padded crop the probe embeds, "
          "yellow = below threshold.")
    if thumbs:
        rows = (len(thumbs) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * THUMB, rows * THUMB), (20, 20, 20))
        for i, t in enumerate(thumbs):
            sheet.paste(t, ((i % COLS) * THUMB, (i // COLS) * THUMB))
        sheet.save(out_dir / "SHEET.jpg", quality=85)
        print(f"contact sheet: {out_dir / 'SHEET.jpg'}")
    print(f"run time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc(file=sys.stdout)
        print("\n^ CRASHED. Traceback is also in logs\\detect_test_09.md")
        raise
