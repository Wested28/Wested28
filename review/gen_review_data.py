"""
gen_review_data.py -- synthetic test data for review.html. Touches no real
index: it builds a throwaway folder that mirrors the _index layout.

  <out>/frames/<id>_<1|2|3>.jpg     448x448 placeholder frames
  <out>/review/review_data.json     the data file (schema v1)
  <out>/review/review_data.js       same data as `window.REVIEW_DATA = {...};`
                                    -- what review.html loads under file://
Copy review.html into <out>/review/ and open it.

Each placeholder draws a coloured "helmet" figure for its character. The
figure appears in only ONE of the three frames on about a third of the
items -- the case the three-frame view exists for. A few items have no
proposal, one proposes a name that is not in the roster, and one frame
file is deliberately missing, so those paths get exercised too.

usage: python gen_review_data.py <out_dir> [n_items=60] [seed=1]
"""
import hashlib
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROSTER = {  # name -> (helmet colour, body colour)
    "SHARK": ((70, 110, 190), (40, 40, 50)), "LION": ((235, 150, 40), (120, 60, 20)),
    "PIG": ((245, 150, 190), (60, 200, 120)), "BUNNY": ((90, 170, 240), (230, 230, 240)),
    "HARE": ((150, 90, 200), (70, 50, 90)), "TURTLE": ((60, 170, 80), (130, 100, 60)),
    "COW": ((240, 240, 240), (30, 30, 30)), "GOAT": ((200, 190, 150), (90, 80, 60)),
    "WOLF": ((120, 120, 130), (60, 60, 70)), "CHICK": ((250, 220, 60), (240, 240, 200)),
    "IVORY LANCER": ((230, 225, 205), (150, 140, 120)), "OLIVE KNOT": ((120, 130, 40), (50, 60, 20)),
}
ALIASES = {"ORANGE LION": "LION", "BLUE BUNNY": "BUNNY", "PINK PIG": "PIG", "PURPLE HARE": "HARE"}


def fid(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def frame(path, name, cid, k, show, rng):
    bg = tuple(rng.randint(15, 60) for _ in range(3))
    im = Image.new("RGB", (448, 448), bg)
    d = ImageDraw.Draw(im)
    for _ in range(6):  # scenery
        x, y = rng.randint(0, 448), rng.randint(200, 448)
        d.rectangle((x, y, x + rng.randint(20, 120), 448), fill=tuple(c + 20 for c in bg))
    if show and name:
        helm, body = ROSTER[name]
        s = rng.choice([0.35, 0.6, 1.0])          # wide shot vs close
        cx, cy = rng.randint(120, 330), 250
        d.rectangle((cx - 40 * s, cy, cx + 40 * s, cy + 150 * s), fill=body)
        d.ellipse((cx - 45 * s, cy - 90 * s, cx + 45 * s, cy), fill=helm)
        d.rectangle((cx - 30 * s, cy - 55 * s, cx + 30 * s, cy - 40 * s), fill=(250, 40, 40))
    d.text((8, 428), f"{cid}_{k}  {name or '?'}{'' if show else ' (off-screen)'}", fill=(235, 238, 245))
    im.save(path, quality=85)


def main():
    out = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(parents=True, exist_ok=True)
    names = list(ROSTER)

    refs = {}
    for name in names:
        refs[name] = []
        for j in range(rng.randint(2, 6) if name != "OLIVE KNOT" else 0):
            cid = fid(f"ref/{name}/{j}")
            frame(out / "frames" / f"{cid}_2.jpg", name, cid, 2, True, rng)
            refs[name].append(f"../frames/{cid}_2.jpg")

    items = []
    sheets = ["N047", "N048", "V003"]
    for i in range(n):
        sheet, cell = sheets[i * len(sheets) // n], i % (n // len(sheets)) + 1
        truth = rng.choice(names)
        cid = fid(f"clip/{i}")
        only = rng.randint(1, 3) if rng.random() < 0.35 else None
        for k in (1, 2, 3):
            if i == 4 and k == 3:
                continue  # one missing frame on purpose
            frame(out / "frames" / f"{cid}_{k}.jpg", truth, cid, k, only in (None, k), rng)
        r = rng.random()
        proposed = None if r < 0.08 else (truth if r < 0.75 else rng.choice(names))
        if i == 7:
            proposed = "PHOENIX"  # not in roster -> accept must be disabled
        items.append({
            "sheet": sheet, "cell": cell, "id": cid,
            "frames": [f"../frames/{cid}_{k}.jpg" for k in (1, 2, 3)],
            "proposed": proposed,
            "distance": None if proposed is None else round(rng.uniform(0.01, 0.25), 3),
            "agreement": None if proposed is None else rng.randint(1, 3),
        })

    data = {"version": 1, "generated": "synthetic test data", "roster": names,
            "aliases": ALIASES, "items": items, "refs": refs}
    js = json.dumps(data, indent=1)
    (out / "review" / "review_data.json").write_text(js, encoding="utf-8")
    (out / "review" / "review_data.js").write_text(f"window.REVIEW_DATA = {js};\n", encoding="utf-8")
    print(f"{len(items)} items on {len(sheets)} sheets, {len(names)} roster names, "
          f"{sum(len(v) for v in refs.values())} reference frames -> {out}")


if __name__ == "__main__":
    main()
