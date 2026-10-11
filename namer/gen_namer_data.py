"""
gen_namer_data.py -- synthetic data for namer.html. Touches no real index.

  <out>/frames/<id>_<1|2|3>.jpg   448x448 placeholders
  <out>/namer/namer_data.js       window.NAMER_DATA = {...}
  <out>/namer/namer_truth.json    the hidden answers, for the test driver only

Shape (scaled by --scale, default 1 = the brief's numbers):
  110 roster names (incl. the four real near-miss pairs' existing halves and
      M-names, so the M-key bug and the guard can be exercised)
  40 groups / 200 clips, from 22 unseen characters -- 8 of them deliberately
      OVER-SPLIT across 2-3 groups, as single-link clustering does
  160 singletons: 40 from 10 unseen characters that appear 4x each (the
      "do any of these belong together" case), 30 more from the grouped
      characters, 40 with no character at all, 50 one-off characters

usage: python gen_namer_data.py <out_dir> [scale=1] [seed=1]
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REAL = ["ROSE LYNX", "IVORY LANCER", "ONYX MAID", "LA BRUINE", "MONKEY PRIME", "MAGENTA PILOT",
        "MIDNIGHT JAZZ", "GREEN GATOR", "YELLOW GECKO", "BLUE SHARK", "PINK PIG", "ORANGE LION"]
COLOURS = ["RED", "BLUE", "GREEN", "GOLD", "SILVER", "BLACK", "WHITE", "PURPLE", "TEAL", "AMBER",
           "CRIMSON", "AZURE", "COBALT", "OLIVE", "IVORY", "JADE", "COPPER", "SCARLET", "VIOLET", "ONYX"]
BEASTS = ["FOX", "OWL", "BEAR", "WOLF", "HAWK", "CRANE", "VIPER", "TOAD", "STAG", "BOAR", "MOTH",
          "WASP", "OTTER", "SEAL", "RAM", "MULE", "IBIS", "NEWT", "KOI", "YAK"]


def frame(path, look, show, label, rng):
    bg = tuple(rng.randint(15, 60) for _ in range(3))
    im = Image.new("RGB", (448, 448), bg)
    d = ImageDraw.Draw(im)
    for _ in range(6):
        x, y = rng.randint(0, 448), rng.randint(180, 448)
        d.rectangle((x, y, x + rng.randint(20, 120), 448), fill=tuple(min(255, c + 22) for c in bg))
    if look and show:
        helm, body, visor = look
        s = rng.choice([0.4, 0.65, 1.0])
        cx, cy = rng.randint(130, 320), 240
        d.rectangle((cx - 40 * s, cy, cx + 40 * s, cy + 150 * s), fill=body)
        d.ellipse((cx - 46 * s, cy - 92 * s, cx + 46 * s, cy), fill=helm)
        d.rectangle((cx - 30 * s, cy - 56 * s, cx + 30 * s, cy - 40 * s), fill=visor)
    d.text((8, 430), label, fill=(235, 238, 245))
    im.save(path, quality=82)


def main():
    out = Path(sys.argv[1]); scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    (out / "frames").mkdir(parents=True, exist_ok=True); (out / "namer").mkdir(parents=True, exist_ok=True)
    roster = list(REAL)
    while len(roster) < 110:
        n = f"{rng.choice(COLOURS)} {rng.choice(BEASTS)}"
        if n not in roster:
            roster.append(n)
    look = lambda: (tuple(rng.randint(40, 255) for _ in range(3)), tuple(rng.randint(20, 200) for _ in range(3)),
                    tuple(rng.randint(150, 255) for _ in range(3)))
    unseen = {f"UNSEEN {k:02d}": look() for k in range(200)}
    names = list(unseen)
    S = lambda n: max(1, int(round(n * scale)))
    clips = []      # (truth, group)
    # 40 groups / 200 clips from 22 characters; first 8 characters over-split into 2-3 groups
    gid = 0
    plan = []
    for c in range(22):
        parts = 3 if c < 4 else 2 if c < 8 else 1
        plan += [names[c]] * parts
    plan = plan[:40] + [names[22 + k] for k in range(max(0, 40 - len(plan)))]
    for g_i in range(S(40)):
        truth = plan[g_i % len(plan)]
        for _ in range(5):
            clips.append((truth, gid))
        gid += 1
    # singletons
    for c in range(S(10)):
        for _ in range(4):
            clips.append((names[40 + c], None))           # belong together, 4 each
    for _ in range(S(30)):
        clips.append((rng.choice(names[:22]), None))      # escaped a group
    for _ in range(S(40)):
        clips.append((None, None))                        # no character at all
    for c in range(S(50)):
        clips.append((names[60 + (c % 140)], None))       # one-offs
    order = list(range(len(clips)))
    items, truth = [], {}
    gsize = {}
    for t, g in clips:
        if g is not None:
            gsize[g] = gsize.get(g, 0) + 1
    for n, k in enumerate(order):
        t, g = clips[k]
        cid = f"{rng.getrandbits(48):012x}"
        sheet, cell = f"U{1 + n // 20:03d}", n % 20 + 1
        only = rng.randint(1, 3) if rng.random() < 0.3 else None
        for f in (1, 2, 3):
            frame(out / "frames" / f"{cid}_{f}.jpg", unseen.get(t), only in (None, f), f"{cid}_{f}", rng)
        near = rng.choice(roster)
        items.append({"id": cid, "sheet": sheet, "cell": cell,
                      "frames": [f"../frames/{cid}_{f}.jpg" for f in (1, 2, 3)],
                      "proposed": None, "nearest": near, "distance": round(rng.uniform(0.12, 0.3), 3),
                      "group": g, "group_size": gsize.get(g) if g is not None else None})
        truth[cid] = t
    data = {"roster": roster, "items": items}
    (out / "namer" / "namer_data.js").write_text("window.NAMER_DATA = " + json.dumps(data) + ";\n")
    (out / "namer" / "namer_truth.json").write_text(json.dumps(truth))
    ng = len({it["group"] for it in items if it["group"] is not None})
    print(f"{len(items)} items: {ng} groups / {sum(1 for it in items if it['group'] is not None)} grouped clips, "
          f"{sum(1 for it in items if it['group'] is None)} singletons; {len(roster)} roster names -> {out}")


if __name__ == "__main__":
    main()
