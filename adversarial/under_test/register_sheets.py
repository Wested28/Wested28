#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
register_sheets.py -- give a sheet/cell ADDRESS to clips that have none.

WHY
    cells.txt rules clips by `<sheet>/<cells> = <NAME>`, and that address
    resolves through charpass/cluster_index.json, where cell N of a sheet is
    ids[N-1]. A clip that appears in no entry there has no address, so it
    cannot be ruled at all. gen_namer_data.py reports these and refuses to
    invent a ref for them -- correctly, because a made-up ref would point at
    somebody else's clip.

    The 2026-10-11 scan surfaced 82 such clips.

⚠️ THIS IS THE ONE FILE THAT CAN RE-POINT EXISTING RULINGS
    Every other tool in this project treats cluster_index.json as read-only.
    An entry's `ids` order IS the cell numbering, so reordering or inserting
    into an existing entry silently re-points every ruling already written
    against that sheet. This script therefore ONLY APPENDS NEW ENTRIES under
    NEW sheet codes, never touches an existing one, and proves it afterwards
    by re-reading the file and comparing the old entries element by element.

WHAT A SHEET IS HERE
    A label, not a picture. The existing `sheet` values (`A001_red-unknown.jpg`)
    do not match the image filenames (sheets/0001.jpg ... 0305.jpg), so the
    field has long been an address rather than a path. These entries are
    labelled `U###_unsheeted.jpg` and no image is built: the namer renders
    `frames/<id>_N.jpg` directly, and inventing a new image numbering
    alongside the 0001-0305 series would be a trap for later.

SAFETY, all enforced before anything is written
    * refuses a clip whose 8-char prefix does not resolve to exactly ONE
      library id -- that is the lookup gen_namer_data.py and
      charpass_cells.py both use, and an ambiguous prefix would resolve to
      the wrong clip (measured 2026-10-11: 0 ambiguous of 4,612)
    * refuses a clip already registered anywhere, so a re-run is a no-op
    * refuses a sheet code that already exists
    * refuses a clip with no frame on disk, because the namer would show a
      blank cell and there would be nothing to judge
    * backs up cluster_index.json with the project's own `.<stamp>.bak`
      naming BEFORE writing
    * after writing: re-reads the file and asserts the old entries are
      unchanged, the count grew by exactly the number of new sheets, and
      every new clip resolves back to its intended (sheet, cell) through an
      INDEPENDENT reimplementation of the lookup

    Order is `sorted(clip_id)`, so the addresses are deterministic: a re-run
    on the same inputs would assign the same cells.

RUN
    & "D:\Video Library\_index\ag.cmd" register_sheets.py
    & "D:\Video Library\_index\ag.cmd" register_sheets.py --write

    --csv  <path>   source of clip ids   (default logs\unnamed_scan.csv)
    --code <letter> new sheet code prefix (default U)
    --per  <n>      clips per sheet       (default 20, matching the A sheets)

OUTPUT
    charpass\cluster_index.json          (appended, with a .bak beside it)
    logs\register_sheets.md
"""

import collections
import csv
import datetime
import json
import os
import shutil
import sys

INDEX = r"D:\Video Library\_index"
CP = os.path.join(INDEX, "charpass")
CIDX = os.path.join(CP, "cluster_index.json")
LIB = os.path.join(INDEX, "library.jsonl")
FRAMES = os.path.join(INDEX, "frames")
DEFAULT_CSV = os.path.join(INDEX, "logs", "unnamed_scan.csv")
REPORT = os.path.join(INDEX, "logs", "register_sheets.md")

PER_SHEET = 20
CODE = "U"


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


def flag(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("--"):
            sys.exit(f"{name} needs a value")
        return sys.argv[i + 1]
    return default


def code_of(sheet):
    """The one place a sheet string becomes a code. Must stay identical to
    gen_namer_data.sheet_refs() and charpass_cells.py, or an address written
    by one tool will not be found by the other."""
    return sheet.rsplit(".", 1)[0].split("_")[0].upper()


def library_prefixes():
    by = collections.defaultdict(list)
    with open(LIB, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("id"):
                by[r["id"][:8]].append(r["id"])
    return by


def frames_of(cid):
    return [k for k in (1, 2, 3)
            if os.path.exists(os.path.join(FRAMES, f"{cid}_{k}.jpg"))]


def resolve_independently(entries, bypre):
    """A SECOND implementation of the sheet/cell lookup, written from the
    rule rather than copied from the readers: cell N of a sheet is ids[N-1],
    first sheet wins when a clip is listed twice. Used only to check the
    written file, so that a bug shared with the readers cannot hide."""
    ref = {}
    for e in entries:
        c = code_of(e["sheet"])
        for n, pid in enumerate(e.get("ids", []), start=1):
            cand = bypre.get(pid[:8], [])
            if len(cand) == 1 and cand[0] not in ref:
                ref[cand[0]] = (c, n)
    return ref


def main():
    write = "--write" in sys.argv
    src = flag("--csv", DEFAULT_CSV)
    code = flag("--code", CODE).upper()
    per = int(flag("--per", str(PER_SHEET)))
    if not src or not os.path.isabs(src):
        src = os.path.join(INDEX, src or "")
    sys.stdout = Tee(REPORT)

    print("# register_sheets\n")
    print(f"python : {sys.executable}")
    print(f"source : {src}")
    print(f"index  : {CIDX}\n")
    for p in (CIDX, LIB, src):
        if not os.path.exists(p):
            print(f"!! not found: {p}")
            return 1

    before = json.load(open(CIDX, encoding="utf-8"))
    registered = {p for e in before for p in e.get("ids", [])}
    codes = {code_of(e["sheet"]) for e in before}
    print(f"entries now          : {len(before)}")
    print(f"registered prefixes  : {len(registered)}")
    print(f"existing codes       : {len(codes)}  "
          f"({''.join(sorted({c[0] for c in codes}))})")
    if len(codes) != len(before):
        print("\n!! two entries share a sheet code. Not appending to a file "
              "whose addresses are already ambiguous.")
        return 1

    bypre = library_prefixes()
    amb = sum(1 for v in bypre.values() if len(v) > 1)
    print(f"library ids          : {sum(len(v) for v in bypre.values())}  "
          f"({amb} ambiguous 8-char prefixes)")

    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    want = []
    seen = set()
    for r in rows:
        cid = (r.get("clip_id") or "").strip()
        if cid and cid not in seen:
            seen.add(cid)
            want.append(cid)
    print(f"clips in source      : {len(want)}\n")

    todo, refused = [], []
    for cid in sorted(want):
        if cid[:8] in registered:
            continue                                  # already has an address
        cand = bypre.get(cid[:8], [])
        if len(cand) != 1:
            refused.append((cid, f"prefix resolves to {len(cand)} library ids"))
            continue
        if cand[0] != cid:
            refused.append((cid, f"prefix belongs to {cand[0]}"))
            continue
        fr = frames_of(cid)
        if not fr:
            refused.append((cid, "no frames on disk"))
            continue
        todo.append(cid)

    print(f"already addressed    : {len(want) - len(todo) - len(refused)}")
    print(f"TO REGISTER          : {len(todo)}")
    print(f"refused              : {len(refused)}\n")
    if refused:
        print("| clip | why refused |")
        print("|---|---|")
        for cid, why in refused[:40]:
            print(f"| `{cid}` | {why} |")
        print()

    if not todo:
        print("Nothing to register. The source's clips all have addresses.")
        return 0

    # plan the new sheets -- deterministic, sorted by clip id
    nmax = 0
    for c in codes:
        if c.startswith(code) and c[len(code):].isdigit():
            nmax = max(nmax, int(c[len(code):]))
    plan = []
    for k in range(0, len(todo), per):
        chunk = todo[k:k + per]
        nmax += 1
        sheet = f"{code}{nmax:03d}_unsheeted.jpg"
        sc = code_of(sheet)
        if sc in codes:
            print(f"!! code {sc} already exists. Pick another with --code.")
            return 1
        codes.add(sc)
        plan.append({
            "cluster": -1,
            "track": "unsheeted",
            "descs": ["registered by register_sheets.py so these clips have "
                      "a cells.txt address; no sheet image exists"],
            "sheet": sheet,
            "ids": [c[:8] for c in chunk],
        })

    print(f"new sheets           : {len(plan)}  "
          f"({plan[0]['sheet'].split('_')[0]}"
          f"{'' if len(plan) == 1 else ' .. ' + plan[-1]['sheet'].split('_')[0]})"
          f", up to {per} clips each\n")
    print("| ref | clip |")
    print("|---|---|")
    for e in plan:
        for n, pid in enumerate(e["ids"], 1):
            print(f"| {code_of(e['sheet'])}/{n:02d} | `{bypre[pid][0]}` |")
    print()

    if not write:
        print("**REPORT ONLY -- nothing written. Re-run with `--write`.**")
        print("\nNo existing entry would be touched; this only appends.")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = f"{CIDX}.{stamp}.bak"
    shutil.copy2(CIDX, bak)
    print(f"backup -> {bak}")

    after = before + plan
    tmp = CIDX + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(after, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, CIDX)
    print(f"wrote  -> {CIDX}")

    # --- prove it, against the file as it now is on disk -------------------
    print("\n## verification (re-read from disk)\n")
    back = json.load(open(CIDX, encoding="utf-8"))
    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"- {'ok  ' if cond else 'FAIL'}  {label}")
        ok = ok and bool(cond)

    check(f"entry count {len(before)} -> {len(back)} "
          f"(+{len(plan)} expected)", len(back) == len(before) + len(plan))
    check("every pre-existing entry is byte-for-byte unchanged",
          back[:len(before)] == before)
    check("every sheet code still unique",
          len({code_of(e["sheet"]) for e in back}) == len(back))

    ref = resolve_independently(back, bypre)
    intended = {}
    for e in plan:
        for n, pid in enumerate(e["ids"], 1):
            intended[bypre[pid][0]] = (code_of(e["sheet"]), n)
    wrong = {c: (intended[c], ref.get(c)) for c in intended
             if ref.get(c) != intended[c]}
    check(f"all {len(intended)} new clips resolve to their intended ref",
          not wrong)
    for c, (want_, got) in list(wrong.items())[:10]:
        print(f"    {c}: wanted {want_}, got {got}")

    # the whole point of the exercise: the OLD addresses must not have moved
    oldref = resolve_independently(before, bypre)
    moved = {c: (oldref[c], ref.get(c)) for c in oldref if ref.get(c) != oldref[c]}
    check(f"none of the {len(oldref)} existing clip addresses moved", not moved)
    for c, (was, now) in list(moved.items())[:10]:
        print(f"    {c}: was {was}, now {now}")

    print()
    if ok:
        print("All checks passed.\n")
        print("Next:")
        print("  gen_namer_data.py --scan logs\\unnamed_scan.csv --write")
        print("  then reload namer.html -- the refused count should drop by "
              f"{len(todo)}.")
    else:
        print("**A CHECK FAILED. Restore the backup before doing anything "
              "else:**")
        print(f"  Copy-Item -Force '{bak}' '{CIDX}'")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stdout)
        print(f"\n^ CRASHED. Traceback is also in {REPORT}")
        raise
