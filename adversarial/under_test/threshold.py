#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
threshold.py -- what distance cut does the HAND-RULED pile actually support?

WHY THIS EXISTS
    The proposal run calibrates on labelled data and reports a precision.
    That number is a ceiling, not an estimate: every clip in the calibration
    set belongs to one of the reference characters by construction, so it
    cannot see the open-set problem. Measured 2026-10-11: 99.3% calibrated,
    **92.9% on the real pile.**

    The only thing that can tell you where to put the cut is clips a human
    has ruled. This reads a review.html export plus the proposal data the
    batch was built from, and reports the precision at every candidate cut.

\u26a0\ufe0f THE RULED SET IS CENSORED AT THE CUT THAT PRODUCED IT
    Every clip in the batch already passed `--cut 0.05`, so this can only
    find a cut TIGHTER than 0.05. It cannot say anything about 0.06, and it
    will say so rather than let the top of its own range look like a
    finding. To learn whether a WIDER cut is safe, re-run the probe with a
    wider `--cut` and rule that batch.

\u26a0\ufe0f IT ALSO ANSWERS THE QUESTION THE THRESHOLD CANNOT
    All 9 errors in the 127-clip pass were out-of-set characters, and all 9
    sat at d <= 0.05 -- below the median CORRECT match of 0.067. No cut
    separates those. So this ALSO breaks the errors down by how many ruled
    clips the true character has, because that is the lever that worked:
    6 of those 9 were characters excluded by MIN_PER_NAME, and 3 had no
    ruled clips at all and could only be fixed by a human naming one.

RUN
    & "D:\Video Library\_index\ag.cmd" threshold.py <rulings.txt>
    & "D:\Video Library\_index\ag.cmd" threshold.py <rulings.txt> --data <review_data.json>

    The rulings file is review.html's export: `<sheet>/<cells> = <NAME>`,
    with `unknown` allowed. A ruling equal to the proposal (or to one of its
    aliases) counts as correct.

OUTPUT
    logs\threshold.md
"""

import collections
import json
import os
import re
import sys

INDEX = r"D:\Video Library\_index"
DATA = os.path.join(INDEX, "review", "review_data.json")
CHARS = os.path.join(INDEX, "characters.json")
LIB = os.path.join(INDEX, "library.jsonl")
REPORT = os.path.join(INDEX, "logs", "threshold.md")

TARGETS = (0.99, 0.98, 0.95)
MIN_ACCEPTED = 20


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


def arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            return sys.argv[i + 1]
        sys.exit(f"{name} needs a value")
    return default


def expand(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < a:
                raise ValueError(f"reversed range {part!r}")
            out.extend(range(a, b + 1))
            continue
        if not re.fullmatch(r"\d+", part):
            raise ValueError(f"cannot parse cell spec {part!r}")
        out.append(int(part))
    return out


def parse_rulings(path):
    ruled = {}
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line or "/" not in line.split("=")[0]:
                continue
            left, name = line.split("=", 1)
            sheet, spec = left.strip().split("/", 1)
            try:
                cells = expand(spec)
            except ValueError as e:
                raise SystemExit(f"{path}:{ln}: {e}")
            for c in cells:
                ruled[(sheet.strip().upper(), c)] = name.strip().upper()
    return ruled


def ruled_counts():
    """How many solo-ruled clips each character has -- the reference depth
    that MIN_PER_NAME filters on."""
    cnt = collections.Counter()
    if not os.path.exists(LIB):
        return cnt
    with open(LIB, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ns = r.get("ag_characters")
            if isinstance(ns, list) and len(ns) == 1:
                cnt[str(ns[0]).strip().upper()] += 1
    return cnt


def curve(rows, target):
    """Largest cut whose accepted set still meets `target`. Cuts only ever
    fall BETWEEN distinct distances, so a threshold never half-includes a
    tie. Returns (cut, precision, n_accepted) or None."""
    rows = sorted(rows, key=lambda t: t[0])
    best = None
    good = 0
    for k, (d, ok, _nm, _pr) in enumerate(rows):
        good += 1 if ok else 0
        if k + 1 < len(rows) and rows[k + 1][0] == d:
            continue
        n = k + 1
        if n >= MIN_ACCEPTED and good / n >= target:
            best = (d, good / n, n)
    return best


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("usage: threshold.py <rulings.txt> [--data review_data.json]")
    rp = args[0]
    data_path = arg("--data", DATA)
    sys.stdout = Tee(REPORT)

    print("# threshold -- from hand rulings, not calibration\n")
    print(f"python  : {sys.executable}")
    print(f"rulings : {rp}")
    print(f"data    : {data_path}\n")
    for p in (rp, data_path):
        if not os.path.exists(p):
            print(f"!! not found: {p}")
            return 1

    d = json.load(open(data_path, encoding="utf-8"))
    items = d["items"]
    alias = {k.upper(): v.upper() for k, v in (d.get("aliases") or {}).items()}

    def canon(n):
        n = (n or "").strip().upper()
        return alias.get(n, n)

    by_ref = {(it["sheet"].upper(), int(it["cell"])): it for it in items}
    ruled = parse_rulings(rp)
    print(f"batch items : {len(items)}")
    print(f"ruled cells : {len(ruled)}")

    stray = sorted(k for k in ruled if k not in by_ref)
    if stray:
        print(f"\n\u26a0\ufe0f {len(stray)} ruled cell(s) are not in this batch. The")
        print("rulings and the data file are from different runs, so every")
        print("number below would be measured against the wrong clips.")
        for s, c in stray[:8]:
            print(f"   {s}/{c:02d}")
        print("\nPass the review_data.json this batch was built from with")
        print("--data, or re-export the rulings. Stopping.")
        return 1

    rows, unknown = [], []
    for ref, nm in ruled.items():
        it = by_ref[ref]
        prop = canon(it.get("proposed"))
        if nm in ("UNKNOWN", "?"):
            unknown.append((ref, it))
            continue
        rows.append((float(it["distance"]), canon(nm) == prop, canon(nm), prop))

    if not rows:
        print("\nNo decided rulings (every one was `unknown`). Nothing to fit.")
        return 0

    good = sum(1 for _, ok, _, _ in rows if ok)
    print(f"decided     : {len(rows)}   ({good} accepted, "
          f"{len(rows) - good} overridden)")
    print(f"unknown     : {len(unknown)}  (excluded -- 'could not call it' is "
          f"not a verdict on the proposal)")
    print(f"\n**precision on the whole batch: {good / len(rows):.1%}**")

    ds = sorted(x[0] for x in rows)
    cds = sorted(x[0] for x in rows if x[1])
    wds = sorted(x[0] for x in rows if not x[1])
    print(f"\ndistance of CORRECT   : min {cds[0]:.4f} \u00b7 "
          f"median {cds[len(cds) // 2]:.4f} \u00b7 max {cds[-1]:.4f}")
    if wds:
        print(f"distance of WRONG     : min {wds[0]:.4f} \u00b7 "
              f"median {wds[len(wds) // 2]:.4f} \u00b7 max {wds[-1]:.4f}")
        if wds[len(wds) // 2] <= cds[len(cds) // 2]:
            print("\n\u26a0\ufe0f **The wrong ones are no further away than the right")
            print("ones.** A distance cut cannot separate them, so no")
            print("threshold below rescues this and the table is only telling")
            print("you what it costs to be stricter.")
    else:
        print("distance of WRONG     : none -- nothing was overridden")

    print("\n## precision at each candidate cut\n")
    print("Accepting everything at or below the cut. `lost` is correct clips")
    print("a cut throws away, which is what a tighter cut actually costs.\n")
    print("| cut | accepted | correct | precision | lost |")
    print("|---:|---:|---:|---:|---:|")
    for cut in (0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05):
        acc = [x for x in rows if x[0] <= cut]
        if not acc:
            continue
        g = sum(1 for x in acc if x[1])
        print(f"| {cut:.3f} | {len(acc)} | {g} | {g / len(acc):.1%} | "
              f"{good - g} |")

    print("\n## the largest cut that holds each target\n")
    print("| target | cut | precision | accepted |")
    print("|---:|---:|---:|---:|")
    any_hit = False
    for t in TARGETS:
        r = curve(rows, t)
        if r:
            any_hit = True
            print(f"| {t:.0%} | {r[0]:.4f} | {r[1]:.1%} | {r[2]} |")
        else:
            print(f"| {t:.0%} | none | \u2014 | \u2014 |")
    print()
    top = max(ds)
    hit_top = [t for t in TARGETS
               if (curve(rows, t) or (None,))[0] == top]
    if hit_top:
        print(f"\u26a0\ufe0f A target landed on {top:.4f}, the LARGEST distance in the")
        print("batch, which means the cut is the edge of the range tested and")
        print("not a value the data chose. Every clip here already passed the")
        print("probe's own 0.05 cut, so this measurement is CENSORED: it")
        print("cannot see whether 0.06 or 0.08 would also have held. Re-run")
        print("the probe with a wider --cut and rule that batch to find out.")
        print("(This project has twice picked a sweep's edge value and called")
        print("it a result. It is not one.)")
    elif not any_hit:
        print("No cut reaches any target with at least "
              f"{MIN_ACCEPTED} clips accepted.")

    if wds:
        print("\n## the overrides, and why a cut was never going to catch them\n")
        cnt = ruled_counts()
        print("`ruled clips` is how many solo-ruled clips the TRUE character")
        print("has. That is what MIN_PER_NAME filters on: at the old value of")
        print("10, every character below it was absent from the reference set")
        print("and its clips could only ever be given somebody else's name.\n")
        print("| ref | distance | proposed | actually | ruled clips of the "
              "true character |")
        print("|---|---:|---|---|---:|")
        errs = [(ref, by_ref[ref], ruled[ref]) for ref in ruled
                if ruled[ref] not in ("UNKNOWN", "?")
                and canon(ruled[ref]) != canon(by_ref[ref].get("proposed"))]
        errs.sort(key=lambda t: float(t[1]["distance"]))
        for ref, it, nm in errs:
            print(f"| {ref[0]}/{ref[1]:02d} | {float(it['distance']):.4f} | "
                  f"{it.get('proposed')} | {nm} | {cnt.get(canon(nm), 0)} |")
        thin = [e for e in errs if cnt.get(canon(e[2]), 0) == 0]
        few = [e for e in errs if 0 < cnt.get(canon(e[2]), 0) < 10]
        print()
        print(f"- **{len(thin)} of {len(errs)}** name a character with ZERO "
              f"ruled clips. No reference-set setting can readmit a character "
              f"that has no references; a human has to name one clip of it "
              f"first. That is the namer's job.")
        print(f"- **{len(few)} of {len(errs)}** name a character with 1-9 "
              f"ruled clips, i.e. one the old MIN_PER_NAME of 10 excluded.")
        print(f"- **{len(errs) - len(thin) - len(few)} of {len(errs)}** name a "
              f"character with 10 or more. Those are the only true in-set "
              f"confusions, and they are the ones a better EMBEDDING would "
              f"have to fix -- nothing else here can.")

    if unknown:
        print("\n## unknown\n")
        print("Excluded from every number above, because \"looked at, could")
        print("not call it\" is a fact about the clip, not a verdict on the")
        print("proposal. Counting them as errors would understate the method")
        print("and counting them as correct would flatter it.\n")
        for ref, it in sorted(unknown):
            print(f"- {ref[0]}/{ref[1]:02d}  d={float(it['distance']):.4f}  "
                  f"proposed {it.get('proposed')}")

    print("\nNothing was written to the index.")
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
