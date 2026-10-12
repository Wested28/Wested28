"""
test_threshold.py -- adversarial tests for threshold.py.

The script is NOT modified: imported from a byte-identical copy (SHA-256
checked before and after), with DATA / CHARS / LIB / REPORT pointed at a
throwaway tree per test. Every precision the report prints is checked
against an independent recomputation, because these numbers decide what
precision gets believed.

usage: python test_threshold.py
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "under_test" / "threshold.py"
SHA = "f47cdf7d74690ed35065c302bc21d8a7c5e9d7d09fff0aa476b34f42c894b406"
RESULTS = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def record(name, verdict, claim, observed, belief=""):
    RESULTS.append((name, verdict, claim, observed, belief))
    print(f"[{verdict}] {name}\n    claim   : {claim}\n    observed: {observed}"
          + (f"\n    you'd believe: {belief}" if belief else ""))


def run(items, rulings_text, aliases=None, lib_names=None, rulings_bytes=None):
    root = Path(tempfile.mkdtemp(prefix="thresh_"))
    (root / "logs").mkdir()
    data = root / "review_data.json"
    data.write_text(json.dumps({"roster": [], "aliases": aliases or {}, "items": items}), encoding="utf-8")
    rp = root / "rulings.txt"
    if rulings_bytes is not None:
        rp.write_bytes(rulings_bytes)
    else:
        rp.write_text(rulings_text, encoding="utf-8")
    with open(root / "library.jsonl", "w", encoding="utf-8") as f:
        for k, nm in enumerate(lib_names or []):
            f.write(json.dumps({"id": f"{k:012x}", "ag_characters": [nm]}) + "\n")
    spec = importlib.util.spec_from_file_location("threshold_ut", SCRIPT)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.DATA, m.LIB, m.REPORT = str(data), str(root / "library.jsonl"), str(root / "logs" / "threshold.md")
    m.CHARS = str(root / "characters.json")
    old_argv, old_out, old_dunder = sys.argv, sys.stdout, sys.__stdout__
    sys.argv = ["threshold.py", str(rp), "--data", str(data)]
    rc, exc, err = None, None, io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            sys.__stdout__ = io.StringIO()
            rc = m.main()
    except BaseException as e:                       # noqa: BLE001
        exc = e
    finally:
        sys.stdout, sys.argv, sys.__stdout__ = old_out, old_argv, old_dunder
    rep = Path(m.REPORT).read_text(encoding="utf-8") if Path(m.REPORT).exists() else ""
    return rc, exc, rep


def item(cell, prop, dist, sheet="U001"):
    return {"sheet": sheet, "cell": cell, "id": f"{sheet}{cell:04d}".lower().ljust(12, "0")[:12],
            "proposed": prop, "distance": dist}


def lines_for(rulings):
    """rulings: list of (cell, NAME) on U001 -> one line per cell"""
    return "".join(f"U001/{c:02d} = {n}\n" for c, n in rulings)


def printed_precision(rep):
    m = re.search(r"precision on the whole batch: ([\d.]+)%", rep)
    return float(m.group(1)) if m else None


def target_rows(rep):
    out = {}
    for t, cut, p, n in re.findall(r"\| (\d+)% \| ([\d.]+|none) \| ([\d.]+%|—) \| (\d+|—) \|", rep):
        out[int(t)] = None if cut == "none" else (float(cut), float(p.rstrip("%")), int(n))
    return out


def indep_curve(rows, target, min_n=20):
    rows = sorted(rows)
    best, good = None, 0
    for k, (d, ok) in enumerate(rows):
        good += ok
        if k + 1 < len(rows) and rows[k + 1][0] == d:
            continue
        if k + 1 >= min_n and good / (k + 1) >= target:
            best = (d, round(100 * good / (k + 1), 1), k + 1)
    return best


# ------------------------------------------------------------------ tests
def t_baseline():
    items, rul, rows = [], [], []
    for c in range(1, 41):
        d = round(0.002 + c * 0.0011, 4)
        wrong = c in (7, 19, 30, 33, 38)
        items.append(item(c, "SHARK", d))
        rul.append((c, "TURTLE" if wrong else "SHARK"))
        rows.append((d, not wrong))
    rc, exc, rep = run(items, lines_for(rul))
    p = printed_precision(rep)
    tr = target_rows(rep)
    want = {t: indep_curve(rows, t / 100) for t in (99, 98, 95)}
    ok = p == 87.5 and all((tr[t] is None) == (want[t] is None) and
                            (tr[t] is None or (abs(tr[t][0] - want[t][0]) < 1e-9 and tr[t][2] == want[t][2]))
                            for t in want)
    record("baseline: 40 decided, 5 overridden", "HELD" if ok else "BUG",
           "whole-batch precision and per-target cuts are right",
           f"printed {p}% (independent 87.5%); targets {tr} vs independent {want}")


def t_ties_at_cut():
    items, rul, rows = [], [], []
    c = 0
    for k in range(10):                               # 10 correct well inside
        c += 1; items.append(item(c, "SHARK", 0.01)); rul.append((c, "SHARK")); rows.append((0.01, True))
    for k in range(25):                               # 25 TIED at exactly 0.03, 3 of them wrong
        c += 1; wrong = k < 3
        items.append(item(c, "SHARK", 0.03)); rul.append((c, "TURTLE" if wrong else "SHARK")); rows.append((0.03, not wrong))
    for k in range(5):
        c += 1; items.append(item(c, "SHARK", 0.035)); rul.append((c, "SHARK")); rows.append((0.035, True))
    items = [dict(x, sheet="U001") for x in items]
    rc, exc, rep = run(items, "".join(f"U001/{k:02d} = {n}\n" for k, n in rul))
    row030 = re.search(r"\| 0\.030 \| (\d+) \| (\d+) \|", rep)
    row035 = re.search(r"\| 0\.035 \| (\d+) \| (\d+) \|", rep)
    tr = target_rows(rep)
    want = {t: indep_curve(rows, t / 100) for t in (99, 98, 95)}
    ok = row030 and row030.group(1) == "35" and row035 and row035.group(1) == "40" and \
        all((tr[t] is None) == (want[t] is None) for t in want)
    record("25 clips tied at exactly d = 0.030 (3 wrong), cut rows 0.030 / 0.035",
           "HELD" if ok else "BUG", "a cut never half-includes a tie; <= includes the tie",
           f"row 0.030 accepted {row030 and row030.group(1)} (want 35), row 0.035 accepted "
           f"{row035 and row035.group(1)} (want 40); targets {tr} vs independent {want}")


def t_alias_both_ways():
    items = [item(1, "LION", 0.01), item(2, "ORANGE LION", 0.01)] + [item(c, "SHARK", 0.01) for c in range(3, 23)]
    rul = [(1, "ORANGE LION"), (2, "LION")] + [(c, "SHARK") for c in range(3, 23)]
    rc, exc, rep = run(items, lines_for(rul), aliases={"Orange Lion": "Lion"})
    ok = "0 overridden" in rep and printed_precision(rep) == 100.0
    record("alias ruled against canonical, and canonical against alias (mixed case in the alias map)",
           "HELD" if ok else "BUG", "a ruling equal to the proposal or its alias counts as correct",
           f"'0 overridden' in report: {'0 overridden' in rep}; precision {printed_precision(rep)}%")


def t_cell_specs():
    items = [item(c, "SHARK", 0.01) for c in range(1, 21)]
    rc, exc, rep = run(items, "U001/01-04,07 = SHARK\nu001/ 9 - 11 , 13 = SHARK\n")
    ok = "ruled cells : 9" in rep
    record("cells spec '01-04,07' and ' 9 - 11 , 13' with a lowercase sheet", "HELD" if ok else "BUG",
           "expands to the listed cells", f"'ruled cells : 9' in report: {ok}")


def t_reversed_range():
    items = [item(c, "SHARK", 0.01) for c in range(1, 21)]
    rc, exc, rep = run(items, "U001/04-01 = SHARK\n")
    stopped = isinstance(exc, SystemExit)
    in_report = "reversed range" in rep
    record("reversed range '04-01'", "BUG" if stopped and not in_report else "HELD",
           "stops, and says why in logs\\threshold.md (the house rule: the report file is the record)",
           f"stops: {stopped}; reason printed: only to the console (SystemExit message); in the report file: {in_report}. "
           f"The report ends after its header lines with no error.",
           "reading the report off disk: a run that stopped for no stated reason.")


def t_cell_beyond_sheet():
    items = [item(c, "SHARK", 0.01) for c in range(1, 21)]
    rc, exc, rep = run(items, "U001/01-20 = SHARK\nU001/25 = SHARK\n")
    ok = rc == 1 and "not in this batch" in rep and "Stopping" in rep
    record("cell number beyond the sheet's length (U001/25 on a 20-cell sheet)", "HELD" if ok else "BUG",
           "refuses to measure against the wrong clips", f"stops with the 'not in this batch' message: {ok}")


def t_crlf_and_bom():
    items = [item(c, "SHARK", 0.01) for c in range(1, 21)]
    body = "".join(f"U001/{c:02d} = SHARK\r\n" for c in range(1, 21))
    rc1, exc1, rep1 = run(items, None, rulings_bytes=body.encode())
    rc2, exc2, rep2 = run(items, None, rulings_bytes=b"\xef\xbb\xbf" + body.encode())
    crlf_ok = "ruled cells : 20" in rep1 and printed_precision(rep1) == 100.0
    bom_stray = "1 ruled cell(s) are not in this batch" in rep2 and "different runs" in rep2
    record("CRLF line endings", "HELD" if crlf_ok else "BUG", "parsed the same as LF",
           f"20 cells, 100.0%: {crlf_ok}")
    record("UTF-8 BOM at the start of the rulings file", "BUG" if bom_stray else "HELD",
           "parses the first line",
           "the BOM sticks to the first sheet code ('\\ufeffU001'), that cell looks foreign, and the run stops "
           "with 'The rulings and the data file are from different runs'",
           "you passed the wrong --data file and go hunting for another. The data file was right.")


def t_all_unknown():
    items = [item(c, "SHARK", 0.01) for c in range(1, 11)]
    rc, exc, rep = run(items, "U001/01-10 = unknown\n")
    ok = rc == 0 and exc is None and "No decided rulings" in rep
    record("every ruling 'unknown'", "HELD" if ok else "BUG", "says there is nothing to fit",
           f"rc={rc}, 'No decided rulings' printed: {'No decided rulings' in rep}")


def t_one_decided():
    items = [item(1, "SHARK", 0.02)] + [item(c, "SHARK", 0.02) for c in range(2, 6)]
    rc, exc, rep = run(items, "U001/01 = SHARK\nU001/02-05 = unknown\n")
    p = printed_precision(rep)
    caveat = re.search(r"(n\s*=|only \d+|too few|sample)", rep.split("precision on the whole batch")[1][:200], re.I) if p else None
    record("exactly one decided clip, ruled correct", "BUG" if p == 100.0 and not caveat else "HELD",
           "the precision it prints can be believed",
           f"prints '**precision on the whole batch: 100.0%**' in bold from ONE clip, with no sample-size caveat; "
           f"every cut row also reads 100.0%",
           "the method is 100% precise. One clip can't say that.")
    rc2, exc2, rep2 = run(items, "U001/01 = TURTLE\nU001/02-05 = unknown\n")
    record("exactly one decided clip, ruled WRONG", "BUG" if isinstance(exc2, IndexError) else "HELD",
           "reports 0.0%", f"crashes: {type(exc2).__name__ if exc2 else None} at 'distance of CORRECT' (cds[0] on an empty list)",
           "no number at all. The run crashes on the batch that matters most.")


def t_all_wrong():
    items = [item(c, "SHARK", 0.01 + c * 0.001) for c in range(1, 26)]
    rc, exc, rep = run(items, "U001/01-25 = TURTLE\n")
    record("25 decided, every one overridden", "BUG" if isinstance(exc, IndexError) else "HELD",
           "reports 0% and the override table",
           f"crashes: {type(exc).__name__ if exc else None} -- same empty-list index as above",
           "no measurement of a batch where the method was always wrong.")


def t_target_below_minimum():
    items, rul = [], []
    for c in range(1, 16):
        items.append(item(c, "SHARK", 0.001 * c)); rul.append((c, "SHARK"))
    for c in range(16, 31):
        wrong = c % 3 == 0
        items.append(item(c, "SHARK", 0.02 + 0.001 * c)); rul.append((c, "TURTLE" if wrong else "SHARK"))
    rc, exc, rep = run(items, lines_for(rul))
    tr = target_rows(rep)
    rows = [(0.001 * c, True) for c in range(1, 16)] + [(0.02 + 0.001 * c, c % 3 != 0) for c in range(16, 31)]
    want = {t: indep_curve(rows, t / 100) for t in (99, 98, 95)}
    same = all((tr[t] is None) == (want[t] is None) and
               (tr[t] is None or (abs(tr[t][0] - want[t][0]) < 1e-9 and tr[t][2] == want[t][2])) for t in want)
    ok = tr.get(99) is None and tr.get(98) is None and same
    record("99% only reachable by accepting 15 clips (< MIN_ACCEPTED 20)", "HELD" if ok else "BUG",
           "never reports a cut backed by fewer than 20 clips; the 95% cut it does report is right",
           f"targets: {tr}; independent: {want}")


def t_no_proposal():
    items = [item(c, "SHARK", 0.01) for c in range(1, 21)] + [item(21, None, 0.012)]
    rc, exc, rep = run(items, lines_for([(c, "SHARK") for c in range(1, 21)] + [(21, "PINK PIG")]))
    counted_wrong = "1 overridden" in rep
    record("a clip with NO proposal (proposed: null) that you named", "BUG" if counted_wrong else "HELD",
           "measures the precision of PROPOSALS",
           f"counted as an override: {counted_wrong}; precision printed {printed_precision(rep)}% instead of 100.0%; "
           f"it also appears in the overrides table as proposed 'None'",
           "the machine got one wrong. It never proposed anything for that clip.")
    items2 = items[:20] + [dict(item(21, None, 0.0), distance=None)]
    rc2, exc2, rep2 = run(items2, lines_for([(c, "SHARK") for c in range(1, 21)] + [(21, "PINK PIG")]))
    record("same, with distance: null", "BUG" if isinstance(exc2, TypeError) else "HELD",
           "skips or reports it", f"crashes: {type(exc2).__name__ if exc2 else None} at float(None)",
           "no report at all for the whole batch.")


def t_group_ruling():
    items = [item(c, "SHARK", 0.01) for c in range(1, 22)]
    rul = [(c, "SHARK") for c in range(1, 21)] + [(21, "GROUP")]
    rc, exc, rep = run(items, lines_for(rul), lib_names=["GROUP"] * 12)
    counted = "1 overridden" in rep
    inset = "**1 of 1** name a character with 10 or more" in rep
    record("a clip ruled GROUP (review.html key 4) when the proposal named one character",
           "BUG" if counted and inset else "HELD",
           "errors are classified by why they happened",
           f"counted as an override: {counted}; and with 12 GROUP clips in the library it is classed as a "
           f"'true in-set confusion ... a better EMBEDDING would have to fix': {inset}",
           "the embedding confuses characters. GROUP is a category, not a character.")


def t_conflicting_lines():
    items = [item(c, "SHARK", 0.01) for c in range(1, 21)]
    rc, exc, rep = run(items, "U001/01-20 = SHARK\nU001/05 = TURTLE\n")
    warned = re.search(r"ruled twice|duplicate|conflict|more than once", rep, re.I)
    silent = "1 overridden" in rep and not warned
    record("the same cell ruled twice with different names", "BUG" if silent else "HELD",
           "every ruling is accounted for",
           "the later line silently wins (dict overwrite); no warning that a cell was ruled twice",
           "a clean 20-cell file. One cell was contradicted.")


def t_malformed_line():
    items = [item(c, "SHARK", 0.01) for c in range(1, 21)]
    rc, exc, rep = run(items, "U001/01-19 = SHARK\nU001-20 = SHARK\n")
    silent = "ruled cells : 19" in rep and "skipped" not in rep.lower()
    record("a malformed line ('U001-20 = SHARK', a dash for a slash)", "BUG" if silent else "HELD",
           "every line is used or reported",
           "silently dropped: 19 cells read, no count of skipped lines",
           "all your rulings were counted.")


def t_censored_top():
    items = [item(c, "SHARK", round(0.001 * c, 4)) for c in range(1, 31)]
    rc, exc, rep = run(items, lines_for([(c, "SHARK") for c in range(1, 31)]))
    ok = "CENSORED" in rep
    record("every clip correct: the cut lands on the batch's largest distance", "HELD" if ok else "BUG",
           "flags that the cut is the edge of the tested range", f"CENSORED warning printed: {ok}")


def main():
    assert sha(SCRIPT) == SHA, "script under test differs from the upload"
    for t in (t_baseline, t_ties_at_cut, t_alias_both_ways, t_cell_specs, t_reversed_range, t_cell_beyond_sheet,
              t_crlf_and_bom, t_all_unknown, t_one_decided, t_all_wrong, t_target_below_minimum, t_no_proposal,
              t_group_ruling, t_conflicting_lines, t_malformed_line, t_censored_top):
        try:
            t()
        except Exception as e:
            record(t.__name__, "HARNESS-ERROR", "-", repr(e))
    assert sha(SCRIPT) == SHA, "script under test was modified during the run"
    bugs = [r for r in RESULTS if r[1] == "BUG"]
    print(f"\n{len(RESULTS)} tests: {len(bugs)} BUG, {sum(r[1] == 'HELD' for r in RESULTS)} HELD, "
          f"{sum(r[1] == 'HARNESS-ERROR' for r in RESULTS)} harness errors. Script SHA-256 unchanged.")


if __name__ == "__main__":
    main()
