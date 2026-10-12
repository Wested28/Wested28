"""
test_register_sheets.py -- adversarial tests for register_sheets.py.

The script is NOT modified. It is imported from a byte-identical copy
(under_test/, SHA-256 checked before and after) and its path constants
(INDEX, CIDX, LIB, FRAMES, DEFAULT_CSV, REPORT) are pointed at a throwaway
index tree per test. main() runs exactly as on the PC, with sys.argv set.

Each test states the CLAIM it attacks, then records HELD or BUG with what
the report said versus what is actually on disk.

usage: python test_register_sheets.py
"""
import contextlib
import csv
import datetime as _dt
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "under_test" / "register_sheets.py"
SHA = "460a6578582aac27cb09715c7cf9bc7e852b4f5da71de5833f98568a927514a1"
RESULTS = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("register_sheets_ut", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def hexid(n):
    return f"{n:012x}"


class World:
    """A fake _index: library.jsonl, charpass/cluster_index.json, frames/, a CSV."""

    def __init__(self, lib_ids, entries, csv_ids, frames_for=None, csv_bytes=None, index_text=None):
        self.root = Path(tempfile.mkdtemp(prefix="regsheets_"))
        (self.root / "charpass").mkdir()
        (self.root / "frames").mkdir()
        (self.root / "logs").mkdir()
        with open(self.root / "library.jsonl", "w", encoding="utf-8") as f:
            for i in lib_ids:
                f.write(json.dumps({"id": i}) + "\n")
        self.cidx = self.root / "charpass" / "cluster_index.json"
        if index_text is not None:
            self.cidx.write_text(index_text, encoding="utf-8")
        else:
            self.cidx.write_text(json.dumps(entries, indent=1), encoding="utf-8")
        for cid in (lib_ids if frames_for is None else frames_for):
            (self.root / "frames" / f"{cid}_2.jpg").write_bytes(b"\xff\xd8jpg")
        self.csv = self.root / "logs" / "unnamed_scan.csv"
        if csv_bytes is not None:
            self.csv.write_bytes(csv_bytes)
        else:
            with open(self.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["clip_id", "group"])
                for c in csv_ids:
                    w.writerow([c, ""])

    def run(self, *argv, patch=None):
        m = load_module()
        r = str(self.root)
        m.INDEX = r
        m.CP = os.path.join(r, "charpass")
        m.CIDX = str(self.cidx)
        m.LIB = os.path.join(r, "library.jsonl")
        m.FRAMES = os.path.join(r, "frames")
        m.DEFAULT_CSV = str(self.csv)
        m.REPORT = os.path.join(r, "logs", "register_sheets.md")
        if patch:
            patch(m)
        old_argv, old_out = sys.argv, sys.stdout
        sys.argv = ["register_sheets.py", *argv]
        rc, exc = None, None
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                sys.__stdout__ = io.StringIO()      # silence the Tee's console echo
                rc = m.main()
        except BaseException as e:               # noqa: BLE001
            exc = e
        finally:
            sys.stdout, sys.argv = old_out, old_argv
            sys.__stdout__ = old_out
        report = Path(m.REPORT).read_text(encoding="utf-8") if Path(m.REPORT).exists() else ""
        return rc, exc, report

    def index(self):
        return json.loads(self.cidx.read_text(encoding="utf-8"))

    def baks(self):
        return sorted(self.root.joinpath("charpass").glob("cluster_index.json.*.bak"))


def record(name, verdict, claim, observed, belief=""):
    RESULTS.append((name, verdict, claim, observed, belief))
    print(f"[{verdict}] {name}\n    claim   : {claim}\n    observed: {observed}"
          + (f"\n    you'd believe: {belief}" if belief else ""))


# ---- a standard world: 40 library clips, A001/A002 hold the first 10 ----------
LIB = [hexid(0xA1B2C3000000 + k * 0x1000001) for k in range(40)]
OLD = [{"cluster": 1, "sheet": "A001_red-unknown.jpg", "ids": [c[:8] for c in LIB[0:5]]},
       {"cluster": 2, "sheet": "A002_blue-x.jpg", "ids": [c[:8] for c in LIB[5:10]]}]
NEW = LIB[10:15]


def t_baseline():
    w = World(LIB, OLD, NEW)
    rc, exc, rep = w.run("--write")
    idx = w.index()
    ok = rc == 0 and exc is None and idx[:2] == OLD and len(idx) == 3 \
        and idx[2]["ids"] == sorted(c[:8] for c in NEW) and "All checks passed" in rep
    record("baseline: 5 unaddressed clips, --write", "HELD" if ok else "BUG",
           "appends one new sheet, old entries unchanged, verification passes",
           f"rc={rc}, entries {len(idx)}, new sheet {idx[-1]['sheet']}, 'All checks passed' in report: {'All checks passed' in rep}")


def t_csv_bom():
    rows = "clip_id,group\r\n" + "".join(f"{c},\r\n" for c in NEW)
    w = World(LIB, OLD, NEW, csv_bytes=b"\xef\xbb\xbf" + rows.encode())
    rc, exc, rep = w.run("--write")
    zero = "clips in source      : 0" in rep
    says_all = "The source's clips all have addresses" in rep
    record("CSV saved with a UTF-8 BOM (Excel / PowerShell -Encoding UTF8)",
           "BUG" if (zero and says_all and len(w.index()) == 2) else "HELD",
           "reads the clip ids from the CSV",
           f"report: 'clips in source : 0' = {zero}; ends 'The source's clips all have addresses.' = {says_all}; "
           f"nothing registered; rc={rc}",
           "all 5 clips already had addresses. They have none, and stay unrulable.")


def t_csv_header_variant():
    rows = "Clip_ID,group\n" + "".join(f"{c},\n" for c in NEW)
    w = World(LIB, OLD, NEW, csv_bytes=rows.encode())
    rc, exc, rep = w.run("--write")
    says_all = "The source's clips all have addresses" in rep
    record("CSV header spelled 'Clip_ID' (or 'clip_id ' with a trailing space)",
           "BUG" if says_all else "HELD", "reads the clip ids or says it could not",
           f"reads 0 rows, then reports 'The source's clips all have addresses.' = {says_all}",
           "same as the BOM case: everything is addressed.")


def t_duplicates():
    w = World(LIB, OLD, NEW + NEW[:3])
    rc, exc, rep = w.run("--write")
    ids = w.index()[-1]["ids"]
    ok = len(ids) == 5 and len(set(ids)) == 5
    record("duplicate clip ids in the CSV", "HELD" if ok else "BUG",
           "each clip registered once", f"new sheet holds {len(ids)} ids, {len(set(ids))} distinct")


def t_ambiguous_prefix():
    x, y = "deadbeef0001", "deadbeef0002"
    w = World(LIB + [x, y], OLD, [x])
    rc, exc, rep = w.run("--write")
    ok = len(w.index()) == 2 and "prefix resolves to 2 library ids" in rep
    record("8-char prefix matching two library ids (clip itself unregistered)", "HELD" if ok else "BUG",
           "refuses it", f"refused with 'prefix resolves to 2 library ids': {'prefix resolves to 2' in rep}; nothing written")


def t_ambiguous_prefix_already_registered():
    x, y = "deadbeef0001", "deadbeef0002"
    old = OLD + [{"cluster": 3, "sheet": "A003_x.jpg", "ids": ["deadbeef"]}]   # y was registered by prefix
    w = World(LIB + [x, y], old, [x])
    rc, exc, rep = w.run("--write")
    silently = "already addressed    : 1" in rep and "refused              : 0" in rep and len(w.index()) == 3
    record("8-char prefix shared by two library ids, ONE of them already registered",
           "BUG" if silently else "HELD",
           "refuses a clip whose prefix does not resolve to exactly one library id",
           "the other clip is counted as 'already addressed' (the prefix check runs before the ambiguity "
           "check) and is neither registered nor listed as refused",
           "that clip has an address. Its prefix resolves to two clips, so readers can't resolve it at all.")


def t_code_collision():
    old = OLD + [{"cluster": 9, "sheet": "U001_unsheeted.jpg", "ids": [LIB[20][:8]]},
                 {"cluster": 9, "sheet": "u002_unsheeted.jpg", "ids": [LIB[21][:8]]}]
    w = World(LIB, old, NEW)
    rc, exc, rep = w.run("--write", "--code", "u")
    codes = [e["sheet"].split("_")[0].upper() for e in w.index()]
    ok = len(codes) == len(set(codes)) and w.index()[-1]["sheet"].startswith("U003")
    record("--code colliding with existing U001 and lowercase u002", "HELD" if ok else "BUG",
           "never reuses a sheet code", f"new sheet {w.index()[-1]['sheet']}, codes unique: {len(codes) == len(set(codes))}")


def t_code_unparseable():
    w = World(LIB, OLD, NEW)
    rc, exc, rep = w.run("--write", "--code", "7")
    sheet = w.index()[-1]["sheet"] if len(w.index()) == 3 else None
    import re
    code = sheet.split("_")[0] if sheet else ""
    grammar = re.fullmatch(r"[A-Za-z]+\d+", code) is not None   # review.html / cells.txt sheet grammar
    record("--code 7 (a digit, not a letter)", "BUG" if sheet and not grammar else "HELD",
           "a new sheet gets a cells.txt address",
           f"wrote sheet code {code!r}; review.html's cells.txt parser needs [A-Za-z]+digits, so "
           f"'{code}/01 = X' cannot be imported: grammar ok = {grammar}",
           "the clips are now rulable. Their address can't be written in the format review.html imports.")


def t_per_extremes():
    w1 = World(LIB, OLD, NEW); rc1, _, _ = w1.run("--write", "--per", "1")
    w2 = World(LIB, OLD, NEW); rc2, _, _ = w2.run("--write", "--per", "999")
    ok = (len(w1.index()) == 7 and all(len(e["ids"]) == 1 for e in w1.index()[2:])
          and len(w2.index()) == 3 and len(w2.index()[2]["ids"]) == 5)
    record("--per 1 and --per 999", "HELD" if ok else "BUG",
           "chunks correctly, old entries untouched",
           f"--per 1 -> {len(w1.index()) - 2} sheets of 1; --per 999 -> 1 sheet of {len(w2.index()[2]['ids'])}")
    w3 = World(LIB, OLD, NEW); rc3, exc3, _ = w3.run("--write", "--per", "0")
    record("--per 0", "HELD" if (exc3 is not None and w3.index() == OLD and not w3.baks()) else "BUG",
           "nothing written on bad input",
           f"crashes with {type(exc3).__name__} before any backup or write; index unchanged")


def t_rerun():
    w = World(LIB, OLD, NEW)
    w.run("--write")
    first = w.cidx.read_bytes()
    rc, exc, rep = w.run("--write")
    ok = w.cidx.read_bytes() == first and "Nothing to register" in rep
    record("re-run with the same CSV", "HELD" if ok else "BUG",
           "a re-run is a no-op", f"index bytes unchanged: {w.cidx.read_bytes() == first}")


def t_crash_between_backup_and_write():
    w = World(LIB, OLD, NEW)
    before = w.cidx.read_bytes()

    def boom(m):
        def dump(*a, **k):
            raise OSError("disk full (simulated)")
        # a stand-in json object for THIS module only -- never mutate the shared json module
        m.json = type("J", (), {"load": staticmethod(json.load), "loads": staticmethod(json.loads),
                                "dump": staticmethod(dump)})
    rc, exc, rep = w.run("--write", patch=boom)
    tmp = Path(str(w.cidx) + ".tmp")
    ok = w.cidx.read_bytes() == before and len(w.baks()) == 1
    record("crash between backup and write (json.dump raises)", "HELD" if ok else "BUG",
           "index never left half-written",
           f"index unchanged: {w.cidx.read_bytes() == before}; backup present: {len(w.baks()) == 1}; "
           f"stray .tmp left behind: {tmp.exists()}")


def t_crash_after_write_before_verify():
    w = World(LIB, OLD, NEW)
    calls = {"n": 0}

    def flaky(m):
        real = json.load
        def load(fh, *a, **k):
            calls["n"] += 1
            if calls["n"] == 2:                          # the verification re-read
                raise OSError("antivirus lock (simulated)")
            return real(fh, *a, **k)
        m.json = type("J", (), {"load": staticmethod(load), "loads": staticmethod(json.loads),
                                "dump": staticmethod(json.dump)})
    rc, exc, rep = w.run("--write", patch=flaky)
    written = len(w.index()) == 3
    verified = "## verification" in rep and "All checks passed" in rep
    rc2, exc2, rep2 = w.run("--write")
    rerun_verifies = "verification" in rep2
    record("crash after the write, before verification; then re-run",
           "BUG" if (written and not verified and not rerun_verifies) else "HELD",
           "every write is proven by the verification block",
           f"file written: {written}; verification ran: {verified}; the re-run says "
           f"'{'Nothing to register' if 'Nothing to register' in rep2 else '?'}' and verifies nothing",
           "after re-running, everything is fine. The written entries were never checked by anything.")


def t_readonly():
    w = World(LIB, OLD, NEW)
    os.chmod(w.cidx, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    rc, exc, rep = w.run("--write")
    mode = stat.S_IMODE(os.stat(w.cidx).st_mode)
    written = len(w.index()) == 3
    record("read-only cluster_index.json (Linux semantics)",
           "BUG" if (written and mode & stat.S_IWUSR) else "HELD",
           "a read-only index is not overwritten",
           f"written: {written}; file mode afterwards {oct(mode)}"
           + (" -- the read-only flag is silently gone (os.replace swaps the directory entry)" if written and mode & stat.S_IWUSR else "")
           + ". NOT verified on Windows, where os.replace onto a read-only file normally fails with Access "
             "denied (a safe crash that leaves a .tmp).",
           "on Linux: that the read-only protection held." if written else "")


def t_full_ids_in_index():
    z = LIB[12]
    old = [OLD[0], {"cluster": 2, "sheet": "A002_blue-x.jpg", "ids": [c[:8] for c in LIB[5:10]] + [z]}]
    w = World(LIB, old, NEW)
    rc, exc, rep = w.run("--write")
    fail = "**A CHECK FAILED" in rep and len(w.index()) == 3
    record("an existing entry lists a FULL 12-char id (not an 8-char prefix) and the CSV holds that clip",
           "BUG" if fail else "HELD",
           "refuses a clip already registered anywhere",
           f"pre-check compares 8-char prefixes to the raw ids, so it misses it; the clip is written to a new sheet, "
           f"then the post-write check FAILS (rc={rc}) and asks you to restore the backup",
           "the safety check caught it, but only after the file was already rewritten.")


def t_backup_same_second():
    w = World(LIB, OLD, NEW[:2])

    class Frozen(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 12, 9, 0, 0)

    def freeze(m):
        m.datetime = type("D", (), {"datetime": Frozen})
    w.run("--write", patch=freeze)
    first_bak = w.baks()[0].read_bytes()
    with open(w.csv, "a", encoding="utf-8") as f:
        f.write(f"{NEW[2]},\n")
    w.run("--write", patch=freeze)
    baks = w.baks()
    lost = len(baks) == 1 and baks[0].read_bytes() != first_bak
    record("two --write runs inside the same second", "BUG" if lost else "HELD",
           "backs up before writing (each run keeps its own backup)",
           f"backup name has 1-second resolution: {len(baks)} .bak file(s); the first backup "
           f"(the pre-everything state) was overwritten: {lost}",
           "that a .bak of the original exists. It now holds the state after run 1.")


def t_byte_for_byte():
    text = json.dumps(OLD, indent=4, ensure_ascii=True)       # different formatting from the script's indent=1
    w = World(LIB, OLD, NEW, index_text=text)
    w.run("--write")
    after = w.cidx.read_text(encoding="utf-8")
    same_prefix = after.startswith(text.rstrip("]").rstrip())
    record("'every pre-existing entry is byte-for-byte unchanged'",
           "BUG" if not same_prefix else "HELD",
           "byte-for-byte",
           "the check compares parsed JSON (==), and the file is re-serialised with indent=1, so the old entries' "
           "bytes change whenever the original formatting differs. The data is equal; the wording is false.",
           "a byte diff of the old file would be empty. It isn't; every line moves.")


def main():
    assert sha(SCRIPT) == SHA, "script under test differs from the upload"
    for t in (t_baseline, t_csv_bom, t_csv_header_variant, t_duplicates, t_ambiguous_prefix,
              t_ambiguous_prefix_already_registered, t_code_collision, t_code_unparseable, t_per_extremes,
              t_rerun, t_crash_between_backup_and_write, t_crash_after_write_before_verify, t_readonly,
              t_full_ids_in_index, t_backup_same_second, t_byte_for_byte):
        try:
            t()
        except Exception as e:                           # a harness error is reported, never hidden
            record(t.__name__, "HARNESS-ERROR", "-", repr(e))
    assert sha(SCRIPT) == SHA, "script under test was modified during the run"
    bugs = [r for r in RESULTS if r[1] == "BUG"]
    print(f"\n{len(RESULTS)} tests: {len(bugs)} BUG, "
          f"{sum(r[1] == 'HELD' for r in RESULTS)} HELD, {sum(r[1] == 'HARNESS-ERROR' for r in RESULTS)} harness errors. "
          f"Script SHA-256 unchanged.")


if __name__ == "__main__":
    main()
