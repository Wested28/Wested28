"""
drive_review.py -- open review.html from file:// in headless Chromium, drive
it with the KEYBOARD ONLY, and check what it exports against an independent
Python implementation of the cells.txt format. Takes screenshots.

usage: python drive_review.py <site_dir> <shots_dir>
  <site_dir> must already hold gen_review_data.py output, with review.html
  copied into <site_dir>/review/.
"""
import json
import shutil
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SITE, SHOTS = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
SHOTS.mkdir(parents=True, exist_ok=True)
URL = (SITE / "review" / "review.html").as_uri()
DATA = json.loads((SITE / "review" / "review_data.json").read_text())
ITEMS, ROSTER = DATA["items"], set(DATA["roster"])
CHROME = next(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"))


def py_collapse(cells):
    """Independent implementation, written separately from the JS one."""
    s = sorted(set(cells))
    runs, start = [], None
    for a, b in zip(s, s[1:] + [None]):
        start = a if start is None else start
        if b != a + 1:
            runs.append(f"{start:02d}" if start == a else f"{start:02d}-{a:02d}")
            start = None
    return ",".join(runs)


def py_lines(rul):
    groups = {}
    for (sheet, cell), v in rul.items():
        groups.setdefault((sheet, v), []).append(cell)
    keys = sorted(groups, key=lambda k: (k[0], k[1] == "unknown", k[1]))
    return [f"{s}/{py_collapse(groups[(s, v)])} = {v}" for s, v in keys]


assert py_collapse([1, 2, 3, 5, 9, 10]) == "01-03,05,09-10" and py_collapse([7]) == "07"

expected = {}          # (sheet, cell) -> value ; what the UI SHOULD hold
errors = []


def key(i):
    return ITEMS[i]["sheet"], ITEMS[i]["cell"]


def prop(i):
    p = ITEMS[i]["proposed"]
    return p if p in ROSTER else None


with sync_playwright() as pw:
    br = pw.chromium.launch(executable_path=str(CHROME))
    ctx = br.new_context(viewport={"width": 1600, "height": 1000}, accept_downloads=True)
    pg = ctx.new_page()
    pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL)
    pg.wait_for_selector("#app:not([hidden])")
    pg.wait_for_timeout(400)
    where = lambda: pg.inner_text("#where")
    status = lambda: pg.inner_text("#status")
    assert where().startswith(f"{ITEMS[0]['sheet']}/{ITEMS[0]['cell']}"), where()
    pg.screenshot(path=str(SHOTS / "01_main_view.png"))
    K = pg.keyboard.press

    def at(i):
        assert where().startswith(f"{ITEMS[i]['sheet']}/{ITEMS[i]['cell']} "), (i, where())

    i = 0
    # 0: accept with Enter (or 3 if no proposal)
    at(i)
    if prop(i): K("Enter"); expected[key(i)] = prop(i)
    else: K("3"); expected[key(i)] = "unknown"
    i += 1
    # 1: unknown
    at(i); K("3"); expected[key(i)] = "unknown"; i += 1
    # 2: reject, then name it via an ALIAS ("pink" -> PIG)
    at(i); K("2"); pg.keyboard.type("pink"); pg.wait_for_timeout(150)
    pg.screenshot(path=str(SHOTS / "02_reject_then_filter_alias.png"))
    assert "PIG" in pg.inner_text("#cands").splitlines()[0], pg.inner_text("#cands")
    assert pg.inner_text("#refname") == "PIG"
    K("Enter"); expected[key(i)] = "PIG"; i += 1
    # 3: type "l", arrow down to the 2nd candidate, assign it
    at(i); pg.keyboard.type("l"); pg.wait_for_timeout(150)
    cands = pg.inner_text("#cands").splitlines()
    pg.screenshot(path=str(SHOTS / "03_filter_candidates.png"))
    K("ArrowDown"); second = cands[1].split("  (")[0].strip()
    K("Enter"); expected[key(i)] = second; i += 1
    # 4: (missing 3rd frame) -- screenshot, then accept/unknown
    at(i); pg.wait_for_timeout(300)
    assert "missing:" in pg.inner_text("#frames")
    if prop(i): K("1"); expected[key(i)] = prop(i)
    else: K("3"); expected[key(i)] = "unknown"
    i += 1
    # 5: skip with ArrowRight -- must emit nothing
    at(i); K("ArrowRight"); i += 1
    # 6: Escape out of a half-typed filter, then accept
    at(i); pg.keyboard.type("wo"); K("Escape")
    assert pg.inner_text("#cands").strip() == ""
    if prop(i): K("Enter"); expected[key(i)] = prop(i)
    else: K("3"); expected[key(i)] = "unknown"
    i += 1
    # 7: proposal not in roster -> Enter must NOT rule or advance
    at(i); assert ITEMS[i]["proposed"] == "PHOENIX"
    K("Enter"); at(i); assert status() == "not yet ruled"
    K("3"); expected[key(i)] = "unknown"; i += 1
    # 8..29: accept everything that has a proposal, unknown otherwise
    while i < 30:
        at(i)
        if prop(i): K("Enter"); expected[key(i)] = prop(i)
        else: K("3"); expected[key(i)] = "unknown"
        i += 1
    # back three, clear one ruling with Delete
    for _ in range(3): K("ArrowLeft")
    at(27); K("Delete"); expected.pop(key(27)); assert status() == "not yet ruled"
    # back to the skipped item 5 -- still unruled
    for _ in range(22): K("ArrowLeft")
    at(5); assert status() == "not yet ruled"
    # "." jumps to the next unruled item: that is 27 (5 is current)
    K("."); at(27)
    pg.wait_for_timeout(300)
    pg.screenshot(path=str(SHOTS / "04_main_view_midsession.png"))

    # --- reload: session must survive ---
    before = pg.inner_text("#counts")
    pg.reload(); pg.wait_for_selector("#app:not([hidden])"); pg.wait_for_timeout(300)
    assert pg.inner_text("#counts") == before, (before, pg.inner_text("#counts"))
    at(27)

    # --- export panel ---
    K("=")
    pg.wait_for_selector("#overlay.on"); pg.wait_for_timeout(200)
    got = pg.input_value("#extext").strip().splitlines()
    want = py_lines(expected)
    pg.screenshot(path=str(SHOTS / "05_export_panel.png"))
    assert got == want, "\n".join(["GOT:"] + got + ["WANT:"] + want)
    with pg.expect_download() as dl:
        K("d")
    dl_path = SHOTS / "downloaded_rulings.txt"
    dl.value.save_as(str(dl_path))
    assert dl_path.read_text().strip().splitlines() == want
    exported = dl_path.read_text()
    K("Escape")

    # --- storage wiped -> restore from the downloaded file ---
    pg.evaluate("localStorage.clear()")
    pg.reload(); pg.wait_for_selector("#app:not([hidden])"); pg.wait_for_timeout(200)
    assert pg.inner_text("#counts").startswith("0 ruled"), pg.inner_text("#counts")
    K("=")
    pg.focus("#imptext"); pg.keyboard.insert_text(exported)
    pg.click("#impgo")
    pg.wait_for_timeout(200)
    msg = pg.inner_text("#impmsg")
    pg.screenshot(path=str(SHOTS / "06_restored_from_download.png"))
    assert pg.input_value("#extext").strip().splitlines() == want, msg
    assert msg.startswith(f"applied {len(expected)} cells"), msg
    K("Escape")

    # --- no review_data.js: loader + file picker fallback ---
    bare = SITE / "review_bare"
    bare.mkdir(exist_ok=True)
    shutil.copy(SITE / "review" / "review.html", bare / "review.html")
    pg2 = ctx.new_page()
    pg2.on("pageerror", lambda e: errors.append("bare: " + str(e)))
    pg2.goto((bare / "review.html").as_uri())
    pg2.wait_for_selector("#loader.on")
    pg2.screenshot(path=str(SHOTS / "07_loader_without_data_js.png"))
    pg2.set_input_files("#file", str(SITE / "review" / "review_data.json"))
    pg2.wait_for_selector("#app:not([hidden])")
    assert pg2.inner_text("#counts").split(" ruled")[0] == str(len(expected))  # shares storage

    br.close()

console_errs = [e for e in errors if "Failed to load resource" not in e]
print(f"ruled {len(expected)} items over {len(set(s for s, _ in expected))} sheets; "
      f"export matched the independent Python formatter line for line ({len(want)} lines)")
print("reload kept the session; wipe + import of the downloaded file restored it exactly")
print("loader fallback (no review_data.js) loaded the JSON via the file picker")
print(f"page/console errors: {console_errs or 'none'} "
      f"(+{len(errors) - len(console_errs)} expected 404s: the missing frame, missing review_data.js)")
print("\n".join(want))
print("DRIVE PASSED")
