"""
drive_review.py -- open review.html from file:// in headless Chromium, drive
it with the KEYBOARD ONLY, and check every export against an independent
Python model of what should have been ruled. Takes screenshots.

usage: python drive_review.py <site_dir> <shots_dir>
  <site_dir> holds gen_review_data.py output with review.html copied into
  <site_dir>/review/.
     python drive_review.py --perf <site_dir>   timing only (use 2400 items)
"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = next(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"))


# ---- independent Python model of the export format and the grouped order ----
def py_collapse(cells):
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
        if v != "#none":
            groups.setdefault((sheet, v), []).append(cell)
    keys = sorted(groups, key=lambda k: (k[0], k[1] == "unknown", k[1]))
    return [f"{s}/{py_collapse(groups[(s, v)])} = {v}" for s, v in keys]


def py_order(items, roster):
    def prop(it):
        return it["proposed"] if it["proposed"] in roster else None
    names = sorted({prop(it) for it in items if prop(it)})
    order = []
    for n in names + [None]:
        idx = [i for i, it in enumerate(items) if prop(it) == n]
        idx.sort(key=lambda i: (items[i]["distance"] if isinstance(items[i]["distance"], (int, float)) else float("inf"), i))
        order += idx
    return order


assert py_collapse([1, 2, 3, 5, 9, 10]) == "01-03,05,09-10" and py_collapse([7]) == "07"


def perf(site):
    url = (Path(site) / "review" / "review.html").resolve().as_uri()
    with sync_playwright() as pw:
        br = pw.chromium.launch(executable_path=str(CHROME))
        pg = br.new_page(viewport={"width": 1600, "height": 1000})
        t = time.time(); pg.goto(url); pg.wait_for_selector("#app:not([hidden])")
        print(f"load to first render: {time.time() - t:.2f}s  ({pg.inner_text('#counts')})")
        def timed(label, keys, n):
            t = time.time()
            for _ in range(n):
                for k in keys: pg.keyboard.press(k)
            pg.inner_text("#counts")
            print(f"{label}: {(time.time() - t) / n * 1000:.1f} ms per action over {n}")
        timed("single accept (Enter)", ["Enter"], 200)
        timed("single no-character (N)", ["n"], 100)
        timed("grid open + accept 12 (G, Enter, Esc)", ["g", "Enter", "Escape"], 30)
        timed("grid toggle (1)", ["g", "1", "1", "Escape"], 30)
        t = time.time(); pg.keyboard.press("="); pg.wait_for_selector("#overlay.on")
        print(f"export panel: {time.time() - t:.2f}s, {len(pg.input_value('#extext').splitlines())} lines")
        br.close()


def main(site, shots):
    site, shots = Path(site).resolve(), Path(shots).resolve()
    shots.mkdir(parents=True, exist_ok=True)
    data = json.loads((site / "review" / "review_data.json").read_text())
    items, roster = data["items"], set(data["roster"])
    by_key = {f"{it['sheet']}/{it['cell']}": i for i, it in enumerate(items)}
    prop = lambda i: items[i]["proposed"] if items[i]["proposed"] in roster else None
    order = py_order(items, roster)
    expected = {}                       # (sheet, cell) -> value
    K = lambda i: (items[i]["sheet"], items[i]["cell"])
    errors = []

    with sync_playwright() as pw:
        br = pw.chromium.launch(executable_path=str(CHROME))
        ctx = br.new_context(viewport={"width": 1600, "height": 1000}, accept_downloads=True)
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto((site / "review" / "review.html").as_uri())
        pg.wait_for_selector("#app:not([hidden])")
        press = pg.keyboard.press

        def cur():                          # item index on screen (single mode)
            return by_key[pg.inner_text("#frames .lab").split()[-1]]

        def grid_items():
            return [by_key[t.split("·")[0].split()[-1]] for t in pg.locator("#ggrid .gc:not(.empty) .gh").all_inner_texts()]

        def ruled_count():
            return sum(int(x) for x in pg.locator("#counts b").all_inner_texts()[:3])

        # --- 1. grouped order and header ---
        assert cur() == order[0], (cur(), order[0])
        g0 = prop(order[0])
        gsize = sum(1 for i in order if prop(i) == g0)
        assert pg.inner_text("#wchar") == g0 and pg.inner_text("#wpos") == f"1 / {gsize}"
        ngroups = len({prop(i) for i in order if prop(i)})
        assert f"{ngroups - 1} more characters after this" in pg.inner_text("#wrest"), pg.inner_text("#wrest")
        pg.wait_for_timeout(300)
        pg.screenshot(path=str(shots / "01_grouped_single.png"))

        # --- 2. single-mode actions ---
        i = cur(); press("Enter"); expected[K(i)] = prop(i)
        i = cur(); press("n"); expected[K(i)] = "#none"
        i = cur(); press("3"); expected[K(i)] = "unknown"
        i = cur(); press("/"); pg.keyboard.type("purp"); press("Enter"); expected[K(i)] = "PURPLE HORN"
        i = cur(); press("2"); assert pg.locator("#filterbox.active").count() == 1
        pg.keyboard.type("frog"); press("Enter"); expected[K(i)] = "GREEN FROG"
        pos_before = pg.inner_text("#wpos")
        assert pos_before == f"6 / {gsize}", pos_before
        assert f"{gsize - 5} unruled left in {g0}" in pg.inner_text("#wrest"), pg.inner_text("#wrest")
        assert ruled_count() == 5

        # --- 3. grid: toggles, Esc rules nothing, paging rules nothing ---
        big = max({prop(i) for i in order if prop(i)}, key=lambda n: sum(prop(i) == n for i in order))
        # jump to the biggest group with "." navigation: switch order is not needed, use G on an item of it
        while prop(cur()) != big:
            press("ArrowRight")
        press("g")
        pg.wait_for_selector("#gridview.on")
        assert pg.inner_text("#gtitle b") == big
        shown = grid_items()
        assert len(shown) == 12 and all(prop(i) == big for i in shown)
        for k in ["2", "5", "="]:
            press(k)
        assert pg.locator("#ggrid .gc.wrong").count() == 3
        pg.wait_for_timeout(300)
        pg.screenshot(path=str(shots / "02_grid_three_marked_wrong.png"))
        press("Escape")
        assert ruled_count() == 5, "Esc from grid must rule nothing"
        press("g"); press("ArrowRight")
        assert grid_items() != shown and ruled_count() == 5
        press("ArrowLeft"); assert grid_items() == shown and pg.locator("#ggrid .gc.wrong").count() == 0
        # --- 4. grid accept: 2 wrong -> named one at a time, rest accepted ---
        press("1"); press("4")
        wrong = [shown[0], shown[3]]
        press("Enter")
        for j in shown:
            if j not in wrong:
                expected[K(j)] = big
        assert pg.locator("#gridview.on").count() == 0
        assert cur() == wrong[0] and pg.locator("#prop.rejected").count() == 1
        assert "1 of 2" in pg.inner_text("#queue"), pg.inner_text("#queue")
        pg.wait_for_timeout(500)
        pg.screenshot(path=str(shots / "03_grid_wrong_sent_to_naming.png"))
        pg.keyboard.type("cow"); press("Enter"); expected[K(wrong[0])] = "COW"
        assert cur() == wrong[1] and "2 of 2" in pg.inner_text("#queue")
        press("Escape"); press("n"); expected[K(wrong[1])] = "#none"
        # queue drained -> back in the grid on the same character
        pg.wait_for_selector("#gridview.on")
        assert pg.inner_text("#gtitle b") == big
        page2 = grid_items()
        assert not set(page2) & set(shown)
        press("Enter")
        for j in page2:
            expected[K(j)] = big
        pg.wait_for_timeout(250)
        pg.screenshot(path=str(shots / "04_grid_next_page.png"))
        press("Escape")
        assert ruled_count() == len(expected), (ruled_count(), len(expected))

        # --- 5. original order toggle and back ---
        press("o"); assert "/" in pg.inner_text("#wchar") and "original order" in pg.inner_text("#wrest")
        press("o"); assert pg.inner_text("#wrest").find("after this") > 0

        # --- 6. reload keeps everything ---
        before = pg.inner_text("#counts")
        pg.reload(); pg.wait_for_selector("#app:not([hidden])")
        assert pg.inner_text("#counts") == before

        # --- 7. the three export panels ---
        want_lines = py_lines(expected)
        want_ids = sorted(items[by_key[f"{s}/{c}"]]["id"] for (s, c), v in expected.items() if v == "#none")
        press("=")
        pg.wait_for_selector("#overlay.on")
        got_lines = pg.input_value("#extext").strip().splitlines()
        if got_lines != want_lines:
            print("ONLY IN PAGE:", sorted(set(got_lines) - set(want_lines)))
            print("ONLY IN MODEL:", sorted(set(want_lines) - set(got_lines)))
        assert got_lines == want_lines
        pg.wait_for_timeout(200)
        pg.screenshot(path=str(shots / "05_export_rulings.png"))
        with pg.expect_download() as dl1:
            press("d")
        dl1.value.save_as(str(shots / "downloaded_rulings.txt"))
        press("2")
        assert pg.input_value("#nonetext").strip().splitlines() == want_ids
        pg.wait_for_timeout(500)
        pg.screenshot(path=str(shots / "06_export_no_character_ids.png"))
        with pg.expect_download() as dl2:
            press("d")
        dl2.value.save_as(str(shots / "downloaded_no_character_ids.txt"))
        press("3")
        pg.wait_for_timeout(500)
        pg.screenshot(path=str(shots / "07_export_restore.png"))
        press("Escape")
        assert (shots / "downloaded_rulings.txt").read_text().strip().splitlines() == want_lines
        assert (shots / "downloaded_no_character_ids.txt").read_text().strip().splitlines() == want_ids
        for line in want_lines:
            assert "#none" not in line

        # --- 8. wipe storage, restore from BOTH downloads ---
        pg.evaluate("localStorage.clear()")
        pg.reload(); pg.wait_for_selector("#app:not([hidden])")
        assert ruled_count() == 0
        press("="); press("3")
        pg.focus("#imptext")
        pg.keyboard.insert_text((shots / "downloaded_rulings.txt").read_text()
                                + (shots / "downloaded_no_character_ids.txt").read_text())
        pg.click("#impgo")
        msg = pg.inner_text("#impmsg")
        press("Escape"); press("=")
        assert pg.input_value("#extext").strip().splitlines() == want_lines, msg
        press("2")
        assert pg.input_value("#nonetext").strip().splitlines() == want_ids, msg
        pg.screenshot(path=str(shots / "08_restored_import.png"))
        br.close()

    named = sum(1 for v in expected.values() if v not in ("unknown", "#none"))
    print(f"grouped order matched the Python model; header counts and positions correct")
    print(f"grid: 3 toggles shown, Esc ruled nothing, paging ruled nothing, Enter ruled 10 + sent 2 to naming,")
    print(f"      queue drained back into the grid, second page accepted with Enter")
    print(f"ruled {len(expected)}: {named} named, "
          f"{sum(v == 'unknown' for v in expected.values())} unknown, {len(want_ids)} no-character")
    print(f"exports matched the Python model: {len(want_lines)} cells.txt lines, {len(want_ids)} ids; "
          f"no '#none' leaked into cells.txt")
    print("reload kept the session; wipe + import of both downloads restored it exactly")
    print(f"page errors: {errors or 'none'}")
    print("DRIVE PASSED")


if __name__ == "__main__":
    if sys.argv[1] == "--perf":
        perf(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2])
