"""
drive_namer.py -- open namer.html from file:// in headless Chromium, drive it
by keyboard, assert every outcome against the hidden truth file, and take
screenshots. Then (--perf) time it at ~2,400 items.

usage: python drive_namer.py <site_dir> <shots_dir>
       python drive_namer.py --perf <site_dir>
"""
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = next(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"))


def py_collapse(cells):
    s = sorted(set(cells)); runs, start = [], None
    for a, b in zip(s, s[1:] + [None]):
        start = a if start is None else start
        if b != a + 1:
            runs.append(f"{start:02d}" if start == a else f"{start:02d}-{a:02d}"); start = None
    return ",".join(runs)


def load_data(site):
    t = (site / "namer" / "namer_data.js").read_text()
    return json.loads(re.match(r"window\.NAMER_DATA = (.*);\s*$", t, re.S).group(1))


SKIP_NOTE = ""


def main(site, shots):
    global SKIP_NOTE
    site, shots = Path(site).resolve(), Path(shots).resolve()
    shots.mkdir(parents=True, exist_ok=True)
    data = load_data(site)
    truth = json.loads((site / "namer" / "namer_truth.json").read_text())
    items = data["items"]
    by_truth = {}
    for it in items:
        by_truth.setdefault(truth[it["id"]], []).append(it)
    errors = []
    with sync_playwright() as pw:
        br = pw.chromium.launch(executable_path=str(CHROME))
        pg = br.new_page(viewport={"width": 1600, "height": 1000})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        url = (site / "namer" / "namer.html").as_uri()
        pg.goto(url); pg.wait_for_selector("#app:not([hidden])")
        press = pg.keyboard.press
        ST = lambda: pg.evaluate("JSON.parse(localStorage.getItem('ag_namer_v1')||'{}')")
        cur_id = lambda: pg.evaluate("cur().id")

        def goto_item(cid):
            pg.evaluate(f"i=items.findIndex(x=>x.id==='{cid}'); typed=''; draw()")

        pg.wait_for_timeout(300)
        pg.screenshot(path=str(shots / "01_single_clip.png"))

        # --- 1. the M bug is gone: typing a name that starts with M names it ---
        first = cur_id()
        pg.keyboard.type("monkey prime"); press("Enter")
        assert ST()[first].get("name") == "MONKEY PRIME", ST()[first]
        assert "bucket" not in ST()[first]
        # backslash is now "not a character"
        second = cur_id(); press("\\")
        assert ST()[second].get("bucket") == "monster", ST()[second]

        # --- 2. earlier decisions inside an over-split character's groups ---
        A = "UNSEEN 00"
        a_groups = sorted({it["group"] for it in by_truth[A] if it["group"] is not None})
        assert len(a_groups) == 3, a_groups
        a_clips = [it for it in items if it["group"] in a_groups]
        prob, ejc = a_clips[0]["id"], a_clips[6]["id"]
        goto_item(prob); press("]")
        goto_item(ejc); press("Tab")
        assert ST()[prob]["bucket"] == "problem"
        assert pg.evaluate(f"ej['{ejc}']") is True

        # --- 3. singleton wall: select the 4 singletons that are one character ---
        press("/")
        pg.wait_for_selector("#wall.on")
        pg.wait_for_timeout(200)
        keys = pg.evaluate("wall.list.map(e=>e.key)")
        B = "UNSEEN 40"
        targets = [it["id"] for it in by_truth[B] if it["group"] is None]
        assert len(targets) == 4
        pos = 0
        for cid in sorted(targets, key=keys.index):
            idx = keys.index(cid)
            while pos < idx:
                press("ArrowRight"); pos += 1
            press(" ")
        assert pg.evaluate("wallSel().length") == 4
        before = len(ST())
        pg.keyboard.type("teal manta")
        pg.wait_for_timeout(250)
        pg.screenshot(path=str(shots / "02_singleton_wall_multiselect.png"))
        press("Enter")
        assert pg.locator("#confirm.on").count() == 1, "naming must ask first"
        assert len(ST()) == before, "nothing ruled before confirm"
        assert "Name 4 clip(s)" in pg.inner_text("#cq")
        pg.screenshot(path=str(shots / "03_wall_confirm.png"))
        press("Enter")
        st = ST()
        assert all(st[c]["name"] == "TEAL MANTA" for c in targets)
        assert pg.evaluate("wallSel().length") == 0

        # --- 4. near-miss warning on a new name, then cancel: nothing ruled ---
        press(" "); press("ArrowRight"); press(" ")
        n0 = len(ST())
        pg.keyboard.type("coral lynx"); press("Enter")
        assert "ROSE LYNX" in pg.inner_text("#cw"), pg.inner_text("#cw")
        pg.screenshot(path=str(shots / "04_near_miss_warning.png"))
        press("Escape")
        assert len(ST()) == n0, "cancel must rule nothing"
        press("Escape"); press("Escape")          # clear typing, then selection
        assert pg.evaluate("wallSel().length") == 0

        # --- 5. paging rules nothing ---
        n1 = len(ST())
        for k in ("PageDown", "PageDown", "PageUp", "ArrowDown", "ArrowUp"):
            press(k)
        assert len(ST()) == n1

        # --- 6. group wall: merge the 3 over-split groups of character A ---
        press("Tab")
        pg.wait_for_timeout(200)
        assert pg.inner_text("#wtitle") == "GROUP WALL"
        gkeys = pg.evaluate("wall.list.map(e=>e.key)")
        pos = pg.evaluate("wall.cur")
        for g in a_groups:
            idx = gkeys.index(f"g:{g}")
            while pos < idx: press("ArrowRight"); pos += 1
            while pos > idx: press("ArrowLeft"); pos -= 1
            press(" ")
        pg.wait_for_timeout(200)
        pg.screenshot(path=str(shots / "05_group_wall_three_selected.png"))
        press("+")
        assert "Merge 3 groups" in pg.inner_text("#cq")
        pg.screenshot(path=str(shots / "06_group_merge_confirm.png"))
        press("Enter")
        t = a_groups[0]
        assert pg.evaluate(f"mergedFrom({t}).length") == 3
        assert pg.evaluate(f"members({t}).length") == 15
        pg.wait_for_timeout(250)
        pg.screenshot(path=str(shots / "07_after_merge.png"))

        # --- 7. name the merged whole in one action; earlier decisions survive ---
        press("/")
        goto_item(a_clips[1]["id"])
        assert pg.evaluate("groupOf(cur())") == t
        st0 = ST()
        decided_before = {c["id"]: st0[c["id"]] for c in a_clips if c["id"] in st0}
        expect_skip = len(decided_before) + 1                   # + the ejected clip
        pg.keyboard.type("violet manta")
        press("Shift+Enter")
        st = ST()
        named = [c["id"] for c in a_clips if st.get(c["id"], {}).get("name") == "VIOLET MANTA"]
        assert len(named) == 15 - expect_skip, (len(named), expect_skip)
        for cid, r in decided_before.items():
            assert st[cid] == r, ("earlier decision overwritten", cid, r, st[cid])
        assert st[prob]["bucket"] == "problem" and ejc not in st
        assert f"skipped {expect_skip}" in pg.inner_text("#flash"), pg.inner_text("#flash")
        SKIP_NOTE = f"{len(named)} named, skipped {expect_skip} ({len(decided_before)} decided earlier, 1 ejected)"

        # --- 8. single-view merge (* then +) and undo (_) ---
        C = "UNSEEN 01"
        c_groups = sorted({it["group"] for it in by_truth[C] if it["group"] is not None})
        c0 = next(it for it in items if it["group"] == c_groups[0])
        c1 = next(it for it in items if it["group"] == c_groups[1])
        goto_item(c0["id"]); press("*")
        goto_item(c1["id"]); press("+"); press("Enter")
        assert pg.evaluate(f"mergedFrom({c_groups[0]}).length") == 2
        goto_item(c0["id"]); press("_"); press("Enter")
        assert pg.evaluate(f"mergedFrom({c_groups[0]}).length") == 1

        # --- 9. export panel matches an independent model of the state ---
        press("=")
        pg.wait_for_selector("#overlay.on")
        st = ST()
        groups = {}
        for cid, r in st.items():
            if r.get("name"):
                groups.setdefault((r["sheet"], r["name"]), []).append(int(r["cell"]))
        want = [f"{s}/{py_collapse(c)} = {nm}" for (s, nm), c in sorted(groups.items(), key=lambda kv: kv[0][0] + "\0" + kv[0][1])]
        got = [l for l in pg.input_value("#exrul").splitlines() if l]
        assert got == want, (got[:5], want[:5])
        new_names = pg.input_value("#exnew").splitlines()
        assert new_names == ["TEAL MANTA", "VIOLET MANTA"], new_names
        bk = pg.input_value("#exbuck")
        assert prob in bk and second in bk
        pg.screenshot(path=str(shots / "08_export_panels.png"))
        press("Escape")

        # --- 10. reload keeps merges; a stale merge is dropped, not re-pointed ---
        pg.reload(); pg.wait_for_selector("#app:not([hidden])")
        assert pg.evaluate(f"mergedFrom({t}).length") == 3
        mutated = json.loads(json.dumps(data))
        victim = next(it for it in mutated["items"] if it["group"] == a_groups[1])
        victim["group"] = None
        js = site / "namer" / "namer_data.js"
        original = js.read_text()
        js.write_text("window.NAMER_DATA = " + json.dumps(mutated) + ";\n")
        pg.reload(); pg.wait_for_selector("#app:not([hidden])")
        dropped = pg.evaluate("droppedMerges")
        js.write_text(original)
        assert dropped == 1, dropped
        assert pg.evaluate(f"mergedFrom({t}).length") == 2
        br.close()

    print("M-name typed and named (no longer filed as not-a-character); \\ files not-a-character")
    print("singleton wall: 4 cells selected across the wall, one name, confirm shown before anything ruled, 4 named")
    print("near-miss: CORAL LYNX warned of ROSE LYNX in the confirm; Esc ruled nothing")
    print("paging (PgDn/PgUp/arrows) ruled nothing")
    print("group wall: 3 over-split groups merged into one of 15 clips")
    print(f"Shift+Enter on the merged group: {SKIP_NOTE}; every earlier decision unchanged")
    print("single-view * then + merged two groups; _ undid it")
    print(f"export: {len(want)} cells.txt lines matched the Python model; new names {new_names}; buckets present")
    print("reload kept merges; changing one group's membership dropped exactly that merge (1), kept the other")
    print(f"page errors: {errors or 'none'}")
    print("DRIVE PASSED")


def perf(site):
    site = Path(site).resolve()
    with sync_playwright() as pw:
        br = pw.chromium.launch(executable_path=str(CHROME))
        pg = br.new_page(viewport={"width": 1600, "height": 1000})
        t = time.time(); pg.goto((site / "namer" / "namer.html").as_uri()); pg.wait_for_selector("#app:not([hidden])")
        n = pg.evaluate("items.length")
        print(f"items: {n}   load to first render: {time.time() - t:.2f}s")
        def timed(label, fn, reps):
            t = time.time()
            for _ in range(reps): fn()
            pg.evaluate("1")
            print(f"{label}: {(time.time() - t) / reps * 1000:.1f} ms each over {reps}")
        timed("type one letter", lambda: pg.keyboard.press("a"), 100)
        pg.keyboard.press("Escape")
        timed("name a clip (type 'ab' + Enter)", lambda: (pg.keyboard.type("ab"), pg.keyboard.press("Enter")), 50)
        timed("skip (ArrowRight)", lambda: pg.keyboard.press("ArrowRight"), 100)
        pg.keyboard.press("/"); pg.wait_for_selector("#wall.on")
        timed("wall: move (ArrowRight)", lambda: pg.keyboard.press("ArrowRight"), 100)
        timed("wall: select (Space)", lambda: pg.keyboard.press(" "), 100)
        timed("wall: page (PageDown)", lambda: pg.keyboard.press("PageDown"), 20)
        pg.keyboard.press("Escape")
        timed("wall: switch singles/groups (Tab)", lambda: pg.keyboard.press("Tab"), 10)
        br.close()


if __name__ == "__main__":
    if sys.argv[1] == "--perf":
        perf(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2])
