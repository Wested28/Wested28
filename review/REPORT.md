# Job 2: review.html

Ran in a Claude Code cloud session. Job 1 still has one open item: the real-model decode never ran, because `huggingface.co` is blocked here.

## Files

| file | what |
|---|---|
| `review.html` | the tool: one file, vanilla JS, no network |
| `gen_review_data.py` | synthetic frames and `review_data.json`/`.js` (`python gen_review_data.py <out> 60 1`) |
| `test_review.js` | unit tests, run against the functions **extracted from review.html itself** (`node test_review.js`) |
| `test_output.txt` | the passing run: 21 tests |
| `drive_review.py` | opens the page from `file://` in headless Chromium, drives it with the keyboard only, and asserts the export |
| `screenshots/` | 7 screenshots of the real UI, plus the `.txt` file the page downloaded |

## Data: put `review_data.js` beside the page, not just the JSON

Chrome and Edge refuse `fetch()` of a sibling file under `file://`. That's why `browse.html` embeds its data. So the page loads data in this order:

1. `<script src="review_data.js">`, a file containing `window.REVIEW_DATA = {…};`
2. `fetch("review_data.json")`, which works over http and in some browsers
3. A file picker: press `O` and choose the JSON. This was tested.

Your generator should write the `.js` file. It is the same JSON, wrapped in one line.

## Schema v1

```json
{
  "version": 1,
  "roster":  ["SHARK", "LION", "..."],
  "aliases": {"ORANGE LION": "LION"},
  "items": [
    {"sheet": "N047", "cell": 12, "id": "a1b2c3d4e5f6",
     "frames": ["../frames/a1b2c3d4e5f6_1.jpg", "../frames/a1b2c3d4e5f6_2.jpg", "../frames/a1b2c3d4e5f6_3.jpg"],
     "proposed": "SHARK", "distance": 0.031, "agreement": 3}
  ],
  "refs": {"SHARK": ["../frames/x_2.jpg"]}
}
```

| field | required? | type | rule |
|---|---|---|---|
| `roster` | required | list of strings | the only names the page can output |
| `aliases` | optional | map, alias → roster name | lets you type "pink" and get PIG; the output is always the roster name |
| `sheet` | required | string | |
| `cell` | required | integer | `sheet/cell` must be unique, or the page refuses to load |
| `id` | required | string | |
| `frames` | required | list of up to 3 paths, relative to the HTML | |
| `proposed` | optional | string or `null` | if it isn't in the roster (after aliases), accept is disabled for that item and it gets a warning |
| `distance`, `agreement` | optional | number or `null` | display only |
| `refs` | optional | map, name → list of paths | the first 8 are shown |

## Key map (also shown on screen)

| key | does |
|---|---|
| `Enter` / `1` | accept the proposal, then move to the next item |
| `2` | reject: crosses out the proposal; then type the correct name |
| `3` | `unknown` |
| letters | filter roster names and aliases (prefix, then word-prefix, then substring) |
| `↑` `↓`, then `Enter` | pick a filtered name and assign it |
| `Esc` | clear the filter |
| `←` or `Backspace` | back one item |
| `→` | skip, recording nothing |
| `.` | jump to the next unruled item |
| `Del` | clear this item's ruling |
| `=` | export / restore panel; `D` download, `C` copy, `Esc` close |

Reject on its own writes nothing. A rejected item is only ruled once you name it or press `3`. That keeps "reject" from silently becoming `unknown`.

## Output

- The format is `SHEET/cells = NAME` or `SHEET/cells = unknown`.
- There is one line per sheet per value. Sheets are sorted, names are alphabetical, and `unknown` comes last within each sheet.
- Unruled and skipped items never emit a line.

**Tested:** `[1,2,3,5,9,10]` → `01-03,05,09-10` and `[7]` → `07` both pass. Also covered:
- numeric (not lexicographic) sort
- de-duplication
- cells ≥100
- rejecting non-integers
- 2,000 random sets that survive collapse → expand unchanged

In the browser run, the export panel and the downloaded file matched a separately written Python formatter line for line. That run covered 28 rulings: accept, unknown, a name typed via an alias, an arrow-picked name, an item that was skipped, an item ruled then cleared, and a proposal not in the roster.

## Persistence and recovery

Rulings live in `localStorage` under the key `charpassReview.v1`, keyed by `sheet/cell` **and the clip id**.

- **Rebuilt sheets:** sheet codes are positional, so after a rebuild a stored ruling whose id no longer matches is not applied or exported. The export panel shows a count of these, with a "Forget them" button.
- **Reload:** survives a reload. Tested.
- **Cleared storage:** the downloaded `.txt` is the backup. Paste or load it in the panel, and every line is ruled again. Tested: wipe storage, then import, gives an identical export.
- **Reminder:** the header shows how many rulings haven't been downloaded yet, and turns yellow at 50 or more.

## Not verified

- Edge/Chrome on Windows, and real `file://` paths with spaces (`D:\Video Library\…`). I tested headless Chromium on Linux only. Paths with spaces should work, because image paths are relative, but I haven't run it.
- Whether `localStorage` persists between `file://` pages in your browser profile. Chromium here persisted it across reloads.
- Real frames and real roster size. Scale was tested synthetically: 2,400 items load in 0.14 s, a ruling takes about 6 ms per keypress, and the export takes 0.02 s.
- The `C` copy key. Headless Chromium has no clipboard; it falls back to `execCommand`.
