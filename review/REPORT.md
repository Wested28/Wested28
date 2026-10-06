# review.html: Job 3 (grouped order, grid mode, no-character outcome)

Ran in a Claude Code cloud session. Verified on synthetic data only (500 items, 15 characters, 25 sheets of 20 cells), driven by keyboard in headless Chromium from `file://`. The data schema is unchanged from v1.

## What changed

1. **Grouped by character (the default).** Items are ordered by proposed name, then by distance ascending. Clips without a usable proposal come last. The header shows:
   - the character
   - your position within it
   - how many of that character are unruled
   - how many characters with open items come after it (the no-proposal bucket is counted separately)

   `O` switches between grouped and original order; the setting is remembered.

2. **Grid mode (`G`).**
   - Shows up to 12 open proposals of the current character, with up to 8 of its references pinned above.
   - Every cell shows all three frames.
   - **Marking wrong:** `1`–`9`, `0`, `-`, `=` (or a click) mark a cell WRONG: red border, greyed frames, diagonal red hatching, a large "✕ WRONG".
   - **Enter** accepts the cells not marked wrong. The wrong ones go to one-at-a-time naming, with the name search already open ("from grid: marked wrong — name it (1 of 2)"). When those are done, you return to the grid on the same character.
   - **Nothing implicit:** `→`/`←` page and `Esc` leaves. Neither rules anything, and both were tested. Page marks reset when you page.
   - Once a character is finished, the next Enter moves the grid to the next character that still has open proposals.

3. **`N` = no character.** It is stored separately from `unknown`. The header counts four states: named, unknown, no character, unruled. No-character clips export as plain ids in their own panel and their own download; they are never written into `cells.txt`, and the test checks for that.

4. **Evidence label.** One line, always on screen in the guess panel: "MACHINE GUESS — evidence, not a verdict; unmeasured here". The grid footer repeats it. Agreement shows as "2/3 frames agree".

### Key change from Job 2

The name search now opens with `/`. `G` and `N` are letters, so letters can no longer start the search directly; otherwise a name beginning with G or N would trigger a command. `2` (reject) also opens the search straight away.

### Two fixes found while testing

- The header counted the no-proposal bucket as a character.
- Searching ranked an alias match above a direct name match: `purp` put HARE ("purple hare") first. Direct name matches now win ties.

## Key map

| mode | keys |
|---|---|
| one at a time | `Enter`/`1` accept · `2` reject → name · `3` unknown · `N` no character · `/` search names · `G` grid · `←` back · `→` skip · `.` next unruled · `Del` clear · `O` order · `=` export |
| search open | type · `↑` `↓` pick · `Enter` assign · `Esc` close |
| grid | `1`–`9` `0` `-` `=` toggle WRONG · `Enter` accept the rest · `→` `←` page · `Esc`/`G` leave |
| export | `1` rulings · `2` no-character ids · `3` restore · `D` download · `C` copy · `Esc` close |

## Verified

- **Unit tests: 25 pass** (`test_output.txt`), all run against the code inside `review.html`. That includes the range collapser, plus the new id-line import and grouped-order functions.
- **`drive_review.py`** checked the grouped order and the header numbers against an independent Python model. It then exercised the grid:
  - three cells marked
  - `Esc` and paging ruled nothing
  - Enter with 2 marked wrong ruled 10 and sent 2 to naming
  - the naming queue returned to the grid
  - a second page was accepted
- **Exports** (20 `cells.txt` lines and the id list) matched the model line for line.
- **Reload and restore:** the session survived a reload, and wiping storage then importing **both** downloads restored the session exactly.

**Timing at 2,400 items** (headless Chromium on Linux):

| action | time |
|---|---|
| load to first render | 0.15 s |
| one-at-a-time accept | 6.2 ms |
| no character (N) | 6.3 ms |
| open grid + accept 12 | 59 ms |
| grid toggle | 33 ms |
| export panel | 0.03 s |

Screenshots in `screenshots/`:
- `01` grouped view
- `02` grid with 3 marked wrong
- `03` wrong cells sent to naming
- `04` next grid page
- `05`–`07` the three export panels
- `08` after wipe + restore

## Not verified

- Edge/Chrome on Windows and real frames. All tests used synthetic data on Linux.
- How fast grid pages fill on your disk. The 36 frames per page load from local JPGs, but I measured only the time to draw the page, not to decode the images.
