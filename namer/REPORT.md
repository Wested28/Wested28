# namer.html: Job 3 (singleton wall, group merge)

Built on your copy as uploaded. The first commit is your file unchanged, so the diff shows exactly what changed. Data format is unchanged: `namer_data.js` → `window.NAMER_DATA = {roster, items}`.

## Three problems found in the current namer, fixed

1. **`M` broke names starting with M.** "Not a character" fired whenever nothing was typed yet, so the first keystroke of MONKEY PRIME / MAGENTA PILOT / MIDNIGHT JAZZ filed the clip as NOT A CHARACTER and moved on. It's the `G`/`X` problem your own comment warns about. **Not-a-character is now `\`**, outside the name charset like the brackets. `M` now just types.
2. **`Shift+Enter`, `{` and `}` overwrote bucketed clips.** They skipped clips with a name, but not clips already filed as not-a-character, group or problematic. Every bulk action now goes through one rule (`bulkSplit`): anything already decided, or ejected, is skipped, and the flash says `× N · skipped M`.
3. **The near-miss guard can't catch BEAVER / LA BRUINE.** No shared word, 8 edits apart. It does catch the other three pairs; the tests pin all four results. No text rule catches that one, so it still needs your eyes.

## New

- **Singleton wall (`/`).**
  - Shows every open singleton as a grid, each cell with all three frames; `,` `.` change the number of columns.
  - **Selecting:** `Space` selects any number of cells across pages. Each selected cell gets a thick blue border, a blue tint and a numbered ✓ badge; the cursor is a separate yellow dashed outline.
  - **Naming:** type one name, then `Enter`. A confirm box states the clip count, any skips, and the near-miss warning. `Enter` again rules; `Esc` cancels with nothing ruled.
  - **Other keys:** `\` `[` `]` file the selection into a bucket, also behind a confirm. `` ` `` reuses the last name, also behind a confirm.
  - Paging and leaving rule nothing.
- **Merging groups.**
  - **From the clip view:** `*` marks a group, go to another, `+` merges it in (with a confirm), and `_` undoes merges into the current group.
  - **Group wall:** `Tab` in the wall switches to the group wall, where you select several groups and `+` merges them into the first one.
  - **Naming the merged whole:** one `Shift+Enter`.
  - **Saving:** merges are stored against each group's exact clip membership. If a rebuilt `namer_data.js` changes a group's clips, that merge is dropped and flagged, never re-pointed.

## Keys added or changed

`\` not a character (was `M`) · `/` walls · `*` mark merge target · `+` merge · `_` undo merges. In the wall: `Space`, arrows, `PgDn`/`PgUp`, `Tab`, `,` `.`, `Esc` (first clears typing, then selection). Every one of these is outside the name charset.

## Verified (headless Chromium, `file://`)

**Synthetic data:** 40 groups / 200 clips, 160 singletons, 110 roster names (`gen_namer_data.py`). Eight characters are deliberately split across 2–3 groups. Some singletons share a character, and 40 contain no character.

**`drive_namer.py` passed:**
- An M-name gets typed and named.
- 4 singletons selected on the wall and named in one action, with nothing ruled before the confirm.
- CORAL LYNX warned of ROSE LYNX, and `Esc` ruled nothing.
- Paging ruled nothing.
- 3 over-split groups merged into one of 15 clips.
- `Shift+Enter` on the merged group named 12 and skipped 3 (2 decided earlier, 1 ejected), leaving every earlier decision unchanged.
- `*`/`+` merge and `_` undo work in the clip view.
- The export matched an independent Python model.
- A reload kept the merges, and a changed group dropped exactly its own merge.

**Unit tests:** 16 for the namer (`test_output.txt`), including the shared collapser cases and the new export path. The existing 25 for `review.html` still pass.

**At 2,403 items:**

| action | time |
|---|---|
| load to first render | 0.20 s |
| type a letter | 7.8 ms |
| name a clip | 27 ms |
| skip | 9.4 ms |
| wall move | 23 ms |
| wall select | 13 ms |
| wall page | 40 ms |
| switch singletons ↔ groups | 46 ms |

Screenshots in `screenshots/`:

| file | shows |
|---|---|
| `02` | multi-select in progress |
| `03` | confirm |
| `04` | near-miss warning |
| `05`–`07` | group merge |
| `08` | export panels |

## Not verified / left alone

- Windows browsers and your real frames.
- **`review.html` untouched**, as asked. The copy you sent is an older layout than the repo's Job 3 version, so the repo's version is not what you run.
- `Shift+Enter` with nothing typed still names the group with the last-used name, without a confirm, as before. Say if you want that behind a confirm too.
