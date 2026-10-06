// test_review.js -- unit tests for the pure functions INSIDE review.html.
// Extracts the code between the BEGIN PURE / END PURE markers from the
// shipped file, so the tested code is the code that runs in the browser.
// usage: node test_review.js [path/to/review.html]
"use strict";
const fs = require("fs");
const path = require("path");
const assert = require("assert");

const html = fs.readFileSync(process.argv[2] || path.join(__dirname, "review.html"), "utf8");
const m = /\/\/ ===== BEGIN PURE[^\n]*\n([\s\S]*?)\/\/ ===== END PURE/.exec(html);
if (!m) throw new Error("PURE block not found in review.html");
const P = new Function(m[1] + "\nreturn {collapseCells, expandCells, buildLines, parseLines, splitIdLines, groupOrder};")();

let n = 0;
function t(name, fn) { fn(); n++; console.log("  ok  " + name); }

console.log("collapseCells");
t("[1,2,3,5,9,10] -> 01-03,05,09-10", () => assert.strictEqual(P.collapseCells([1, 2, 3, 5, 9, 10]), "01-03,05,09-10"));
t("[7] -> 07 (single stays single)", () => assert.strictEqual(P.collapseCells([7]), "07"));
t("[1,2,3,4,7,14,18,19,20] -> 01-04,07,14,18-20", () => assert.strictEqual(P.collapseCells([1, 2, 3, 4, 7, 14, 18, 19, 20]), "01-04,07,14,18-20"));
t("pair collapses: [9,10] -> 09-10", () => assert.strictEqual(P.collapseCells([9, 10]), "09-10"));
t("unsorted input sorted NUMERICALLY: [10,9,2,1] -> 01-02,09-10", () => assert.strictEqual(P.collapseCells([10, 9, 2, 1]), "01-02,09-10"));
t("lexicographic trap: [2,10,11,3] -> 02-03,10-11", () => assert.strictEqual(P.collapseCells([2, 10, 11, 3]), "02-03,10-11"));
t("duplicates removed: [5,5,6,6] -> 05-06", () => assert.strictEqual(P.collapseCells([5, 5, 6, 6]), "05-06"));
t("no runs: [1,3,5] -> 01,03,05", () => assert.strictEqual(P.collapseCells([1, 3, 5]), "01,03,05"));
t("three digits keep their width: [99,100,101] -> 99-101", () => assert.strictEqual(P.collapseCells([99, 100, 101]), "99-101"));
t("empty -> ''", () => assert.strictEqual(P.collapseCells([]), ""));
t("rejects '3' (string)", () => assert.throws(() => P.collapseCells(["3"])));
t("rejects 2.5", () => assert.throws(() => P.collapseCells([2.5])));
t("rejects -1", () => assert.throws(() => P.collapseCells([-1])));
t("rejects NaN", () => assert.throws(() => P.collapseCells([NaN])));

console.log("expandCells");
t("'01-03,05,09-10' -> [1,2,3,5,9,10]", () => assert.deepStrictEqual(P.expandCells("01-03,05,09-10"), [1, 2, 3, 5, 9, 10]));
t("rejects '05-03'", () => assert.throws(() => P.expandCells("05-03")));
t("rejects '0x'", () => assert.throws(() => P.expandCells("0x")));

console.log("round trip: 2000 random sets, expand(collapse(s)) == sorted unique s");
t("random round trip", () => {
  let seed = 12345;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let k = 0; k < 2000; k++) {
    const s = Array.from({ length: 1 + Math.floor(rnd() * 30) }, () => 1 + Math.floor(rnd() * 40));
    const want = [...new Set(s)].sort((a, b) => a - b);
    const out = P.collapseCells(s);
    assert.deepStrictEqual(P.expandCells(out), want, "set " + s);
    assert.ok(/^\d{2,}(-\d{2,})?(,\d{2,}(-\d{2,})?)*$/.test(out), out);
    // no run of >=2 consecutive numbers left uncollapsed
    out.split(",").forEach((p, i, a) => {
      if (i) { const prev = +a[i - 1].split("-").pop(), cur = +p.split("-")[0]; assert.ok(cur > prev + 1, out); }
    });
  }
});

console.log("buildLines");
t("one line per sheet per name, unknown last, sheets natural-ordered", () => {
  const lines = P.buildLines([
    { sheet: "N047", cell: 3, value: "SHARK" }, { sheet: "N047", cell: 1, value: "SHARK" },
    { sheet: "N047", cell: 2, value: "SHARK" }, { sheet: "N047", cell: 7, value: "unknown" },
    { sheet: "N047", cell: 5, value: "LION" }, { sheet: "N009", cell: 12, value: "PIG" },
    { sheet: "N047", cell: 9, value: "SHARK" }, { sheet: "N047", cell: 10, value: "SHARK" },
  ]);
  assert.deepStrictEqual(lines, [
    "N009/12 = PIG",
    "N047/05 = LION",
    "N047/01-03,09-10 = SHARK",
    "N047/07 = unknown",
  ]);
});
t("nothing ruled -> no lines", () => assert.deepStrictEqual(P.buildLines([]), []));

console.log("parseLines");
t("parses own output back, ignores comments/blank, reports junk", () => {
  const p = P.parseLines("# header\nN047/01-03,09-10 = SHARK\n\nN047/07 = unknown\nA053/05 = PINK PIG  # note\ngarbage\nN1/05-03 = X\n");
  assert.strictEqual(p.ok.length, 7);
  assert.deepStrictEqual(p.ok[6], { sheet: "A053", cell: 5, value: "PINK PIG" });
  assert.strictEqual(p.errors.length, 2);
});


console.log("splitIdLines (no-character import)");
t("ids split from cells lines; comments ok; case folded", () => {
  const r = P.splitIdLines("N047/01-03 = SHARK\na1b2c3d4e5f6\nA1B2C3D4E5F7  # bg\nN047/07 = unknown\nnot-an-id\n");
  assert.deepStrictEqual(r.ids, ["a1b2c3d4e5f6", "a1b2c3d4e5f7"]);
  const p = P.parseLines(r.rest);
  assert.strictEqual(p.ok.length, 4); assert.strictEqual(p.errors.length, 1);
});
t("a cells line is never mistaken for an id", () => assert.deepStrictEqual(P.splitIdLines("N047/12 = PIG").ids, []));

console.log("groupOrder (grouped review order)");
t("by name, distance ascending, null distance last, no-proposal group last, ties stable", () => {
  const rows = [
    { prop: "SHARK", distance: 0.20 }, { prop: null, distance: null }, { prop: "LION", distance: 0.05 },
    { prop: "SHARK", distance: 0.01 }, { prop: "SHARK", distance: null }, { prop: "LION", distance: 0.05 },
    { prop: "BEAR", distance: 0.3 },
  ];
  const g = P.groupOrder(rows);
  assert.deepStrictEqual(g.groups.map(x => x.name), ["BEAR", "LION", "SHARK", null]);
  assert.deepStrictEqual(g.order, [6, 2, 5, 3, 0, 4, 1]);
});
t("every index appears exactly once (500 random rows)", () => {
  let seed = 7; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  const names = ["A", "B", "C", null];
  const rows = Array.from({ length: 500 }, () => ({ prop: names[Math.floor(rnd() * 4)], distance: rnd() < 0.1 ? null : rnd() }));
  const g = P.groupOrder(rows);
  assert.deepStrictEqual([...g.order].sort((a, b) => a - b), rows.map((_, i) => i));
  g.groups.forEach(gr => { for (let i = 1; i < gr.idx.length; i++) {
    const a = rows[gr.idx[i - 1]].distance ?? Infinity, b = rows[gr.idx[i]].distance ?? Infinity; assert.ok(a <= b); } });
});

console.log(`\nALL ${n} TESTS PASSED`);
