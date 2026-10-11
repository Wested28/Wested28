// test_namer.js -- unit tests for the pure functions INSIDE namer.html
// (extracted between the BEGIN PURE / END PURE markers, so the tested code is
// the shipped code). usage: node test_namer.js [path/to/namer.html]
"use strict";
const fs = require("fs"), path = require("path"), assert = require("assert");
const html = fs.readFileSync(process.argv[2] || path.join(__dirname, "namer.html"), "utf8");
const m = /\/\* ===== BEGIN PURE[^\n]*\n([\s\S]*?)\/\* ===== END PURE/.exec(html);
if (!m) throw new Error("PURE block not found in namer.html");
const P = new Function(m[1] + "\nreturn {collapse, lev, nearMissesIn, sig, resolveGroup, bulkSplit, buildExportFrom};")();

let n = 0;
const t = (name, fn) => { fn(); n++; console.log("  ok  " + name); };

console.log("collapse (same cases as review.html's collapser)");
t("[1,2,3,5,9,10] -> 01-03,05,09-10", () => assert.strictEqual(P.collapse([1, 2, 3, 5, 9, 10]), "01-03,05,09-10"));
t("[7] -> 07", () => assert.strictEqual(P.collapse([7]), "07"));
t("[1,2,3,4,7,14,18,19,20] -> 01-04,07,14,18-20", () => assert.strictEqual(P.collapse([1, 2, 3, 4, 7, 14, 18, 19, 20]), "01-04,07,14,18-20"));
t("numeric not lexicographic: [10,9,2,1] -> 01-02,09-10", () => assert.strictEqual(P.collapse([10, 9, 2, 1]), "01-02,09-10"));
t("duplicates: [5,5,6] -> 05-06", () => assert.strictEqual(P.collapse([5, 5, 6]), "05-06"));
t("2,000 random sets round-trip", () => {
  let seed = 9; const r = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let k = 0; k < 2000; k++) {
    const s = Array.from({ length: 1 + Math.floor(r() * 25) }, () => 1 + Math.floor(r() * 30));
    const want = [...new Set(s)].sort((a, b) => a - b), got = [];
    P.collapse(s).split(",").forEach(p => { const [a, b] = p.split("-").map(Number); for (let x = a; x <= (b || a); x++) got.push(x); });
    assert.deepStrictEqual(got, want);
  }
});

console.log("near-miss guard (the four real invented duplicates)");
const roster = ["ROSE LYNX", "IVORY LANCER", "ONYX MAID", "LA BRUINE", "BLUE SHARK"];
t("CORAL LYNX -> warns ROSE LYNX (shared word, 4 edits)", () => assert.deepStrictEqual(P.nearMissesIn("CORAL LYNX", roster), ["ROSE LYNX"]));
t("IVORY MECHA -> warns IVORY LANCER", () => assert.deepStrictEqual(P.nearMissesIn("IVORY MECHA", roster), ["IVORY LANCER"]));
t("DEATH MAID -> warns ONYX MAID", () => assert.deepStrictEqual(P.nearMissesIn("DEATH MAID", roster), ["ONYX MAID"]));
t("BEAVER -> no warning (no shared word, 8 edits: documented limit)", () => assert.deepStrictEqual(P.nearMissesIn("BEAVER", roster), []));
t("exact roster name is not its own near miss", () => assert.deepStrictEqual(P.nearMissesIn("ROSE LYNX", roster), []));

console.log("bulk actions skip earlier decisions");
t("skips named, bucketed and ejected; reports both lists", () => {
  const st = { a: { name: "X" }, b: { bucket: "problem" }, c: { bucket: "monster" } };
  const r = P.bulkSplit(["a", "b", "c", "d", "e"], st, { e: true });
  assert.deepStrictEqual(r.hit, ["d"]); assert.deepStrictEqual(r.skipped, ["a", "b", "c", "e"]);
});

console.log("merges");
t("sig is order-independent and changes when membership changes", () => {
  assert.strictEqual(P.sig(["a", "b", "c"]), P.sig(["c", "a", "b"]));
  assert.notStrictEqual(P.sig(["a", "b", "c"]), P.sig(["a", "b"]));
});
t("resolveGroup follows chains; unmerged groups resolve to themselves", () => {
  const into = { 3: 7, 7: 9 };
  assert.strictEqual(P.resolveGroup(3, into), 9); assert.strictEqual(P.resolveGroup(5, into), 5);
  assert.strictEqual(P.resolveGroup(null, into), null);
});

console.log("export: wall naming + buckets");
t("names from a wall action spanning sheets collapse per sheet; new names listed; buckets separate", () => {
  const st = {
    i1: { name: "TEAL MANTA", sheet: "U001", cell: 3 }, i2: { name: "TEAL MANTA", sheet: "U001", cell: 4 },
    i3: { name: "TEAL MANTA", sheet: "U001", cell: 5 }, i4: { name: "TEAL MANTA", sheet: "U002", cell: 1 },
    i5: { name: "ROSE LYNX", sheet: "U001", cell: 9 },
    i6: { bucket: "monster", sheet: "U001", cell: 10 }, i7: { bucket: "problem", sheet: "U002", cell: 2 },
    i8: { bucket: "group", sheet: "U002", cell: 3 },
  };
  const ex = P.buildExportFrom(st, new Set(roster));
  assert.deepStrictEqual(ex.lines, ["U001/09 = ROSE LYNX", "U001/03-05 = TEAL MANTA", "U002/01 = TEAL MANTA"]);
  assert.deepStrictEqual(ex.names, ["TEAL MANTA"]);
  assert.deepStrictEqual(ex.buckets, { monster: ["i6"], group: ["i8"], problem: ["i7"] });
  ex.lines.forEach(l => assert.ok(!/monster|problem|group/i.test(l)));
});
t("nothing ruled -> nothing exported", () => {
  const ex = P.buildExportFrom({}, new Set(roster));
  assert.deepStrictEqual(ex.lines, []); assert.deepStrictEqual(ex.names, []);
});
console.log(`\nALL ${n} TESTS PASSED`);
