"""
verify_probe_09.py -- sandbox checks for embed_probe_09.py (run in the cloud
container, NOT on the target machine). Does NOT use the real models.

  1. decode() vs imgutils' own _nms_postprocess (functions lifted from the
     dghs-imgutils 0.19.0 wheel source by AST, so no HF/hbutils imports).
  2. group_min() vs a brute-force double loop.
  3. p_at_1() / shuffled() vs a brute-force loop.
  4. pad_box() bounds.
  5. main() end to end on a fake _index with tiny synthetic ONNX models
     (detector [1,5,8400], feat [b,768], metrics [b,b]) on CPU.

usage: python -I verify_probe_09.py <path/to/imgutils/generic/yolo.py> <workdir>
"""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("probe", HERE / "embed_probe_09.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def load_ref(yolo_py):
    tree = ast.parse(Path(yolo_py).read_text())
    want = {"_yolo_xywh2xyxy", "_yolo_nms", "_xy_postprocess", "_nms_postprocess"}
    mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                           and n.name in want], type_ignores=[])
    ns = {"np": np, "List": list, "Tuple": tuple}
    exec(compile(mod, "imgutils_yolo_ref", "exec"), ns)
    return ns["_nms_postprocess"]


def test_decode(ref):
    rng = np.random.default_rng(1)
    n_cmp = 0
    for trial in range(300):
        N = 8400
        out = np.zeros((5, N), np.float32)
        out[0] = rng.uniform(0, 640, N); out[1] = rng.uniform(0, 640, N)
        out[2] = rng.uniform(2, 300, N); out[3] = rng.uniform(2, 400, N)
        out[4] = rng.uniform(0, 0.36, N) ** 1.5           # mostly below threshold
        k = rng.integers(0, 12)
        idx = rng.choice(N, k, replace=False)
        out[4, idx] = rng.uniform(0.3, 0.99, k)
        # duplicates near planted boxes so NMS has work to do
        dup = idx[: k // 2]
        if len(dup):
            j = rng.choice(N, len(dup), replace=False)
            out[:4, j] = out[:4, dup] + rng.normal(0, 4, (4, len(dup)))
            out[4, j] = out[4, dup] - 0.01
        old = (448, 448) if trial % 2 else (1280, 720)
        theirs = ref(out, 0.348, 0.5, old, (640, 640), ["person"])
        mine = probe.decode(out[None], 0.348, 0.5, old, (640, 640))
        assert len(theirs) == len(mine), (trial, len(theirs), len(mine))
        for (tb, _, ts), m in zip(theirs, mine):
            mb = tuple(int(np.round(v)) for v in m[:4])
            assert mb == tb, (trial, tb, mb)
            assert abs(ts - m[4]) < 1e-6
            n_cmp += 1
    assert probe.decode(np.zeros((1, 5, 10), np.float32), 0.3, 0.5, (448, 448), (640, 640)) == []
    print(f"1. decode == imgutils reference on 300 random outputs ({n_cmp} boxes compared)")


def test_group_min():
    rng = np.random.default_rng(2)
    for _ in range(50):
        C = rng.integers(2, 30)
        sizes = rng.integers(1, 5, C)
        M = int(sizes.sum()) + 5
        D = rng.random((M, M)).astype(np.float32)
        perm = rng.permutation(M)
        groups, p = [], 0
        for s in sizes:
            groups.append(list(perm[p:p + s])); p += s
        G = probe.group_min(D, groups)
        B = np.array([[D[np.ix_(gi, gj)].min() for gj in groups] for gi in groups])
        assert np.array_equal(G, B)
    print("2. group_min == brute force on 50 random cases (non-contiguous, shuffled groups)")


def test_p_at_1():
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = rng.integers(3, 60)
        D = rng.random((n, n))
        lab = np.array(list("ABCD"))[rng.integers(0, 4, n)]
        hits = probe.p_at_1(D, lab)
        brute = []
        for i in range(n):
            best = min((D[i, j], j) for j in range(n) if j != i)[1]
            brute.append(lab[best] == lab[i])
        assert np.array_equal(hits, np.array(brute))
    D = np.zeros((4, 4)); lab = np.array(["A", "A", "B", "B"])
    D[:] = 1; D[0, 1] = D[1, 0] = D[2, 3] = D[3, 2] = 0.1
    assert probe.p_at_1(D, lab).all()
    sh = probe.shuffled(np.random.default_rng(4).random((400, 400)),
                        np.array(list("ABCDEFGHIJ"))[np.arange(400) % 10])
    assert 0.03 < sh < 0.2, sh
    print(f"3. p_at_1 == brute force on 50 cases; shuffled on random D with 10 classes = "
          f"{sh:.3f} (chance 0.10)")


def test_pad():
    rng = np.random.default_rng(5)
    for _ in range(1000):
        x0, y0 = rng.uniform(0, 448, 2); w, h = rng.uniform(1, 448, 2)
        b = probe.pad_box((x0, y0, min(448, x0 + w), min(448, y0 + h)), 448, 448)
        assert 0 <= b[0] < b[2] <= 448 and 0 <= b[1] < b[3] <= 448, b
    assert probe.pad_box((100, 100, 200, 300), 448, 448) == (85, 70, 215, 330)
    print("4. pad_box stays inside the frame on 1000 random boxes; (100,100,200,300) -> "
          "(85,70,215,330) at PAD_FRAC 0.15")




def _noise(h, w, rng):
    return rng.integers(0, 255, (h, w, 3)).astype(np.uint8)


def test_letterbox():
    rng = np.random.default_rng(8)
    W = H = 448
    cases = []
    # 16:9 picture padded to square, white bars (what the real frames show)
    a = np.full((H, W, 3), 255, np.uint8); a[98:350] = _noise(252, W, rng)
    cases.append(("white letterbox 98/98", a, (0, 98, 448, 350)))
    a = np.zeros((H, W, 3), np.uint8); a[98:350] = _noise(252, W, rng)
    cases.append(("black letterbox", a, (0, 98, 448, 350)))
    a = np.zeros((H, W, 3), np.uint8); a[:, 98:350] = _noise(H, 252, rng)
    cases.append(("pillarbox", a, (98, 0, 350, 448)))
    cases.append(("no bars", _noise(H, W, rng), (0, 0, 448, 448)))
    # white-background character art: blank top, figure touches the bottom
    a = np.full((H, W, 3), 255, np.uint8); a[14:] = _noise(H - 14, W, rng)
    cases.append(("white bg, one-sided -> NOT cropped", a, (0, 0, 448, 448)))
    # bars that differ by more than BAR_SYM -> not a letterbox
    a = np.full((H, W, 3), 255, np.uint8); a[40:400] = _noise(360, W, rng)
    cases.append(("asymmetric 40/48 -> NOT cropped", a, (0, 0, 448, 448)))
    # JPEG round trip: bars are not perfectly flat after compression
    a = np.full((H, W, 3), 255, np.uint8); a[98:350] = _noise(252, W, rng)
    import io
    buf = io.BytesIO(); Image.fromarray(a).save(buf, "JPEG", quality=85)
    jim = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    jgot = got = probe.content_box(jim)
    assert abs(got[1] - 98) <= 2 and abs(got[3] - 350) <= 2 and got[0] == 0 and got[2] == 448, got
    for name, arr, want in cases:
        got = probe.content_box(Image.fromarray(arr))
        assert got == want, (name, got, want)
    # the real report's full-content box, padded inside the picture
    b = probe.pad_box((0, 98, 443, 349, 0.6), 448, 448, bounds=(0, 98, 448, 350))
    assert b == (0, 98, 448, 350), b
    b = probe.pad_box((138, 159, 257, 331, 0.8), 448, 448, bounds=(0, 98, 448, 350))
    assert 98 <= b[1] and b[3] <= 350, b
    print(f"6. content_box right on {len(cases)} synthetic frames + a JPEG round trip "
          f"(JPEG bars found at {jgot}); padded crops stay inside the picture")


def test_ccip_search(work):
    import os
    hub = Path(work) / "hfhome" / "hub" / "models--deepghs--ccip_onnx" / "snapshots" / "abc123" / "ccip-caformer_b36-24"
    hub.mkdir(parents=True, exist_ok=True)
    (hub / "model_feat.onnx").write_bytes(b"x" * 10)
    wrong = Path(work) / "_index" / "tools" / "stuff" / "ccip-caformer_b36-24"
    wrong.mkdir(parents=True, exist_ok=True)
    (wrong / "model_feat.onnx").write_bytes(b"x" * 9)
    old = os.environ.get("HF_HOME"); os.environ["HF_HOME"] = str(Path(work) / "hfhome")
    probe.INDEX = Path(work) / "_index"
    try:
        c = probe.ccip_candidates(["x"])
        real = sys.stdout
        import io
        sys.stdout = io.StringIO()
        try:
            got = probe.find_model("ccip feat", 10, c)
            out = sys.stdout.getvalue()
        finally:
            sys.stdout = real
        assert got == hub / "model_feat.onnx", got
        assert "WRONG SIZE" not in out  # right-size file found first is returned
        c2 = probe.ccip_candidates(["x", str(wrong)])
        assert c2[0] == wrong / "model_feat.onnx"
        sys.stdout = io.StringIO()
        try:
            got2 = probe.find_model("ccip feat", 10, c2)
            out2 = sys.stdout.getvalue()
        finally:
            sys.stdout = real
        assert got2 == hub / "model_feat.onnx" and "WRONG SIZE" in out2
    finally:
        if old is None: os.environ.pop("HF_HOME", None)
        else: os.environ["HF_HOME"] = old
    print("7. CCIP found in the HF cache snapshot layout; a wrong-size file is reported and skipped")


# ---- synthetic models ---------------------------------------------------
def make_det(path):
    """Constant YOLOv8-shaped output: two person boxes, plus 0*mean(input)
    so the graph really depends on the input."""
    out = np.zeros((1, 5, 8400), np.float32)
    out[0, :, 0] = [160, 320, 100, 300, 0.9]    # x 110..210, y 170..470 (640 space)
    out[0, :, 1] = [480, 320, 80, 200, 0.6]
    out[0, :, 2] = [482, 322, 80, 200, 0.55]    # NMS duplicate of #1
    c = numpy_helper.from_array(out, "C")
    zero = numpy_helper.from_array(np.zeros((1,), np.float32), "Z")
    nodes = [helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
             helper.make_node("Mul", ["m", "Z"], ["mz"]),
             helper.make_node("Add", ["C", "mz"], ["output0"])]
    g = helper.make_graph(nodes, "det",
                          [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
                          [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 8400])],
                          [c, zero])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    m.metadata_props.append(onnx.StringStringEntryProto(key="imgsz", value="[640, 640]"))
    onnx.save(m, path)


def make_feat(path):
    W = numpy_helper.from_array(np.random.default_rng(6).normal(size=(3, 768)).astype(np.float32), "W")
    nodes = [helper.make_node("GlobalAveragePool", ["input"], ["p"]),
             helper.make_node("Flatten", ["p"], ["f"]),
             helper.make_node("MatMul", ["f", "W"], ["output"])]
    g = helper.make_graph(nodes, "feat",
                          [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 384, 384])],
                          [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 768])], [W])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    onnx.save(m, path)


def make_metrics(path):
    """Distance = 1 - cosine."""
    one = numpy_helper.from_array(np.ones((1,), np.float32), "one")
    nodes = [helper.make_node("LpNormalization", ["input"], ["n"], axis=1, p=2),
             helper.make_node("Transpose", ["n"], ["nt"], perm=[1, 0]),
             helper.make_node("MatMul", ["n", "nt"], ["s"]),
             helper.make_node("Sub", ["one", "s"], ["output"])]
    g = helper.make_graph(nodes, "met",
                          [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 768])],
                          [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", "batch"])], [one])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    onnx.save(m, path)


def test_end_to_end(work):
    idx = Path(work) / "_index"
    (idx / "frames").mkdir(parents=True, exist_ok=True)
    mdir = idx / "models"
    (mdir / "anime_person_detection" / probe.DET_NAME).mkdir(parents=True, exist_ok=True)
    (mdir / "ccip-caformer_b36-24").mkdir(parents=True, exist_ok=True)
    dp = mdir / "anime_person_detection" / probe.DET_NAME / "model.onnx"
    fp = mdir / "ccip-caformer_b36-24" / "model_feat.onnx"
    mp = mdir / "ccip-caformer_b36-24" / "model_metrics.onnx"
    make_det(dp); make_feat(fp); make_metrics(mp)
    probe.DET_BYTES, probe.CCIP_FEAT_BYTES, probe.CCIP_METRICS_BYTES = (
        dp.stat().st_size, fp.stat().st_size, mp.stat().st_size)

    rng = np.random.default_rng(7)
    recs, expect = [], 0
    colours = {"LION": (220, 140, 30), "SHARK": (40, 90, 200), "PIG": (240, 150, 190)}
    for ch, col in colours.items():
        for i in range(12):
            cid = f"{ch[:3].lower()}{i:09d}"
            srcs = [{"ag_source": "W"}, {"ag_source": {"OLDNAME": "c"}},
                    {"ag_cluster": "A001/01"}]
            r = {"id": cid, "ag_characters": [ch], **srcs[i % 3]}
            recs.append(r)
            a = np.full((448, 448, 3), 128, np.uint8)
            a[120:330, 77:147] = np.clip(np.array(col) + rng.integers(-30, 30, 3), 0, 255)
            Image.fromarray(a).save(idx / "frames" / f"{cid}_2.jpg")
            expect += 1
    # rows that must be EXCLUDED
    recs += [
        {"id": "bad000000001", "ag_characters": ["LION"], "ag_source": 5},            # bad shape
        {"id": "bad000000002", "ag_characters": ["LION"], "ag_source": {"X": 1}},     # bad shape
        {"id": "two000000001", "ag_characters": ["LION", "PIG"], "ag_source": "W"},   # two names
        {"id": "nof000000001", "ag_characters": ["LION"], "ag_source": "W"},          # no frame
        {"id": "unr000000001", "ag_characters": ["LION"], "ag_source": "R"},          # not a ruling
        {"id": "few000000001", "ag_characters": ["OWL"], "ag_source": "C"},           # <10 clips
    ]
    for r in recs[-6:]:
        if r["id"] != "nof000000001":
            Image.new("RGB", (448, 448)).save(idx / "frames" / f"{r['id']}_2.jpg")
    with open(idx / "library.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    probe.INDEX = idx
    real_stdout = sys.stdout
    try:
        probe.main()
    finally:
        sys.stdout = real_stdout
    rep = (idx / "logs" / "embed_probe_09.md").read_text()
    assert f"labelled set: {expect} clips / 3 characters" in rep, rep[:2000]
    assert "2 records with an unknown ag_source shape" in rep
    assert "ZERO detections (fall back to full frame): 0/36" in rep
    assert "| cropN+full |" in rep and "Per character" in rep
    assert rep.startswith("# !!!!! CONTROL FAILED")   # synthetic data cannot hit 88.8
    # the box the fake detector plants maps to (77,119)-(147,329) in 448 space
    d = probe.decode(_det_out(dp), 0.348, 0.5, (448, 448), (640, 640))
    print(f"5. end to end OK on a fake _index: {expect} clips kept, 6 decoy rows excluded, "
          f"report {len(rep)} bytes. Decoded synthetic boxes (448 px): "
          f"{[tuple(round(v, 1) for v in b) for b in d]}")
    return rep


def _det_out(dp):
    import onnxruntime as ort
    s = ort.InferenceSession(str(dp), providers=["CPUExecutionProvider"])
    return s.run(None, {"images": np.zeros((1, 3, 640, 640), np.float32)})[0]


if __name__ == "__main__":
    ref = load_ref(sys.argv[1])
    test_decode(ref)
    test_group_min()
    test_p_at_1()
    test_pad()
    test_letterbox()
    test_ccip_search(sys.argv[2])
    rep = test_end_to_end(sys.argv[2])
    Path(sys.argv[2], "synthetic_report.md").write_text(rep)
    print("ALL CHECKS PASSED")
