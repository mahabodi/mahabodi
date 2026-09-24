"""Check that the Triton configs in deploy/models/triton match the real exported ONNX models:
same input/output names, dtypes, and ranks (Triton dims + the implicit batch dimension).

    python deploy/models/check_model_configs.py        # needs onnxruntime and models/{laya-v2,minilm}
"""
import os, re, sys
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAIRS = {"laya": "models/laya-v2/model.onnx", "minilm": "models/minilm/model.onnx"}
ONNX2TRITON = {"tensor(int64)": "TYPE_INT64", "tensor(float)": "TYPE_FP32", "tensor(bool)": "TYPE_BOOL"}


def parse(path):
    txt = open(path).read()
    ents = {}
    for kind in ("input", "output"):
        block = re.search(kind + r"\s*\[(.*?)\n\]", txt, re.S).group(1)
        for m in re.finditer(r'name:\s*"(\w+)"\s+data_type:\s*(\w+)\s+dims:\s*\[([^\]]*)\](\s*reshape:\s*\{\s*shape:\s*\[([^\]]*)\])?', block):
            dims = [d.strip() for d in m.group(3).split(",") if d.strip()]
            reshape = m.group(5)
            rank = 1 + (len([d for d in reshape.split(",") if d.strip()]) if reshape is not None else len(dims))
            ents[(kind, m.group(1))] = (m.group(2), rank)
    return ents


def main():
    bad = 0
    for name, onnx in PAIRS.items():
        cfg = parse(os.path.join(ROOT, "deploy", "models", "triton", name, "config.pbtxt"))
        s = ort.InferenceSession(os.path.join(ROOT, onnx), providers=["CPUExecutionProvider"])
        real = {("input", i.name): (ONNX2TRITON[i.type], len(i.shape)) for i in s.get_inputs()}
        real.update({("output", o.name): (ONNX2TRITON[o.type], len(o.shape)) for o in s.get_outputs()})
        for key in sorted(set(real) | set(cfg)):
            ok = real.get(key) == cfg.get(key)
            bad += not ok
            print("%-4s %-7s %-6s %-17s onnx=%s config=%s" % ("ok" if ok else "BAD", name, key[0], key[1], real.get(key), cfg.get(key)))
    print("RESULT:", "PASS" if not bad else "FAIL (%d mismatches)" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
