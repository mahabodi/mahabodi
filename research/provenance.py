"""Machine / build provenance recorded in every result file (required for results produced on any
machine other than the original macOS box, and harmless everywhere): host, OS, CPU, package versions,
repo git rev, native module sha256 and model file sha256."""
import hashlib, os, platform, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _cpu():
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


MODEL_FILES = tuple("models/%s/%s" % (m, f) for m in ("laya-v2", "minilm")
                    for f in ("model.onnx", "tokenizer.json", "tokenizer_config.json", "embedder_config.json", "config.json"))


def provenance(models=MODEL_FILES):
    import importlib.metadata as md
    out = {"hostname": platform.node(), "os": platform.platform(), "cpu": _cpu(), "logical_cpus": os.cpu_count(),
           "python": platform.python_version(), "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    for p in ("torch", "onnxruntime", "transformers", "numpy", "laya"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    out["git_rev"] = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    diff = subprocess.run(["git", "-C", ROOT, "diff", "HEAD", "--", "crates", "bindings", "research", "scripts"],
                          capture_output=True).stdout
    out["git_dirty"] = bool(diff)
    out["git_diff_sha256"] = hashlib.sha256(diff).hexdigest() if diff else None  # ties a dirty run to exact source
    untracked = subprocess.run(["git", "-C", ROOT, "ls-files", "--others", "--exclude-standard", "--", "crates", "bindings", "research", "scripts"],
                               capture_output=True, text=True).stdout.split()
    out["untracked_sha256"] = {f: _sha(os.path.join(ROOT, f)) for f in untracked}
    out["git_dirty"] = out["git_dirty"] or bool(untracked)
    try:
        import mahabodi
        d = os.path.dirname(mahabodi.__file__)
        out["mahabodi_module"] = mahabodi.__file__
        out["native_sha256"] = {f: _sha(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".so")}
    except Exception as e:  # noqa: BLE001
        out["mahabodi_module"] = "unavailable: %s" % e
    out["models_sha256"] = {m: _sha(os.path.join(ROOT, m)) for m in models if os.path.exists(os.path.join(ROOT, m))}
    try:
        import torch
        out["torch_cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        out["torch_cuda"] = None
    return out
