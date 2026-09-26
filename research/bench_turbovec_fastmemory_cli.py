"""F arm of research/PREREG_TURBOVEC.md: the fastmemory CLI at 64cb29b (the newest commit whose CLI compiles; its macOS
rust-louvain helper is a universal binary, so it runs on macOS x86_64 and reproduces fastmemory's shipped
example/world_events/output.json). Every CLI run happens under `sandbox-exec` with outbound network denied, so the CLI's
license telemetry ping cannot be sent (recorded as ping_blocked).

Search = fastmemory's query::search_memory (src/query.rs lines 3-85 at 64cb29b), ported below line for line:
search_memory / has_match / extract_deepest_matching_block. A parity check compares the port with the real
`fastmemory query` CLI. The CLI rebuilds (re-clusters) the memory on every query, and clustering ties vary per process, so parity
is checked on the SET of Function ids each returns. CLI-vs-CLI variation between two runs is reported alongside.

Page identity: a returned Function node `F_ATF_<wikipedia_id>` identifies that page. Pages are ranked in the order they
appear in the result (search_memory returns blocks without scores); recall@k counts the gold page among the first k.

Input: research/results/bench_turbovec_inputs/fm_atf_<n>.md (the same ATF markdown as probe_fastmemory.atf_md, written on
Ubuntu) and queries_<n>.json. Output: research/results/bench_turbovec_F_<n>.json.

    python3 research/bench_turbovec_fastmemory_cli.py --cli ~/tools/src/fm-cli/target/release/fastmemory --n 10000
"""
import argparse, json, os, re, resource, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SB = "(version 1)\n(allow default)\n(deny network*)\n"


# ---- port of fastmemory src/query.rs @64cb29b ----
def has_match(val, query):
    q = query.lower()
    if isinstance(val, dict):
        for k in ("name", "action", "id"):
            if isinstance(val.get(k), str) and q in val[k].lower():
                return True
        for n in val.get("nodes", []) if isinstance(val.get("nodes"), list) else []:
            if has_match(n, query):
                return True
        for b in val.get("sub_blocks", []) if isinstance(val.get("sub_blocks"), list) else []:
            if has_match(b, query):
                return True
    return False


def extract_deepest(block, query):
    if not has_match(block, query):
        return None
    subs = block.get("sub_blocks") if isinstance(block.get("sub_blocks"), list) else None
    if subs is not None:
        deeper = [m for m in (extract_deepest(sb, query) for sb in subs) if m is not None]
        if deeper:
            c = dict(block); c["sub_blocks"] = deeper
            return c
    node_matches = any(has_match(n, query) for n in (block.get("nodes") if isinstance(block.get("nodes"), list) else []))
    q = query.lower()
    self_matches = any(isinstance(block.get(k), str) and q in block[k].lower() for k in ("name", "id"))
    if node_matches or self_matches:
        c = dict(block); c["sub_blocks"] = []
        return c
    return None


def search_memory(memory, query):
    return [d for d in (extract_deepest(b, query) for b in memory if isinstance(memory, list)) if d is not None]
# ---- end port ----


def function_ids(v, out):
    if isinstance(v, dict):
        if v.get("topology_level") == "Function" and isinstance(v.get("id"), str) and v["id"] not in out:
            out.append(v["id"])
        for x in v.values():
            function_ids(x, out)
    elif isinstance(v, list):
        for x in v:
            function_ids(x, out)
    return out


def run_cli(cli, args, sb):
    t = time.time()
    p = subprocess.run(["sandbox-exec", "-f", sb, cli] + args, capture_output=True, text=True)
    return p, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--inputs", default=os.path.join(HERE, "results", "bench_turbovec_inputs"))
    ap.add_argument("--parity-queries", type=int, default=25)
    a = ap.parse_args()
    sb = os.path.join(a.inputs, "deny_net.sb"); open(sb, "w").write(SB)
    md = os.path.join(a.inputs, "fm_atf_%d.md" % a.n)
    Q = json.load(open(os.path.join(a.inputs, "queries_%d.json" % a.n)))
    p, build_s = run_cli(a.cli, ["build", md], sb)
    if p.returncode != 0:
        raise SystemExit("CLI build failed: " + p.stderr[-800:])
    memory_json = p.stdout
    mem = json.loads(memory_json)
    ids = set(i.strip() for i in re.findall(r"\[ID:\s*([^\]]+)\]", open(md).read()))
    surviving = set(x[len("F_"):] for x in function_ids(mem, []))
    # recall via the port
    ranks, lat, sizes = [], [], []
    for x in Q:
        t = time.perf_counter(); res = search_memory(mem, x["q"]); lat.append((time.perf_counter() - t) * 1000)
        pages = [int(f[len("F_ATF_"):]) for f in function_ids(res, []) if f.startswith("F_ATF_") and f[len("F_ATF_"):].isdigit()]
        sizes.append(len(pages))
        gold = int(x["wikipedia_id"])
        ranks.append(pages.index(gold) + 1 if gold in pages else None)
    # parity: port vs real CLI query (sandboxed), on Function-id sets; plus CLI-vs-CLI variation
    par = []
    for x in Q[:a.parity_queries]:
        c1, _ = run_cli(a.cli, ["query", md, x["q"]], sb)
        c2, _ = run_cli(a.cli, ["query", md, x["q"]], sb)
        s1 = set(function_ids(json.loads(c1.stdout or "[]"), [])); s2 = set(function_ids(json.loads(c2.stdout or "[]"), []))
        sp = set(function_ids(search_memory(mem, x["q"]), []))
        par.append({"q": x["q"], "port_eq_cli1": sp == s1, "cli1_eq_cli2": s1 == s2, "sizes": [len(sp), len(s1), len(s2)]})
    lat.sort()
    out = {"arm": "F", "code_path": "fastmemory CLI @64cb29b, macOS x86_64, sandbox-exec deny network*",
           "machine_note": "different machine from the other arms: build time, latency and RSS are not comparable",
           "ping_blocked": "yes: every CLI run is under sandbox-exec with (deny network*)", "stored_bytes": len(memory_json.encode()),
           "build_s": round(build_s, 1), "atf_ids_in_input": len(ids), "atf_ids_surviving": len(surviving & ids),
           "ranks": ranks, "result_pages_per_query_mean": sum(sizes) / max(1, len(sizes)),
           "latency_ms_p50": lat[len(lat) // 2] if lat else None, "latency_ms_p95": lat[int(.95 * len(lat))] if lat else None,
           "parity": {"n": len(par), "port_eq_cli": sum(p_["port_eq_cli1"] for p_ in par),
                      "cli_eq_cli": sum(p_["cli1_eq_cli2"] for p_ in par), "items": par},
           "peak_rss_mb_python": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576)}
    json.dump(out, open(os.path.join(HERE, "results", "bench_turbovec_F_%d.json" % a.n), "w"), indent=1)
    print({k: v for k, v in out.items() if k not in ("ranks", "parity")}, out["parity"]["port_eq_cli"], "/", out["parity"]["n"])


if __name__ == "__main__":
    main()
