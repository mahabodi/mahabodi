"""fastmemory (third party, MIT) on KILT Wikipedia pages: topology size vs raw text, time, and whether the topology keeps
the page text and ids. Uses the `fastmemory` Python module built from source at a7dec441 (github.com/fastBuilderAI/memory),
because the PyPI 0.4.6 sdist embeds a 0-byte Linux helper (bin/linux/rust-louvain) and process_markdown returns "[]" on Linux.

Pages become ATF markdown in the shape of fastmemory's own sentence converter (build_huggingface_examples.py
generate_atfs): ## [ID: ATF_<wikipedia_id>], **Action:** Process_<title>, **Input:** {Context}, **Logic:** <abstract>,
**Data_Connections:** the first 3 words longer than 4 characters, **Access:** Role_Analyst, **Events:** Trigger_Default.
That conversion is ours: a different conversion could change the numbers.

    .venv/bin/python research/probe_fastmemory.py --out research/results/probe_fastmemory.json
"""
import argparse, json, os, random, re, resource, sys, time, zlib
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance  # noqa: E402
import fastmemory  # noqa: E402


def atf_md(ids, titles, abstracts):
    out = []
    for i, ti, ab in zip(ids, titles, abstracts):
        nouns = [w.lower() for w in re.sub(r"[^\w\s]", "", ab).split() if len(w) > 4][:3]
        out.append("## [ID: ATF_%s]\n**Action:** Process_%s\n**Input:** {Context}\n**Logic:** %s\n**Data_Connections:** %s\n"
                   "**Access:** Role_Analyst\n**Events:** Trigger_Default\n\n" % (i, re.sub(r"\W+", "_", ti), ab.replace("\n", " "),
                                                                                 ", ".join("[%s]" % x for x in nouns)))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="/media/sda/data/kilt/pages/part-0000.parquet")
    ap.add_argument("--sizes", default="1000,3000,10000")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "probe_fastmemory.json"))
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    T = pq.read_table(a.pages, columns=["wikipedia_id", "title", "abstract"]).slice(0, max(sizes)).to_pydict()
    res = {"fastmemory_module": fastmemory.__file__, "provenance": provenance(), "steps": []}
    for n in sizes:
        ids, ti, ab = T["wikipedia_id"][:n], T["title"][:n], T["abstract"][:n]
        raw = "".join(t + a_ for t, a_ in zip(ti, ab)).encode()
        md = atf_md(ids, ti, ab)
        t0 = time.time(); s = fastmemory.process_markdown(md); dt = time.time() - t0
        s = s if isinstance(s, str) else json.dumps(s)
        rng = random.Random(0); pick = rng.sample(range(n), min(200, n))
        step = {"n": n, "raw_text_bytes": len(raw), "atf_markdown_bytes": len(md.encode()), "topology_json_bytes": len(s.encode()),
                "raw_over_topology": round(len(raw) / max(1, len(s.encode())), 3),
                "zlib9_raw_ratio": round(len(raw) / len(zlib.compress(raw, 9)), 3), "seconds": round(dt, 1),
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                "sampled_pages": len(pick),
                "abstract_first200_found_verbatim": sum(1 for i in pick if ab[i].replace("\n", " ")[:200] in s),
                "atf_id_found": sum(1 for i in pick if ("ATF_%s" % ids[i]) in s)}
        res["steps"].append(step); print(step, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
