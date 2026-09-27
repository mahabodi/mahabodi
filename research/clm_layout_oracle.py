"""Layout oracle for CLM (research/PREREG_CLM.md, reviewer ruling 2026-09-26): which state/candidate layout reproduces
CLM's own README typed-decision example? CLM's README example is CLM's reference, not one of our test items, so fitting
to it is allowed.

The variants below are fixed BEFORE any is tried (committed first). Each replaces clm.engine.build_pairs; scoring,
heads and token accounting stay CLM's own (`Engine.answer`, a fresh Engine per variant so the cache is cold).
Acceptance: all three README values within 1e-3 AND cold `usage.input_tokens` == 106. If exactly one variant passes,
it is adopted; if none passes, CLM stays "not evaluated".

    CLM_HEAD=<.pt> CLM_EMB_URL=... .venv/bin/python research/clm_layout_oracle.py --qwen <Qwen3-8B dir>
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_clm import README_EXAMPLE  # noqa: E402

TOL, TOKENS = 1e-3, 106


def make_variants(tok):
    from clm import schema
    chat = lambda s: tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=False)

    def noul_plain(q):
        ins = schema.to_text(q.get("instructions")).strip()
        return ["false", "true"], [f"No. This is false: {ins}", f"Yes. This is true: {ins}"]

    def noul_bare(q):
        return ["false", "true"], ["false", "true"]

    def cands(q, noul=None):
        return noul(q) if (noul and q["type"] == "noul") else schema.candidates(q)

    def state(st, q, qfirst=False, tmpl=False):
        s, i = schema.to_text(st).strip(), schema.to_text(q.get("instructions")).strip()
        t = f"{i}\n\n{s}" if qfirst else f"{s}\n\n{i}"
        return chat(t) if tmpl else t

    def V(qfirst=False, tmpl=False, noul=None, tmpl_cands=False):
        def bp(st, questions):
            out = {}
            for qid, q in questions.items():
                keys, texts = cands(q, noul)
                if tmpl_cands:
                    texts = [chat(x) for x in texts]
                out[qid] = (state(st, q, qfirst, tmpl), keys, texts)
            return out
        return bp

    return {
        "V1_current_code": V(),
        "V2_chat_template_state": V(tmpl=True),
        "V3_question_first": V(qfirst=True),
        "V4_question_first_chat_template": V(qfirst=True, tmpl=True),
        "V5_noul_without_key_prefix": V(noul=noul_plain),
        "V6_noul_bare_keys": V(noul=noul_bare),
        "V7_chat_template_state_and_candidates": V(tmpl=True, tmpl_cands=True),
        "V8_chat_template_state_noul_without_prefix": V(tmpl=True, noul=noul_plain),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "clm_layout_oracle.json"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import clm.engine as ce
    from clm import Engine
    tok = AutoTokenizer.from_pretrained(a.qwen)
    orig = ce.build_pairs
    res = {"acceptance": {"tolerance": TOL, "input_tokens": TOKENS}, "expected": README_EXAMPLE["expected"], "variants": {}}
    for name, bp in make_variants(tok).items():
        ce.build_pairs = bp
        e = Engine(checkpoint=os.environ["CLM_HEAD"], device="cpu", emb_url=os.environ.get("CLM_EMB_URL"), action_cache=0)
        r = e.answer(README_EXAMPLE["state"], README_EXAMPLE["questions"])
        A = {k: (v if isinstance(v, dict) else v.__dict__) for k, v in r["answers"].items()}
        got = {"urgency.noul": A["urgency"]["noul"], "department.billing": A["department"]["probabilities"]["billing"],
               "frustration.score": A["frustration"]["score"]}
        diff = {k: abs(got[k] - v) for k, v in README_EXAMPLE["expected"].items()}
        ntok = (r.get("usage") or {}).get("input_tokens")
        ok = all(d <= TOL for d in diff.values()) and ntok == TOKENS
        res["variants"][name] = {"got": {k: round(v, 5) for k, v in got.items()}, "max_abs_diff": round(max(diff.values()), 5),
                                 "input_tokens": ntok, "pass": ok,
                                 "sample": {qid: {"state": p[0], "candidates": p[2]} for qid, p in bp(README_EXAMPLE["state"], README_EXAMPLE["questions"]).items()}}
        print(name, res["variants"][name]["got"], "tokens", ntok, "PASS" if ok else "fail", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    ce.build_pairs = orig
    passing = [n for n, v in res["variants"].items() if v["pass"]]
    res["passing"] = passing
    res["verdict"] = ("adopt " + passing[0]) if len(passing) == 1 else ("none passes: CLM stays not evaluated" if not passing else "several pass: ambiguous, not adopted")
    json.dump(res, open(a.out, "w"), indent=1)
    print("VERDICT:", res["verdict"])


if __name__ == "__main__":
    main()
