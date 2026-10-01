"""MahaBodi as an agent router: the NON-LLM arms of research/PREREG_ROUTER.md (clarification 1: the LLM arms
L, L2 and LF are deferred; only M, ML and E run, and only the secondary comparisons M vs E and ML vs E are
reported, labelled "LLM arms pending").

Suites: CLINC150 "plus" (clinc_oos/plus, test), Banking77 (PolyAI/banking77, test), MASSIVE en-US
(AmazonScience/massive en-US, test).

Phases (run on the Mac mini, one benchmark at a time; nothing here calls any API):

  --phase selftest  Pure-Python checks, no data or models: Newcombe (1998) method-10 worked examples to 4 decimals,
                  the selection rule, the id hash, McNemar. Also run first by --phase items.

  --phase items   Build the fresh test items, the 200 validation items and the labelled pool per suite, after
                  excluding every test item an earlier MahaBodi run scored (reconstructed below and asserted
                  against the gold lists those runs saved). First enforces the audit of PREREG_ROUTER.md
                  clarification 3 (AUDIT below): every results/*.json naming a suite is classified, test-using files
                  are asserted inside the excluded spans, text-only files are excluded by text match. Writes research/results/router_items.json and prints
                  the SHA-256 of each suite's sorted test-id list joined by newlines (PREREG_SCALE_V2 clar. 1).
  --phase run --arm M|ML|E
                  M   Bodi().load_laya(models/laya-v2), decide() with shipped defaults, cache off. Options = the
                      intent names, in the question/option format of the earlier published runs. CLINC150: the
                      shipped opt-in out-of-scope gate at the W11 thresholds (bench_clinc_oos.json arm C).
                  ML  learn(pool, calibrate=200) as shipped, then decide() with the cache off.
                  E   all-MiniLM-L6-v2 kNN (MahaBodi's own models/minilm export via embed_text), k = 10 over the
                      labelled pool, majority vote, ties by summed similarity.
                  Each arm runs clean and misspelled (bench_retrieval.typo_all, random.Random(item position)).
                  Latency per item, batch 1, over the scored pass in item order, no warm-up (PREREG_SCALE_V2
                  clarification 3d). Output research/results/router_<arm>.json, checkpointed per suite x variant.
  --phase score   Wilson CIs, exact McNemar, Newcombe's paired hybrid-score interval (method 10), for M vs E and
                  ML vs E per suite per variant (never pooled). Writes research/results/router_score.json.

    .venv/bin/python research/bench_router.py --phase items
    .venv/bin/python research/bench_router.py --phase run --arm M      # then ML, then E
    .venv/bin/python research/bench_router.py --phase score
"""
import argparse, hashlib, json, math, os, platform, random, re, resource, sys, time
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "research", "results")
ITEMS = os.path.join(R, "router_items.json")
sys.path.insert(0, HERE)

SEED = 20260930
N_TEST, N_VAL, K_E, CALIBRATE = 1000, 200, 10, 200
SUITES = ("clinc150", "banking77", "massive")
VARIANTS = ("clean", "misspelled")
LLM_ARMS = ("L", "L2", "LF")
SOURCES = {"clinc150": ("clinc_oos", "plus"), "banking77": ("PolyAI/banking77", None), "massive": ("AmazonScience/massive", "en-US")}

# Question / option formats of the earlier published MahaBodi runs (reused verbatim; see fmt() below).
INS_CLINC = "Which intent does `utterance` express?"                     # bench_clinc.py, bench_clinc_oos.py, check_oos_product.py
INS_BANKING = "Which banking intent does `message` express?"             # bench.py build_suites
INS_MASSIVE = "What is the user asking for in `utterance`?"              # bench_massive.py build


# ----------------------------------------------------------------------------------------------- statistics
# research/bench.py's wilson and mcnemar, verbatim (bench.py imports torch and laya at module level, which the
# score phase does not need; store_parity_gate.py keeps the same light copies).
def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar(correct_a, correct_b):
    """Exact two-sided McNemar (binomial on discordant pairs)."""
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p": 1.0}
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return {"a_only": b, "b_only": c, "p": round(min(1.0, p), 6)}


def _wilson_raw(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_paired(correct_a, correct_b, z=1.96):
    """95 % CI for p_a - p_b on paired binary data: Newcombe (1998), method 10 (hybrid score, no continuity
    correction; phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d)), 0 when that product is 0)."""
    n = len(correct_a)
    assert n == len(correct_b) and n > 0
    a = sum(1 for x, y in zip(correct_a, correct_b) if x and y)
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    d = n - a - b - c
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = _wilson_raw(a + b, n, z)
    l2, u2 = _wilson_raw(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    diff = p1 - p2
    return {"diff": round(diff, 4), "ci95": [round(max(-1.0, diff - dl), 4), round(min(1.0, diff + du), 4)],
            "both": a, "a_only": b, "b_only": c, "neither": d, "phi": round(phi, 6), "method": "Newcombe 1998 method 10"}


# Newcombe (1998) worked examples, (both, A only, B only, neither) -> 95 % CI of p_A - p_B, method 10.
NEWCOMBE_1998 = [((36, 12, 2, 0), [0.0569, 0.3404]), ((1, 1, 0, 48), [-0.0479, 0.1039]), ((2, 1, 1, 46), [-0.0824, 0.0824])]


def selftest():
    """Pure-Python checks (no data, no models); --phase items runs this first."""
    for (a_, b_, c_, d_), want in NEWCOMBE_1998:
        x = [1] * (a_ + b_) + [0] * (c_ + d_)
        y = [1] * a_ + [0] * b_ + [1] * c_ + [0] * d_
        got = newcombe_paired(x, y)
        assert got["ci95"] == want and (got["both"], got["a_only"], got["b_only"], got["neither"]) == (a_, b_, c_, d_), \
            "Newcombe method 10 %s: got %s, want %s" % ((a_, b_, c_, d_), got["ci95"], want)
        rev = newcombe_paired(y, x)["ci95"]
        assert rev == [-want[1], -want[0]], rev
    t, n = select_fresh(3080, set(range(2000)), k=1000)
    rest = list(range(2000, 3080))
    random.Random(SEED).shuffle(rest)
    assert t == rest[:1000] and n == 1080 and len(set(t)) == 1000
    assert sha_ids(["2", "10"]) == hashlib.sha256(b"10\n2").hexdigest()
    assert mcnemar([1, 1, 0], [0, 1, 1]) == {"a_only": 1, "b_only": 1, "p": 1.0}
    assert pct([1, 2, 3, 4], 50) == 2.5
    print("selftest ok: Newcombe (1998) method-10 examples %s reproduced to 4 decimals; selection, sha, McNemar, percentile ok"
          % [w for _, w in NEWCOMBE_1998], flush=True)


def pct(xs, q):
    """Linear-interpolated percentile (numpy's default), pure Python."""
    s = sorted(xs)
    if not s:
        return None
    k = (len(s) - 1) * q / 100
    f = math.floor(k)
    return s[f] if f == len(s) - 1 else s[f] + (s[f + 1] - s[f]) * (k - f)


# ----------------------------------------------------------------------------------------------- helpers
def sha_ids(ids):
    """PREREG_SCALE_V2 clarification 1: SHA-256 of the sorted ids joined by newlines (ids are strings)."""
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode()).hexdigest()


def sha_texts(texts):
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


def norm(t):
    return " ".join(str(t).lower().split())


def select_fresh(n_rows, excluded, seed=SEED, k=N_TEST):
    """The pre-registered rule: the non-excluded split indices in file order, shuffled by random.Random(seed),
    first k. A fresh Random(seed) per call (per suite)."""
    rest = [i for i in range(n_rows) if i not in excluded]
    random.Random(seed).shuffle(rest)
    assert len(rest) >= k, "only %d eligible rows" % len(rest)
    return rest[:k], len(rest)


def load_split(name, cfg, split):
    """load_dataset, falling back to the Hub's parquet conversion when the repo still carries a loading script
    (unsupported by datasets >= 4). Returns (dataset, how)."""
    from datasets import load_dataset
    args = (name, cfg) if cfg else (name,)
    try:
        return load_dataset(*args, split=split), "load_dataset(%r, %r, split=%r)" % (name, cfg, split)
    except Exception as e:  # noqa: BLE001
        d = load_dataset(*args, split=split, revision="refs/convert/parquet")
        return d, "load_dataset(%r, %r, split=%r, revision='refs/convert/parquet') [default revision failed: %s]" % (
            name, cfg, split, str(e).splitlines()[0][:200])


def col(ds, c):
    return list(ds[c])


def shuffled_original_indices(ds, seed, text_col="text"):
    """Original split indices in the order of ds.shuffle(seed=seed), as the earlier harnesses drew them.
    Derived by shuffling an index column; asserted equal to the plain shuffle's texts row by row."""
    idx = col(ds.add_column("_router_idx", list(range(len(ds)))).shuffle(seed=seed), "_router_idx")
    plain = col(ds.shuffle(seed=seed), text_col)
    texts = col(ds, text_col)
    assert plain == [texts[i] for i in idx], "index-column shuffle differs from the plain shuffle"
    return idx


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 1048576 if platform.system() == "Darwin" else r / 1024, 1)  # bytes on macOS, KiB on Linux


def jload(name):
    return json.load(open(os.path.join(R, name)))


# ----------------------------------------------------------------------------------------------- phase: items
def excl_clinc(test):
    """bench_clinc.py: clinc_oos/plus test .shuffle(seed=7) positions 0..999; bench_clinc_oos.py (also re-scored by
    check_oos_embedder.py and check_oos_product.py): positions 1000..1999 of the same shuffle."""
    names = test.features["intent"].names
    oos = names.index("oos")
    idx = shuffled_original_indices(test, 7)[:2000]
    lab = col(test, "intent")
    gold = [names[lab[i]].replace("_", " ") if lab[i] != oos else "oos" for i in idx]
    assert gold[:1000] == jload("bench_clinc.json")["gold"], "bench_clinc.json gold differs from the seed-7 positions 0..999"
    assert gold[1000:2000] == jload("bench_clinc_oos_arrays.json")["test"]["gold"], "bench_clinc_oos_arrays.json test gold differs"
    ttexts = col(test, "text")
    ctx = {"covered": {"clinc7": (0, 2000)}, "gold": {"clinc7": gold}, "texts": {"clinc7": [ttexts[i] for i in idx]}, "target_texts": ttexts}
    report = [
        {"file": "research/bench_clinc.py -> results/bench_clinc.json", "method": "clinc_oos/plus test .shuffle(seed=7) positions 0..999 "
         "(re-derived; gold list asserted equal to the saved 'gold')", "n": 1000},
        {"file": "research/bench_clinc_oos.py -> results/bench_clinc_oos.json, bench_clinc_oos_arrays.json; also re-scored by "
         "check_oos_embedder.py and check_oos_product.py", "method": "same shuffle, positions 1000..1999 (re-derived; gold asserted "
         "equal to bench_clinc_oos_arrays.json test.gold)", "n": 1000}]
    return set(idx), report, "split index (clinc_oos has no id field; texts are unique in the test split)", ctx


def excl_banking(target):
    """Earlier Banking77 runs used mteb/banking77 (test, 3,076 rows), not PolyAI/banking77 (3,080): bench.py's
    build_suites = test.shuffle(seed=0) positions 0..n-1, sliced per run; bench_smoke_v1.py = the first n UNSHUFFLED
    rows. mteb rows are mapped onto PolyAI split indices by exact text (every PolyAI row with that text is excluded)."""
    mteb, how = load_split("mteb/banking77", None, "test")
    lt = col(mteb, "label_text")
    lab = [x.replace("_", " ") for x in sorted(set(lt))]
    idx = shuffled_original_indices(mteb, 0)[:2000]
    gold = [lab.index(lt[i].replace("_", " ")) for i in idx]
    checks = [("bench.json", ["suites", "banking77", "gold"], 0, 500),
              ("bench_fresh_experience.json", ["suites", "banking77", "gold"], 500, 1000),
              ("bench_fresh3_experience.json", ["suites", "banking77", "gold"], 1000, 1500),
              ("bench_fresh4_head.json", ["suites", "banking77", "gold"], 1500, 2000)]
    for f, path, lo, hi in checks:
        g = jload(f)
        for p in path:
            g = g[p]
        assert g == gold[lo:hi], "%s gold differs from mteb/banking77 test.shuffle(0)[%d:%d]" % (f, lo, hi)
    n_smoke = int(jload("bench_smoke.json")["env"]["n_per_suite"])
    lat = jload("latency_ubuntu_i9-9900X.json")
    n_lat = int(lat["calls"]) + int(lat["warmup"])
    assert n_lat <= 2000
    used = set(idx) | set(range(n_smoke))
    mtexts = col(mteb, "text")
    ttexts = col(target, "text")
    by_exact, by_norm = {}, {}
    for i, t in enumerate(ttexts):
        by_exact.setdefault(t, []).append(i)
        by_norm.setdefault(norm(t), []).append(i)
    excluded, n_exact, n_norm, multi, missing = set(), 0, 0, 0, []
    for m in sorted(used):
        hit = by_exact.get(mtexts[m])
        if hit:
            n_exact += 1
        else:
            hit = by_norm.get(norm(mtexts[m]))
            n_norm += bool(hit)
        if not hit:
            missing.append(m)
            continue
        multi += len(hit) > 1
        excluded.update(hit)
    assert not missing, "mteb/banking77 rows with no PolyAI text match: %s" % missing[:10]
    assert sorted(set(lt)) == sorted(target.features["label"].names), "mteb and PolyAI label names differ"
    ctx = {"covered": {"b77s0": (0, 2000), "b77raw": (0, n_smoke)}, "gold": {"b77s0": gold},
           "texts": {"b77s0": [mtexts[i] for i in idx], "b77raw": mtexts}, "target_texts": ttexts}
    report = [
        {"file": "research/bench.py -> results/bench.json (also bench_ece.py, bench_experience*.py, bench_knn_only.py, bench_clm.py, "
         "finetune_laya_*.py test, bench_harness_check.json n=8)", "method": "mteb/banking77 test .shuffle(seed=0) positions 0..499 "
         "(gold asserted equal)", "n": 500},
        {"file": "research/bench_fresh_experience.py -> results/bench_fresh_experience.json", "method": "same shuffle, positions 500..999 "
         "(gold asserted)", "n": 500},
        {"file": "research/bench_fresh_experience.py --start 1000 -> results/bench_fresh3_experience.json (also finetune_laya_*.py fresh3)",
         "method": "same shuffle, positions 1000..1499 (gold asserted)", "n": 500},
        {"file": "research/bench_fresh4_head.py -> results/bench_fresh4_head.json", "method": "same shuffle, positions 1500..1999 (gold asserted)", "n": 500},
        {"file": "research/bench_latency.py -> results/latency_ubuntu_i9-9900X.json", "method": "build_suites(%d): same shuffle positions "
         "0..%d (timing only; inside 0..1999)" % (n_lat, n_lat - 1), "n": n_lat},
        {"file": "research/bench_smoke_v1.py -> results/bench_smoke.json", "method": "first %d UNSHUFFLED mteb test rows (n_per_suite from the file)" % n_smoke, "n": n_smoke},
        {"mapping": "mteb/banking77 -> PolyAI/banking77 test by exact text (normalised-text fallback)", "mteb_rows_used": len(used),
         "matched_exact": n_exact, "matched_normalised": n_norm, "mteb_rows_matching_several_polyai_rows": multi,
         "polyai_rows_excluded": len(excluded), "mteb_source": how}]
    return excluded, report, "PolyAI split index; earlier runs' items located by text (they used mteb/banking77, a different 3,076-row file)", ctx


def excl_massive(target):
    """bench_massive.py: mteb/amazon_massive_intent 'en' test, first per_lang (100) rows, in file order. Mapped onto
    AmazonScience/massive en-US test by the shared MASSIVE 'id' (text asserted equal); text if mteb carries no id."""
    mt, how = load_split("mteb/amazon_massive_intent", "en", "test")
    bm = jload("bench_massive.json")
    per_lang, seed, n_opts = int(bm["per_lang"]), int(bm["seed"]), int(bm["n_options"])
    rows = [mt[i] for i in range(per_lang)]
    labels = sorted(set(col(mt, "label_text")))  # re-derive bench_massive.build's gold to prove the rows are these
    rng = random.Random(seed)
    gold = []
    for r in rows:
        pool = [x for x in labels if x != r["label_text"]]
        keys = [r["label_text"]] + rng.sample(pool, min(n_opts - 1, len(pool)))
        rng.shuffle(keys)
        gold.append(keys.index(r["label_text"]))
    assert gold == bm["per_language"]["en"]["gold"], "bench_massive.json en gold differs from the first %d mteb rows" % per_lang
    tid, tutt = col(target, "id"), col(target, "utt")
    excluded, method = set(), None
    if "id" in rows[0] and len(set(tid)) == len(tid):
        pos = {str(x): i for i, x in enumerate(tid)}
        for r in rows:
            i = pos[str(r["id"])]
            assert norm(tutt[i]) == norm(r["text"]), "id %s: texts differ" % r["id"]
            excluded.add(i)
        method = "MASSIVE id (text asserted equal)"
    else:
        by = {}
        for i, t in enumerate(tutt):
            by.setdefault(norm(t), []).append(i)
        for r in rows:
            excluded.update(by[norm(r["text"])])
        method = "normalised text (mteb rows carry no usable id)"
    ctx = {"covered": {"massive_raw": (0, per_lang)}, "gold": {"massive_raw": gold}, "texts": {"massive_raw": [r["text"] for r in rows]},
           "target_texts": tutt}
    report = [{"file": "research/bench_massive.py -> results/bench_massive.json (per_language.en)",
               "method": "mteb/amazon_massive_intent 'en' test, first %d rows in file order (gold re-derived with random.Random(%d) "
               "and asserted equal); mapped by %s" % (per_lang, seed, method), "n": per_lang, "mteb_source": how}]
    return excluded, report, "AmazonScience/massive 'id' field (split index also stored)", ctx


# ----------------------------------------------------------------------------------------------- audit (PREREG_ROUTER clar. 3)
# Every research/results/*.json whose text matches AUDIT_RE, plus the item-level files that don't (EXTRA), classified:
#   trainval  train/validation rows only            test  test positions of a named shuffle (must be covered)
#   none      not item-level for these suites       text  items recoverable only as texts -> excluded by text match
# Copies: clinc7 = clinc_oos/plus test .shuffle(seed=7); b77s0 = mteb/banking77 test .shuffle(seed=0); b77raw = mteb/banking77
# test file order; massive_raw = mteb/amazon_massive_intent 'en' test file order. Checks: ("gold", path, copy, lo, hi),
# ("len", path, n), ("eq", path, value), ("texts_sha", path, copy, lo, hi), ("npz_gold", npz, key, copy, lo, hi).
AUDIT_RE = re.compile(r"banking77|clinc|massive", re.I)
B77_TEST_0_499 = [("b77s0", 0, 500)]
FT_FULL = "finetune_laya_full.py:139-140 (train_suites(2000,300) = mteb train.shuffle(2); build_suites(1500)), :160-161 (test [:500], fresh3 [1000:1500])"
FT_HEAD = "finetune_laya_head.py:118,126-127 (build_suites(1500), build_suites(500)), :147,155 (fresh3 [1000:1500]); train via tune_experience.py:43"
TUNE_NPZ = "tune_experience.py:33-44,149,156 (suites(2000,300): mteb/banking77 TRAIN .shuffle(2)[0:2300]; npz cache tune_text_*_m2000_v300)"
AUDIT = [
    # ---- Banking77
    dict(f="bench.json", suite="banking77", cls="test", ranges=B77_TEST_0_499, ev="bench.py:59-61,80-83 (mteb/banking77 test .shuffle(0), first n); file env.n_per_suite = 500",
         checks=[("eq", "env/n_per_suite", 500), ("gold", "suites/banking77/gold", "b77s0", 0, 500)]),
    dict(f="bench_harness_check.json", suite="banking77", cls="test", ranges=[("b77s0", 0, 8)], ev="bench.py (same env layout) with --n 8: positions 0..7",
         checks=[("eq", "env/n_per_suite", 8), ("gold", "suites/banking77/gold", "b77s0", 0, 8)]),
    dict(f="bench_smoke.json", suite="banking77", cls="test", ranges=[("b77raw", 0, 12)], ev="bench_smoke_v1.py:62-67 (mteb/banking77 test, list(d)[:n], UNSHUFFLED); env.n_per_suite = 12",
         checks=[("eq", "env/n_per_suite", 12)]),
    dict(f="bench_ece.json", suite="banking77", cls="test", ranges=B77_TEST_0_499, ev="bench_ece.py:14,77 (test = build_suites(500), bench.json's 500); validation train.shuffle(2)[2000:2300] (:6)",
         checks=[("npz_gold", "bench_ece_probs.npz", "banking77__bodi__test_Y", "b77s0", 0, 500)]),
    dict(f="bench_experience.json", suite="banking77", cls="test", ranges=B77_TEST_0_499, ev="bench_experience.py:45,66 (build_suites(--n 500)); memory = train (tune_experience.py:43)",
         checks=[("eq", "suites/banking77/bodi_experience_per_suite/n", 500)]),
    dict(f="bench_experience_gated.json", suite="banking77", cls="test", ranges=B77_TEST_0_499, ev="bench_experience.py:45,66 (--gate-file run)",
         checks=[("eq", "suites/banking77/bodi_experience_global/n", 500)]),
    dict(f="bench_experience_trust.json", suite="banking77", cls="test", ranges=B77_TEST_0_499, ev="bench_experience.py:45,66 (--auto-trust run)",
         checks=[("eq", "suites/banking77/bodi_experience_per_suite/n", 500)]),
    dict(f="bench_knn_only.json", suite="banking77", cls="test", ranges=B77_TEST_0_499, ev="bench_knn_only.py:30 (build_suites(500))",
         checks=[("len", "suites/banking77/global/pred", 500)]),
    dict(f="bench_clm_INVALID_degenerate.json", suite="banking77", cls="test", ranges=B77_TEST_0_499, ev="bench_clm.py:82,101 (build_suites(--n 500)); file suites.banking77.n = 500",
         checks=[("eq", "suites/banking77/n", 500), ("len", "suites/banking77/pred", 500)]),
    dict(f="bench_fresh_experience.json", suite="banking77", cls="test", ranges=[("b77s0", 500, 1000)], ev="bench_fresh_experience.py:74,79,87 (positions start..start+500, start 500)",
         checks=[("gold", "suites/banking77/gold", "b77s0", 500, 1000)]),
    dict(f="bench_fresh3_experience.json", suite="banking77", cls="test", ranges=[("b77s0", 1000, 1500)], ev="bench_fresh_experience.py:23-24,74,87 (--start 1000)",
         checks=[("gold", "suites/banking77/gold", "b77s0", 1000, 1500)]),
    dict(f="bench_fresh4_head.json", suite="banking77", cls="test", ranges=[("b77s0", 1500, 2000)], ev="bench_fresh4_head.py:122,138 (positions 1500..1999); selection rows train.shuffle(2)[2300:2600] (:141-143)",
         checks=[("gold", "suites/banking77/gold", "b77s0", 1500, 2000)]),
    dict(f="latency_ubuntu_i9-9900X.json", suite="banking77", cls="test", ranges=[("b77s0", 0, 550)], ev="bench_latency.py:79-80,87 (build_suites(warmup+calls = 550), timing only)",
         checks=[("eq", "calls", 500), ("eq", "warmup", 50)]),
    dict(f="laya_full_finetuned.json", suite="banking77", cls="test", ranges=[("b77s0", 0, 500), ("b77s0", 1000, 1500)], ev=FT_FULL, checks=[]),
    dict(f="laya_full_finetuned_attempt1_oom.json", suite="banking77", cls="test", ranges=[("b77s0", 0, 500), ("b77s0", 1000, 1500)], ev=FT_FULL + "; banking77 not_run (counted as used anyway)", checks=[]),
    dict(f="laya_full_finetuned_attempt2.json", suite="banking77", cls="test", ranges=[("b77s0", 0, 500), ("b77s0", 1000, 1500)], ev=FT_FULL + "; banking77 not_run (counted as used anyway)", checks=[]),
    dict(f="laya_full_finetuned_attempt3.json", suite="banking77", cls="test", ranges=[("b77s0", 0, 500), ("b77s0", 1000, 1500)], ev=FT_FULL + "; banking77 not_run (counted as used anyway)", checks=[]),
    dict(f="laya_full_finetuned_attempt4_fp16math.json", suite="banking77", cls="test", ranges=[("b77s0", 0, 500), ("b77s0", 1000, 1500)], ev=FT_FULL + "; banking77 not_run (counted as used anyway)", checks=[]),
    dict(f="laya_head_finetuned_ubuntu.json", suite="banking77", cls="test", ranges=[("b77s0", 0, 500), ("b77s0", 1000, 1500)], ev=FT_HEAD + "; file test_items 'bench.json 0..499', fresh3 'positions 1000..1499'",
         checks=[("len", "suites/banking77/fresh3/pred", 500)]),
    dict(f="laya_head_finetuned_rerun_savehead.json", suite="banking77", cls="test", ranges=[("b77s0", 0, 500), ("b77s0", 1000, 1500)], ev=FT_HEAD + "; file test_items 'bench.json 0..499'", checks=[]),
    dict(f="laya_head_finetuned_banking77_record_repro.json", suite="banking77", cls="test", ranges=[("b77s0", 0, 500), ("b77s0", 1000, 1500)], ev=FT_HEAD + "; file fresh3 'positions 1000..1499'", checks=[]),
    dict(f="calibration_train.json", suite="banking77", cls="trainval", ev="calibrate_train.py:16,23 (learn(calibrate=200) on suites(2000,300) memory = mteb TRAIN .shuffle(2)[:2000])"),
    dict(f="check_agree_real.json", suite="banking77", cls="trainval", ev="check_agree_real.py:14,19-20 (memory and val from suites(2000,300): TRAIN .shuffle(2)[0:2300])"),
    dict(f="probe_minilm_knn.json", suite="banking77", cls="trainval", ev="probe_minilm_knn.py:11-17 (suites(2000,300) mem/val: TRAIN)"),
    dict(f="tune_knn_only.json", suite="banking77", cls="trainval", ev="tune_knn_only.py:10 (npz cache); " + TUNE_NPZ),
    dict(f="tune_agree_gate.json", suite="banking77", cls="trainval", ev="tune_agree_gate.py:26 (npz cache); " + TUNE_NPZ),
    dict(f="tune_margin_gate.json", suite="banking77", cls="trainval", ev="tune_margin_gate.py:14 (npz cache); " + TUNE_NPZ),
    dict(f="tune_memory_first.json", suite="banking77", cls="trainval", ev="tune_memory_first.py:30,33 (npz cache + calibration_train.json); " + TUNE_NPZ),
    dict(f="tune_experience_text.json", suite="banking77", cls="trainval", ev=TUNE_NPZ + "; test split read for label names only (tune_experience.py:42)"),
    dict(f="tune_experience_text_trust.json", suite="banking77", cls="trainval", ev=TUNE_NPZ + " (--auto-trust)"),
    dict(f="tune.json", suite="banking77", cls="trainval", ev="tune.py:33-34 (mteb/banking77 TRAIN .shuffle(1)[:300]; test split for label names only); file splits.banking77 = 'train seed=1'",
         checks=[("eq", "splits/banking77", "train seed=1")]),
    dict(f="bench_typed.json", suite="banking77", cls="none", ev="bench_typed.py:32 (LocalLLaMA/typed-decisions); 'banking77' only in concurrent_load note"),
    dict(f="clm_template_samples.json", suite="banking77", cls="text", ev="producer script not in the repo; samples.banking77.state_text holds one utterance ('message: ...')",
         texts="clm_samples"),
    # ---- CLINC150
    dict(f="bench_clinc.json", suite="clinc150", cls="test", ranges=[("clinc7", 0, 1000)], ev="bench_clinc.py:30-32,87 (test .shuffle(7) first 1000); validation .shuffle(7)[:600] (:79)",
         checks=[("gold", "gold", "clinc7", 0, 1000), ("eq", "test_n", 1000)]),
    dict(f="bench_clinc_oos.json", suite="clinc150", cls="test", ranges=[("clinc7", 1000, 2000)], ev="bench_clinc_oos.py:76-84 (test .shuffle(7) positions 1000..1999; validation oos + shuffle(7) in-scope)",
         checks=[("eq", "test_n", 1000), ("len", "test/C/pred", 1000)]),
    dict(f="bench_clinc_oos_arrays.json", suite="clinc150", cls="test", ranges=[("clinc7", 1000, 2000)], ev="bench_clinc_oos.py:84,109-111; file test_items 'test.shuffle(7) positions 1000..1999'",
         checks=[("gold", "test/gold", "clinc7", 1000, 2000), ("eq", "test_items", "test.shuffle(7) positions 1000..1999")]),
    dict(f="check_oos_embedder.json", suite="clinc150", cls="test", ranges=[("clinc7", 1000, 2000)], ev="check_oos_embedder.py:18-20 (same 1000..1999)",
         checks=[("eq", "test/n", 1000)]),
    dict(f="check_oos_product_prefix_build.json", suite="clinc150", cls="test", ranges=[("clinc7", 1000, 2000)], ev="check_oos_product.py:26-28 (same 1000..1999)",
         checks=[("texts_sha", "test/texts_sha256", "clinc7", 1000, 2000)]),
    dict(f="check_oos_product_ubuntu.json", suite="clinc150", cls="test", ranges=[("clinc7", 1000, 2000)], ev="check_oos_product.py:26-28 (same 1000..1999)",
         checks=[("texts_sha", "test/texts_sha256", "clinc7", 1000, 2000)]),
    dict(f="check_oos_product.json", suite="clinc150", cls="test", ranges=[("clinc7", 1000, 2000)], optional=True, ev="check_oos_product.py:26-28,82 default output (present only where it was run)",
         checks=[("texts_sha", "test/texts_sha256", "clinc7", 1000, 2000)]),
    dict(f="retrieval_2000.json", suite="clinc150", cls="none", ev="bench_retrieval.py:94 (rajpurkar/squad); regex hit is the misspelling 'clincal' in a SQuAD question"),
    dict(f="retrieval_2000_qfix.json", suite="clinc150", cls="none", ev="bench_retrieval.py:94 (rajpurkar/squad); regex hit is 'clincal' in a SQuAD question"),
    dict(f="retrieval_fastmemory_p2000.json", suite="clinc150", cls="none", ev="bench_retrieval_fastmemory.py (rajpurkar/squad, same sample as bench_retrieval.py); regex hit is 'clincal' in a SQuAD question"),
    # ---- MASSIVE
    dict(f="bench_massive.json", suite="massive", cls="test", ranges=[("massive_raw", 0, 100)], ev="bench_massive.py:41,45 (mteb/amazon_massive_intent <lang> test, list(d)[:per_lang]); file per_lang = 100",
         checks=[("eq", "per_lang", 100), ("gold", "per_language/en/gold", "massive_raw", 0, 100)]),
    dict(f="bench_turbovec.json", suite="massive", cls="none", ev="bench_turbovec.py (PREREG_TURBOVEC corpus); regex hit is the word 'massive' in a passage"),
]


def _get(d, path):
    for p in path.split("/"):
        d = d[p]
    return d


def audit_discover():
    """Every results/*.json matching AUDIT_RE (router_* excluded) must be in AUDIT; every non-optional AUDIT file must exist."""
    import glob
    known = {e["f"] for e in AUDIT}
    hits = set()
    for p in sorted(glob.glob(os.path.join(R, "*.json"))):
        f = os.path.basename(p)
        if f.startswith("router_"):
            continue
        if AUDIT_RE.search(open(p, encoding="utf-8", errors="replace").read()):
            hits.add(f)
    new = sorted(hits - known)
    assert not new, "result files mention banking77/clinc/massive but are not in the audit (classify them in AUDIT): %s" % new
    missing = [e["f"] for e in AUDIT if not e.get("optional") and not os.path.exists(os.path.join(R, e["f"]))]
    assert not missing, "audited files missing: %s" % missing
    return sorted(hits)


def _clm_samples():
    st = jload("clm_template_samples.json")["samples"]["banking77"]["state_text"]
    m = re.match(r"message: (.*?)\n\n", st, re.S)
    assert m, st[:80]
    return [m.group(1)]


def audit_suite(s, ctx, excluded):
    """Enforce the audit for suite s: test files inside the covered spans and their recorded checks pass; text-only files
    excluded by text match. Returns (excluded, rows, newly_excluded)."""
    import numpy as np
    rows, added = [], set()
    by = {}
    for i, t in enumerate(ctx["target_texts"]):
        by.setdefault(norm(t), []).append(i)
    for e in [x for x in AUDIT if x["suite"] == s]:
        p = os.path.join(R, e["f"])
        if e.get("optional") and not os.path.exists(p):
            rows.append({"file": e["f"], "classification": e["cls"], "status": "absent here (optional)"})
            continue
        d = json.load(open(p))
        if e["cls"] == "test":
            for copy, lo, hi in e["ranges"]:
                clo, chi = ctx["covered"][copy]
                assert clo <= lo and hi <= chi, "%s: %s[%d:%d] is outside the excluded span %s[%d:%d]" % (e["f"], copy, lo, hi, copy, clo, chi)
            for c in e.get("checks", []):
                if c[0] == "eq":
                    assert _get(d, c[1]) == c[2], "%s: %s = %r, expected %r" % (e["f"], c[1], _get(d, c[1]), c[2])
                elif c[0] == "len":
                    assert len(_get(d, c[1])) == c[2], "%s: len(%s) != %d" % (e["f"], c[1], c[2])
                elif c[0] == "gold":
                    assert _get(d, c[1]) == ctx["gold"][c[2]][c[3]:c[4]], "%s: %s differs from %s[%d:%d]" % ((e["f"],) + c[1:])
                elif c[0] == "texts_sha":
                    assert _get(d, c[1]) == sha_texts(ctx["texts"][c[2]][c[3]:c[4]]), "%s: texts differ from %s[%d:%d]" % ((e["f"],) + c[2:])
                elif c[0] == "npz_gold":
                    z = np.load(os.path.join(R, c[1]))
                    assert [int(x) for x in z[c[2]]] == ctx["gold"][c[3]][c[4]:c[5]], "%s: %s differs" % (c[1], c[2])
            how = "inside the excluded span (asserted); %d recorded check(s) passed" % len(e.get("checks", []))
        elif e["cls"] == "text":
            texts = {"clm_samples": _clm_samples}[e["texts"]]()
            hit = set()
            for t in texts:
                m_ = by.get(norm(t), [])
                assert m_, "%s: text not found in the target split: %r" % (e["f"], t)
                hit.update(m_)
            added |= hit - excluded
            how = "%d text(s) matched to %d target row(s); %d not already excluded -> now excluded" % (len(texts), len(hit), len(hit - excluded))
        else:
            how = "not test items; nothing to exclude"
        rows.append({"file": e["f"], "classification": e["cls"], "evidence": e["ev"], "covered": how})
    return excluded | added, rows, sorted(added)


def phase_items(a):
    out = {"prereg": "research/PREREG_ROUTER.md", "seed": SEED, "n_test": N_TEST, "n_val": N_VAL,
           "id_sha_rule": "sha256('\\n'.join(sorted(ids))) over string ids (PREREG_SCALE_V2 clarification 1)",
           "selection_rule": "non-excluded test split indices in file order, random.Random(%d).shuffle, first %d; a fresh "
                             "Random per suite. Validation: the same rule on the validation split (Banking77: train), first %d." % (SEED, N_TEST, N_VAL),
           "misspelled_rule": "bench_retrieval.typo_all(text, random.Random(i)), i = the item's position 0..999 in test_index",
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "suites": {}}
    selftest()
    out["audit_files_matching"] = audit_discover()
    out["audit_rule"] = "PREREG_ROUTER.md clarification 3: every results/*.json matching /banking77|clinc|massive/i is classified; " \
                        "test-using files asserted inside the excluded spans; text-only files excluded by text match"
    for s in SUITES:
        name, cfg = SOURCES[s]
        test, how = load_split(name, cfg, "test")
        tcol = "utt" if s == "massive" else "text"
        excluded, report, id_note, ctx = {"clinc150": excl_clinc, "banking77": excl_banking, "massive": excl_massive}[s](test)
        n_before = len(excluded)
        excluded, audit_rows, added = audit_suite(s, ctx, excluded)
        report.append({"audit": audit_rows, "excluded_before_audit": n_before, "newly_excluded_by_audit": added})
        tidx, n_elig = select_fresh(len(test), excluded)
        assert not set(tidx) & excluded
        ids = [str(test[i]["id"]) for i in tidx] if s == "massive" else [str(i) for i in tidx]
        texts = col(test, tcol)
        if s == "banking77":
            train, how_tr = load_split(name, cfg, "train")
            vidx, _ = select_fresh(len(train), set(), k=N_VAL)
            vsplit, vids = "train", [str(i) for i in vidx]
            vtexts = col(train, tcol)
            pool = {"split": "train", "how": how_tr, "n_rows": len(train), "excluded_validation_indices": sorted(vidx),
                    "n": len(train) - len(vidx), "order": "train file order minus the validation indices"}
        else:
            val, how_v = load_split(name, cfg, "validation")
            vidx, _ = select_fresh(len(val), set(), k=N_VAL)
            vsplit, vtexts = "validation", col(val, tcol)
            vids = [str(val[i]["id"]) for i in vidx] if s == "massive" else [str(i) for i in vidx]
            train, how_tr = load_split(name, cfg, "train")
            pool = {"split": "train", "how": how_tr, "n_rows": len(train), "excluded_validation_indices": [], "n": len(train),
                    "order": "train file order"}
        vskip = set(pool["excluded_validation_indices"])
        pool["texts_sha256"] = sha_texts([t for i, t in enumerate(col(train, tcol)) if i not in vskip])
        if s == "clinc150":
            names = train.features["intent"].names
            n_oos = sum(1 for x in col(train, "intent") if x == names.index("oos"))
            pool["note"] = ("%d train rows are labelled oos: E keeps them as an 'oos' class; ML cannot learn them (oos is not an "
                            "option of the 150-intent question), so ML learns the other %d" % (n_oos, len(train) - n_oos))
        out["suites"][s] = {
            "source": how, "split_rows": len(test), "id_kind": id_note,
            "excluded_count": len(excluded), "eligible_after_exclusion": n_elig, "exclusion_report": report,
            "test_index": tidx, "test_ids": ids, "test_ids_sha256": sha_ids(ids),
            "test_texts_sha256": sha_texts([texts[i] for i in tidx]),
            "validation": {"split": vsplit, "index": vidx, "ids": vids, "ids_sha256": sha_ids(vids),
                           "texts_sha256": sha_texts([vtexts[i] for i in vidx])},
            "pool": pool}
        print("%-9s rows %d  excluded %d (audit added %d: %s)  eligible %d  test %d  pool %d  sha256 %s" % (
            s, len(test), len(excluded), len(added), added, n_elig, len(ids), pool["n"], sha_ids(ids)), flush=True)
    if os.path.exists(ITEMS) and not a.force:
        old = json.load(open(ITEMS))
        same = all(old["suites"][s]["test_ids_sha256"] == out["suites"][s]["test_ids_sha256"] for s in SUITES)
        print("router_items.json exists; rebuilt ids %s it. Not overwritten (pass --force)." % ("MATCH" if same else "DIFFER FROM"))
        return
    json.dump(out, open(ITEMS, "w"), indent=1)
    print("wrote", ITEMS)


# ----------------------------------------------------------------------------------------------- phase: run
def suite_data(s, items):
    """Test items, question and labelled pool for suite s, in the earlier runs' formats; texts verified against
    router_items.json."""
    name, cfg = SOURCES[s]
    I = items["suites"][s]
    test, _ = load_split(name, cfg, "test")
    train, _ = load_split(name, cfg, "train")
    if s == "clinc150":
        # bench_clinc.py / check_oos_product.py: options = the 150 in-scope names, '_' -> ' ', ClassLabel order, no descriptions
        names = test.features["intent"].names
        oos = names.index("oos")
        options = [n.replace("_", " ") for i, n in enumerate(names) if i != oos]
        crit, key, ins, tcol, lcol = {l: None for l in options}, "utterance", INS_CLINC, "text", "intent"
        lab = lambda y: "oos" if y == oos else names[y].replace("_", " ")
    elif s == "banking77":
        # bench.py build_suites: options = sorted label names, '_' -> ' ', no descriptions; state key `message`
        names = test.features["label"].names
        options = [x.replace("_", " ") for x in sorted(names)]
        crit, key, ins, tcol, lcol = {l: None for l in options}, "message", INS_BANKING, "text", "label"
        lab = lambda y: names[y].replace("_", " ")
    else:
        # bench_massive.py build: option key = raw intent name, description = name with '_' -> ' ' and '.' -> ': ';
        # all 60 intents here (sorted, as bench_massive's `labels`) instead of gold + 19 distractors
        names = test.features["intent"].names
        options = sorted(names)
        crit, key, ins, tcol, lcol = {k: k.replace("_", " ").replace(".", ": ") for k in options}, "utterance", INS_MASSIVE, "utt", "intent"
        lab = lambda y: names[y]
    texts, ys = col(test, tcol), col(test, lcol)
    tidx = I["test_index"]
    tt = [texts[i] for i in tidx]
    assert sha_texts(tt) == I["test_texts_sha256"], "%s: test texts differ from router_items.json" % s
    skip = set(I["pool"]["excluded_validation_indices"])
    ptexts, pys = col(train, tcol), col(train, lcol)
    pool = [(t, lab(y)) for i, (t, y) in enumerate(zip(ptexts, pys)) if i not in skip]
    assert sha_texts([t for t, _ in pool]) == I["pool"]["texts_sha256"], "%s: pool differs from router_items.json" % s
    return {"ids": I["test_ids"], "texts": tt, "gold": [lab(ys[i]) for i in tidx], "options": options, "state_key": key,
            "q": {"intent": {"type": "choice", "instructions": ins, "criteria": crit}}, "pool": pool}


def typo_fn():
    try:
        from bench_retrieval import typo_all
        return typo_all, "research/bench_retrieval.typo_all (imported)"
    except Exception as e:  # noqa: BLE001  (its module imports rank_bm25 / datasets / mahabodi)
        def typo_all(q, rng):  # verbatim copy of research/bench_retrieval.py typo_all
            def f(m):
                w = m.group(0)
                if len(w) < 5:
                    return w
                i = rng.randrange(1, len(w) - 1)
                return w[:i] + w[i + 1:]
            return re.sub(r"[A-Za-z]+", f, q)
        return typo_all, "verbatim copy of bench_retrieval.typo_all (import failed: %s)" % str(e)[:120]


def gate_opts(s):
    if s != "clinc150":
        return {}
    th = jload("bench_clinc_oos.json")["thresholds"]["C"]
    assert (th["tau"], th["s"]) == (0.5, 0.3), th  # the W11 thresholds named in the pre-registration
    return {"oos_min_similarity": th["s"], "oos_below_probability": th["tau"]}


def summarize_pass(s, d, pred, lat_ms, wall, extra):
    gold = d["gold"]
    corr = [p == g for p, g in zip(pred, gold)]
    n = len(corr)
    h = n // 2
    r = {"n": n, "accuracy": round(sum(corr) / n, 4), "accuracy_ci95": wilson(sum(corr), n),
         "latency_ms": {"first": round(lat_ms[0], 3), "p50": round(pct(lat_ms, 50), 3), "p95": round(pct(lat_ms, 95), 3),
                        "second_half_p50": round(pct(lat_ms[h:], 50), 3), "second_half_p95": round(pct(lat_ms[h:], 95), 3),
                        "second_half_items": "%d..%d" % (h, n - 1), "per_item": [round(x, 3) for x in lat_ms]},
         "pass_wall_seconds": round(wall, 3), "decisions_per_hour": round(n / wall * 3600, 1) if wall > 0 else None,
         "peak_rss_mb_after_pass": rss_mb(), "ids": d["ids"], "gold": gold, "pred": pred, "correct": corr}
    if s == "clinc150":
        ins = [c for c, g in zip(corr, gold) if g != "oos"]
        n_oos = sum(g == "oos" for g in gold)
        tp = sum(p == "oos" and g == "oos" for p, g in zip(pred, gold))
        flagged = sum(p == "oos" for p in pred)
        r.update({"in_scope_accuracy": round(sum(ins) / len(ins), 4), "in_scope_ci95": wilson(sum(ins), len(ins)), "n_oos_gold": n_oos,
                  "oos_recall": round(tp / max(1, n_oos), 4), "oos_recall_ci95": wilson(tp, n_oos), "oos_flagged": flagged,
                  "oos_precision": round(tp / max(1, flagged), 4)})
    r.update(extra)
    return r


def phase_run(a):
    if a.arm in LLM_ARMS:
        raise SystemExit("Arm %s is an LLM arm: deferred by PREREG_ROUTER.md clarification 1 (no API key; nothing about it may be "
                         "chosen now). TODO when approved: tool-use enum prompt, prompt caching, token logging, same items "
                         "and per-item files, per the pre-registration's Arms / Budget sections." % a.arm)
    from mahabodi import Bodi
    from provenance import provenance
    items = json.load(open(ITEMS))
    typo_all, typo_src = typo_fn()
    out_path = os.path.join(R, "router_%s.json" % a.arm)
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    resumed = bool(res.get("suites"))
    proc = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(), "resumed_from_checkpoint": resumed,
            "provenance": provenance(), "threads": a.threads}
    res.setdefault("processes", []).append(proc)
    res.update({"arm": a.arm, "prereg": "research/PREREG_ROUTER.md (clarification 1: LLM arms pending)",
                "items_sha256": {s: items["suites"][s]["test_ids_sha256"] for s in SUITES}, "typo_source": typo_src,
                "rss_note": "peak RSS of the whole Python process (ru_maxrss), datasets and pool texts included",
                "latency_rule": "per item, batch 1, scored pass in item order, no warm-up (PREREG_SCALE_V2 clarification 3d); "
                                "passes run clinc150, banking77, massive, each clean then misspelled, in one process unless resumed"})
    res.setdefault("suites", {})
    minilm = os.path.join(ROOT, "models", "minilm")
    src = json.load(open(os.path.join(minilm, "embedder_config.json"))).get("source")
    assert src == "sentence-transformers/all-MiniLM-L6-v2", "models/minilm is %r" % src
    b = Bodi()
    t0 = time.time()
    if a.arm in ("M", "ML"):
        b.load_laya(os.path.join(ROOT, "models", "laya-v2"), intra_threads=a.threads)
    b.load_embedder(minilm, intra_threads=a.threads)  # E: the kNN embedder; M/ML: the OOS gate's similarity (and ML's memory)
    proc["load_seconds"] = round(time.time() - t0, 2)
    proc["decide_defaults"] = b.decide_defaults() if a.arm != "E" else None
    if a.arm == "M":
        res["embedder"] = "models/minilm (all-MiniLM-L6-v2): used only by the CLINC150 OOS gate; experience memory is empty"
    elif a.arm == "E":
        res["embedder"] = ("MahaBodi's own all-MiniLM-L6-v2 ONNX export (models/minilm, embed_text: mean pooling + L2 norm; parity with "
                           "sentence-transformers shown in results/check_oos_embedder.json, max |delta| ~1e-6)")
    import numpy as np
    for s in SUITES:
        todo = [v for v in VARIANTS if v not in res["suites"].get(s, {})]
        if not todo:
            continue
        d = suite_data(s, items)
        q, qid, key = d["q"], "intent", d["state_key"]
        gate = gate_opts(s)
        S = res["suites"].setdefault(s, {})
        S["options"] = d["options"]
        S["question"] = q
        S["n_pool"] = len(d["pool"])
        if a.arm == "M":
            opts = dict(cache=False, experience_k=0, round_probabilities=False, **gate)  # = bench_clinc.py arm C (+ W11 gate)
        elif a.arm == "ML":
            b.forget()
            learnable = [(t, y) for t, y in d["pool"] if y in q["intent"]["criteria"]]
            t1 = time.time()
            learned = b.learn([{key: t} for t, _ in learnable], q, [{qid: y} for _, y in learnable], calibrate=CALIBRATE)
            S.setdefault("learn", []).append({"pid": os.getpid(), "seconds": round(time.time() - t1, 2), "n_learned": len(learnable),
                                              "n_pool_not_learnable": len(d["pool"]) - len(learnable), "result": learned,
                                              "rss_mb_after": rss_mb()})
            opts = dict(cache=False, round_probabilities=False, **gate)  # product defaults (bench_fresh_experience.py `calibrated`)
        else:
            plab = [y for _, y in d["pool"]]
            t1 = time.time()
            P = []
            for i in range(0, len(d["pool"]), 256):
                P.extend(b.embed_text([t for t, _ in d["pool"][i:i + 256]]))
            P = np.asarray(P, dtype=np.float32)
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            S.setdefault("pool_embed", []).append({"pid": os.getpid(), "seconds": round(time.time() - t1, 2), "n": len(P)})
            order = {y: i for i, y in enumerate(sorted(set(plab)))}
            opts = None
        S["decide_options"] = opts
        for v in todo:
            texts = d["texts"] if v == "clean" else [typo_all(t, random.Random(i)) for i, t in enumerate(d["texts"])]
            pred, lat, extra_rows = [], [], []
            tp0 = time.perf_counter()
            for t in texts:
                if a.arm == "E":
                    t1 = time.perf_counter()
                    u = np.asarray(b.embed_text([t])[0], dtype=np.float32)
                    u /= np.linalg.norm(u)
                    sims = P @ u
                    top = np.argpartition(-sims, K_E)[:K_E]
                    top = top[np.argsort(-sims[top], kind="stable")]
                    votes, ssum = {}, {}
                    for j in top:
                        votes[plab[j]] = votes.get(plab[j], 0) + 1
                        ssum[plab[j]] = ssum.get(plab[j], 0.0) + float(sims[j])
                    p = max(votes, key=lambda y: (votes[y], ssum[y], -order[y]))  # majority, then summed similarity, then label order
                    lat.append((time.perf_counter() - t1) * 1000)
                    extra_rows.append({"nn_pool_index": [int(j) for j in top], "nn_sim": [round(float(sims[j]), 5) for j in top]})
                else:
                    t1 = time.perf_counter()
                    ans = b.decide({key: t}, q, **opts)["answers"][qid]
                    lat.append((time.perf_counter() - t1) * 1000)
                    pr = ans.get("probabilities") or {}
                    top1 = max(pr, key=pr.get) if pr else ans.get("choice")
                    meta = ans.get("bodi", {})
                    flag = bool(meta.get("out_of_scope")) if gate else False
                    p = "oos" if flag else top1
                    ex = meta.get("experience", {}) if isinstance(meta.get("experience"), dict) else {}
                    extra_rows.append({"top1": top1, "top1_prob": pr.get(top1) if top1 is not None else None,
                                       "max_sim": meta.get("max_similarity"), "oos_flag": flag, "choice": ans.get("choice"),
                                       "memory_first": bool(ex.get("memory_first")), "strategy": meta.get("strategy")})
                pred.append(p)
            wall = time.perf_counter() - tp0
            per = {k: [r[k] for r in extra_rows] for k in extra_rows[0]}
            S[v] = summarize_pass(s, d, pred, lat, wall, {"per_item": per, "pid": os.getpid(), "resumed_process": resumed,
                                                         "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                         "texts_sha256": sha_texts(texts)})
            print(a.arm, s, v, {k: S[v][k] for k in ("accuracy", "accuracy_ci95", "decisions_per_hour", "peak_rss_mb_after_pass")},
                  {k: S[v]["latency_ms"][k] for k in ("first", "p50", "p95", "second_half_p50", "second_half_p95")},
                  {k: S[v][k] for k in ("in_scope_accuracy", "oos_recall") if k in S[v]}, flush=True)
            res["peak_rss_mb"] = rss_mb()
            json.dump(res, open(out_path + ".tmp", "w"))
            os.replace(out_path + ".tmp", out_path)  # checkpoint: one suite x variant
    print("wrote", out_path)


# ----------------------------------------------------------------------------------------------- phase: score
def phase_score(a):
    arms = {}
    for arm in ("M", "ML", "E"):
        p = os.path.join(R, "router_%s.json" % arm)
        if os.path.exists(p):
            arms[arm] = json.load(open(p))
    out = {"label": "LLM arms pending (PREREG_ROUTER.md clarification 1): secondary comparisons only; no router claim against any LLM",
           "comparisons": ["M vs E", "ML vs E"], "pooled": False,
           "notes": ["MASSIVE here uses 60 options and laya-v2 English; not comparable to the published MASSIVE 0.405 "
                     "(laya-multilingual, 20 options)",
                     "CLINC labelled condition: ML learns in-scope rows only (learn rejects non-option labels) while E learns "
                     "out_of_scope as a class, which favours E on OOS; OOS recall is reported separately"],
           "suites": {}}
    for note in out["notes"]:
        print("NOTE:", note)
    for s in SUITES:
        for v in VARIANTS:
            cell = {}
            have = {arm: r["suites"][s][v] for arm, r in arms.items() if v in r.get("suites", {}).get(s, {})}
            for arm, x in have.items():
                cell[arm] = {k: x.get(k) for k in ("n", "accuracy", "accuracy_ci95", "in_scope_accuracy", "in_scope_ci95", "oos_recall",
                                                   "oos_recall_ci95", "oos_precision", "decisions_per_hour", "peak_rss_mb_after_pass")}
                cell[arm]["latency_ms"] = {k: x["latency_ms"][k] for k in ("first", "p50", "p95", "second_half_p50", "second_half_p95")}
                cell[arm]["resumed_process"] = x.get("resumed_process")
            for A_ in ("M", "ML"):
                if A_ in have and "E" in have:
                    xa, xe = have[A_], have["E"]
                    assert xa["ids"] == xe["ids"] and xa["gold"] == xe["gold"], "%s/%s: items differ between %s and E" % (s, v, A_)
                    cell["%s_vs_E" % A_] = {"mcnemar": mcnemar(xa["correct"], xe["correct"]),
                                            "paired_difference": newcombe_paired(xa["correct"], xe["correct"]),
                                            "label": "LLM arms pending"}
            out["suites"].setdefault(s, {})[v] = cell
            for k, c in cell.items():
                if k.endswith("_vs_E"):
                    print("%-9s %-10s %-6s diff %+.4f %s  McNemar %d/%d p=%s  [LLM arms pending]" % (
                        s, v, k, c["paired_difference"]["diff"], c["paired_difference"]["ci95"], c["mcnemar"]["a_only"],
                        c["mcnemar"]["b_only"], c["mcnemar"]["p"]))
                else:
                    print("%-9s %-10s %-6s acc %.4f %s  p50 %.1f ms" % (s, v, k, c["accuracy"], c["accuracy_ci95"], c["latency_ms"]["p50"]))
    json.dump(out, open(os.path.join(R, "router_score.json"), "w"), indent=1)
    print("wrote", os.path.join(R, "router_score.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase", required=True, choices=["selftest", "items", "run", "score"])
    ap.add_argument("--arm", choices=["M", "ML", "E", *LLM_ARMS])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="items: overwrite an existing router_items.json")
    a = ap.parse_args()
    if a.phase == "selftest":
        selftest()
    elif a.phase == "items":
        phase_items(a)
    elif a.phase == "run":
        if not a.arm:
            ap.error("--phase run needs --arm")
        phase_run(a)
    else:
        phase_score(a)


if __name__ == "__main__":
    main()
