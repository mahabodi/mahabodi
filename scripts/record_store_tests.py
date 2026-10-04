"""Run the Python store tests verbosely and record a checkable result file.

Fails (exit 1) unless every expected test PASSES — skips are failures here, so a missing server,
password or feature can never record a hollow pass (the trap that bit the first Linux run).

    MAHABODI_TEST_PG_DSN=... python scripts/record_store_tests.py --out research/results/store_fork_linux.json
"""
import argparse, json, os, platform, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = ["bindings/python/tests/test_store_pg.py", "bindings/python/tests/test_store_pg_feature.py"]
EXPECTED = ["test_round_trip_matches_in_process", "test_fork_child_opens_its_own_connections",
            "test_store_feature_compiled_in"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dsn = os.environ.get("MAHABODI_TEST_PG_DSN", "")
    assert dsn, "MAHABODI_TEST_PG_DSN required"
    r = subprocess.run([sys.executable, "-m", "pytest", "-v", *TESTS], cwd=ROOT, capture_output=True, text=True)
    outcomes = dict(re.findall(r"::(\w+)\s+(PASSED|FAILED|SKIPPED|ERROR)", r.stdout))
    import psycopg
    with psycopg.connect(dsn) as c:
        pgver = c.execute("SHOW server_version").fetchone()[0]
    rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ok = all(outcomes.get(t) == "PASSED" for t in EXPECTED) and len(outcomes) == len(EXPECTED)
    out = {"what": "python store tests (round trip, fork safety, feature compiled in); skips count as failure",
           "outcomes": outcomes, "expected": EXPECTED, "all_passed": ok,
           "host": platform.node(), "os": platform.platform(), "machine": platform.machine(),
           "postgres_server_version": pgver, "git_rev": rev,
           "dsn_fields": "host/port/dbname only: " + " ".join(kv for kv in dsn.split() if not kv.startswith("password"))}
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "what"}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
