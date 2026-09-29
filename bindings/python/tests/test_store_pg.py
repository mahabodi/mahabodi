"""PostgreSQL store through the Python binding (built with --features postgres). Skipped unless
MAHABODI_TEST_PG_DSN is set. Covers the round trip and fork safety (a forked child must open its own connections)."""
import os
import pytest

from mahabodi import Bodi

DSN = os.environ.get("MAHABODI_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="MAHABODI_TEST_PG_DSN not set")

DOCS = [
    {"text": "# Refund policy\n\nRefunds take five days. Escalate after two days to the billing team.", "source": "kb"},
    {"text": "# Billing\n\nInvoices are sent monthly. The billing team answers payment questions.", "source": "kb2"},
]


def _store(ns):
    b = Bodi()
    try:
        b.store_open(DSN, ns)
    except Exception as e:  # a wheel built without the postgres feature
        pytest.skip("store not available in this build: %s" % e)
    b.store_ingest_batch(DOCS)
    b.store_build_index()
    b.store_ensure_density()
    return b


def test_round_trip_matches_in_process():
    ns = "py%d" % os.getpid()
    b = _store(ns)
    b.ingest_batch(DOCS)
    for q in ("refunds", "billing team", "invoice"):
        a, s = b.query(q, k=5), b.store_query(q, k=5)
        assert a["stage"] == s["stage"]
        assert [h["id"] for h in a["hits"]] == [h["id"] for h in s["hits"]]


def test_fork_child_opens_its_own_connections():
    ns = "pyf%d" % os.getpid()
    b = _store(ns)
    before = b.store_query("refunds", k=3)["hits"][0]["id"]
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            got = b.store_query("refunds", k=3)["hits"][0]["id"]
            os.write(w, got.encode())
        finally:
            os._exit(0)
    os.close(w)
    child = os.read(r, 4096).decode()
    os.waitpid(pid, 0)
    assert child == before
    assert b.store_query("refunds", k=3)["hits"][0]["id"] == before  # the parent's pool still works after the fork
