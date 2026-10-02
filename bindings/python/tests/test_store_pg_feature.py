"""Non-skipping: proves the `postgres` feature is compiled into this build, with no server needed.
A build without the feature answers "unknown method 'store_open'"; a build with it fails to connect
to port 1 with a store/postgres error. Runs regardless of MAHABODI_TEST_PG_DSN."""
import pytest

from mahabodi import Bodi


def test_store_feature_compiled_in():
    b = Bodi()
    with pytest.raises(Exception) as ei:
        b.store_open("host=127.0.0.1 port=1 user=x dbname=x connect_timeout=1", "py_unreach")
    msg = str(ei.value)
    assert "unknown method" not in msg, "store not compiled into this build: " + msg
    assert any(w in msg.lower() for w in ("postgres", "store", "connect")), msg
