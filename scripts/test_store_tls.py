"""0.2.0 blocker: the store's TLS handshake, both sides, against a scratch PostgreSQL with a self-signed CA.

Creates its own cluster (never the store's pgdata) on --port with ssl=on and a server certificate (CN=localhost,
SAN DNS:localhost) signed by a throwaway CA; pg_hba has hostssl lines only, so a client that silently fell back to
plaintext is told "no pg_hba.conf entry" — distinguishable from a certificate error.

Cases, all through the mahabodi binding (store_open + store_stats):
  1. sslmode=require sslrootcert=<ca>        -> must succeed (rustls verifies cert + hostname: libpq verify-full).
  2. sslmode=require sslrootcert=<wrong ca>  -> must fail with a certificate error.
  3. sslmode=require, no sslrootcert         -> must fail (self-signed CA is not in the Mozilla roots).
  4. sslmode=prefer  sslrootcert=<wrong ca>  -> recorded: a certificate error means no plaintext fallback;
                                                "no pg_hba.conf entry" would mean it fell back to plaintext.
  5. sslmode=disable                         -> must fail with "no pg_hba.conf entry" (setup sanity: hostssl only).

Writes research/results/store_tls_test.json.

    python scripts/test_store_tls.py --pgbin ~/pgenv/bin --dir ~/tls-test-cluster --port 5544
"""
import argparse, json, os, shutil, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research"))
from provenance import provenance  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "results")


def sh(*cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def certs(d):
    """A CA, a server cert for localhost signed by it, and a second (wrong) CA."""
    for name in ("ca", "wrong-ca"):
        sh("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2",
           "-keyout", f"{d}/{name}.key", "-out", f"{d}/{name}.crt", "-subj", f"/CN=mahabodi-{name}")
    sh("openssl", "req", "-newkey", "rsa:2048", "-nodes",
       "-keyout", f"{d}/server.key", "-out", f"{d}/server.csr", "-subj", "/CN=localhost")
    ext = f"{d}/san.cnf"
    open(ext, "w").write("subjectAltName=DNS:localhost,IP:127.0.0.1\n")
    # -set_serial instead of -CAcreateserial: LibreSSL derives the .srl path by truncating the CA path at the
    # first ".", which lands outside the scratch dir when a parent directory name contains one
    sh("openssl", "x509", "-req", "-in", f"{d}/server.csr", "-CA", f"{d}/ca.crt", "-CAkey", f"{d}/ca.key",
       "-set_serial", "7391", "-days", "2", "-out", f"{d}/server.crt", "-extfile", ext)
    os.chmod(f"{d}/server.key", 0o600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgbin", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--port", type=int, default=5544)
    a = ap.parse_args()
    pgbin, d = os.path.expanduser(a.pgbin), os.path.expanduser(a.dir)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    certs(d)
    data = os.path.join(d, "data")
    sh(os.path.join(pgbin, "initdb"), "-D", data, "-A", "trust", "-U", "postgres")
    with open(os.path.join(data, "postgresql.conf"), "a") as f:
        f.write(f"\nlisten_addresses='localhost'\nport={a.port}\nssl=on\n"
                f"ssl_cert_file='{d}/server.crt'\nssl_key_file='{d}/server.key'\n"
                f"unix_socket_directories='{d}'\n")
    open(os.path.join(data, "pg_hba.conf"), "w").write(
        f"local   all postgres                 trust\n"
        f"hostssl all postgres 127.0.0.1/32    trust\n"
        f"hostssl all postgres ::1/128         trust\n")
    sh(os.path.join(pgbin, "pg_ctl"), "-D", data, "-l", os.path.join(d, "pg.log"), "-w", "start")
    out = {"what": "store TLS handshake test (scratch cluster, self-signed CA)", "port": a.port,
           "provenance": provenance(), "cases": {}}
    code = 0
    try:
        from mahabodi import Bodi
        base = f"host=localhost port={a.port} user=postgres dbname=postgres"
        cases = {
            "require_right_ca": (f"{base} sslmode=require sslrootcert={d}/ca.crt", True, None),
            "require_wrong_ca": (f"{base} sslmode=require sslrootcert={d}/wrong-ca.crt", False, "certificate"),
            "require_mozilla_roots": (f"{base} sslmode=require", False, "certificate"),
            "prefer_wrong_ca": (f"{base} sslmode=prefer sslrootcert={d}/wrong-ca.crt", None, None),
            "disable": (f"{base} sslmode=disable", False, "pg_hba"),
        }
        for name, (dsn, want_ok, want_err) in cases.items():
            b = Bodi()
            try:
                b.call("store_open", dsn=dsn, namespace="tlstest", vector_type="vector", create=True)
                st = b.call("store_stats")
                res = {"ok": True, "stats_passages": st.get("passages")}
            except Exception as e:  # noqa: BLE001
                res = {"ok": False, "error": str(e)[:300]}
            res["expected_ok"] = want_ok
            if want_ok is True:
                res["pass"] = res["ok"]
            elif want_ok is False:
                res["pass"] = (not res["ok"]) and (want_err is None or want_err.lower() in res.get("error", "").lower())
            else:  # recorded, not asserted: classify the prefer behaviour
                res["pass"] = True
                res["reading"] = ("no plaintext fallback (certificate error)" if not res["ok"] and "certificat" in res.get("error", "").lower()
                                  else "FELL BACK TO PLAINTEXT (pg_hba refusal)" if not res["ok"] and "pg_hba" in res.get("error", "")
                                  else "connected (inspect: unexpected)" if res["ok"] else "failed: " + res.get("error", "")[:80])
            out["cases"][name] = res
            print(name, res, flush=True)
            if not res["pass"]:
                code = 1
    finally:
        subprocess.run([os.path.join(pgbin, "pg_ctl"), "-D", data, "-m", "fast", "stop"], capture_output=True)
        if code == 0:
            shutil.rmtree(d, ignore_errors=True)  # keep the dir (certs + pg.log) on failure
    out["all_pass"] = code == 0
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(os.path.join(R, "store_tls_test.json"), "w"), indent=1, default=str)
    print("DONE all_pass", out["all_pass"], flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
