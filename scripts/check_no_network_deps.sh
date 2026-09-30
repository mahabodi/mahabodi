#!/usr/bin/env bash
# mahabodi-core must never pull in fastmemory's CLI/server/telemetry or any network crate (fastmemory's CLI sends
# license telemetry). Fails if any appear in its dependency graph, for every feature set that ships.
set -u
cd "$(dirname "$0")/.."
bad='^(reqwest|hyper|axum|tokio|sysinfo|dotenv|ureq|isahc|surf) |fastmemory feature "(cli|telemetry|default)"'
fail=0
for feats in "" "--features laya" "--features postgres" "--features laya,postgres"; do
  # postgres uses tokio-postgres (client to the user's own DB, not telemetry): only fastmemory's features and
  # HTTP/telemetry crates are banned there
  pat="$bad"; [[ "$feats" == *postgres* ]] && pat='^(reqwest|hyper|axum|sysinfo|dotenv|ureq|isahc|surf) |fastmemory feature "(cli|telemetry|default)"'
  hits=$(cargo tree -p mahabodi-core $feats -e normal,features --prefix none 2>/dev/null | sort -u | grep -E "$pat")
  if [ -n "$hits" ]; then echo "FAIL [$feats]:"; echo "$hits" | head; fail=1; else echo "ok   [$feats]"; fi
done
exit $fail
