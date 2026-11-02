#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

# The stock image ships the in-memory `memory` catalog, which is all the
# battery needs. Trino takes a while to boot; poll the info endpoint
# until it is no longer starting.
docker run -d --name db -p 8080:8080 trinodb/trino:476
# Readiness needs BOTH: /v1/info starting:false (unauthenticated; the
# dispatcher rejects real queries with "Trino server is still
# initializing" until then) and a statement that runs to completion
# (authenticated — Trino's "insecure" auth 401s without X-Trino-User).
# A fresh query's first page is QUEUED with only id/nextUri; scheduling
# failures like "No nodes available to run query" only appear on later
# pages, so the probe must follow nextUri until the query finishes.
trino_ready() {
  curl -sf http://localhost:8080/v1/info | grep -q '"starting":false' || return 1
  local resp next
  resp=$(curl -sf -X POST -H "X-Trino-User: test" -d "SELECT 1" \
    http://localhost:8080/v1/statement) || return 1
  for _ in $(seq 1 60); do
    grep -q '"error"' <<<"$resp" && return 1
    next=$(jq -r '.nextUri // empty' <<<"$resp")
    [ -z "$next" ] && return 0
    sleep 1
    resp=$(curl -sf -H "X-Trino-User: test" "$next") || return 1
  done
  return 1
}
wait_for trino trino_ready
# The driver defaults to HTTPS; SSL=false selects plain HTTP.
export_uri trino "trino://test@localhost:8080/memory/default?SSL=false"
