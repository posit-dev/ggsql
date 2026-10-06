#!/usr/bin/env bash
# Druid leg: the vendor-supported docker-compose cluster — zookeeper,
# postgres metadata, and one container per Druid service at
# micro-quickstart sizing (~6 GiB total). (The 37.x image is distroless:
# busybox + static bash, no /usr/bin/env, no perl — the classic
# all-in-one supervise quickstart cannot run inside it.) The datasource
# is created by an MSQ ingestion job submitted to the router (8888) and
# polled to completion; the Foundry druid ADBC driver (installed with
# --pre by the workflow step — it is prerelease-only) then talks SQL to
# the broker (8082). DruidDialect requires_cache, so the battery runs
# through ggsql's automatic sqlite cache wrap.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"

druid_dir=$(mktemp -d)
curl -sSL -o "$druid_dir/docker-compose.yml" \
  https://raw.githubusercontent.com/apache/druid/37.0.0/distribution/docker/docker-compose.yml
curl -sSL -o "$druid_dir/environment" \
  https://raw.githubusercontent.com/apache/druid/37.0.0/distribution/docker/environment
docker compose -f "$druid_dir/docker-compose.yml" up -d
wait_for druid-broker curl -sf http://localhost:8082/status/health
wait_for druid-overlord curl -sf http://localhost:8081/status/health
wait_for druid-mm curl -sf http://localhost:8091/status/health
wait_for druid-router curl -sf http://localhost:8888/status/health

# MSQ INSERT: every Druid datasource needs a __time column; one shared
# timestamp suffices. PARTITIONED BY ALL puts everything in one segment.
# The query must go to the dedicated MSQ endpoint /druid/v2/sql/task:
# the regular /druid/v2/sql endpoint plans with the native engine, which
# refuses INSERT ("consider using MSQ"), while an explicit
# "engine": "msq-task" context is rejected there ("Unsupported engine").
# The row data comes from the shared fixture CSV (kept in lockstep with
# the battery's ROWS by the fixture_csv_is_current test) as an inline
# EXTERN source: a UNION ALL of literal SELECTs is NOT plannable by MSQ
# ("Union operation is only supported between regular tables") because
# each literal select plans as an inline datasource. The CSV header row
# names the columns, but the planner still requires an explicit signature
# as EXTERN's third argument ("EXTERN requires either a [signature] value
# or an EXTEND clause"). All extern columns are strings, so CAST the
# numerics in the SELECT.
payload=$(FIXTURE_CSV="$repo_root/.github/scripts/live/ggsql_live_test.csv" python3 - <<'EOF'
import json, os, pathlib
csv_text = pathlib.Path(os.environ["FIXTURE_CSV"]).read_text().strip()
inline = json.dumps({"type": "inline", "data": csv_text})
csv_fmt = json.dumps({"type": "csv", "findColumnsFromHeader": True})
sig = json.dumps([
    {"name": "id", "type": "STRING"},
    {"name": "val", "type": "STRING"},
    {"name": "grp", "type": "STRING"},
    {"name": "day", "type": "STRING"},
    {"name": "mixed Case", "type": "STRING"},
])
query = f"""
INSERT INTO ggsql_live_test
SELECT TIMESTAMP '2020-01-01 00:00:00' AS __time,
       CAST(id AS BIGINT) AS id,
       CAST(val AS DOUBLE) AS val,
       grp,
       CAST("day" AS DATE) AS "day",
       CAST("mixed Case" AS DOUBLE) AS "mixed Case"
FROM TABLE(EXTERN('{inline}', '{csv_fmt}', '{sig}'))
PARTITIONED BY ALL
"""
print(json.dumps({"query": query}))
EOF
)
# No curl -f here: on an HTTP error the response body carries Druid's
# actual error message, which is the only useful diagnostic.
resp=$(curl -sS -X POST -H 'Content-Type: application/json' \
  -d "$payload" http://localhost:8888/druid/v2/sql/task)
task_id=$(printf '%s' "$resp" | sed -n 's/.*"taskId":"\([^"]*\)".*/\1/p')
if [ -z "$task_id" ]; then
  echo "MSQ ingest submission failed; response:"
  echo "$resp"
  docker compose -f "$druid_dir/docker-compose.yml" logs || true
  exit 1
fi
echo "MSQ ingest submitted; response: $resp"
# A plan-time failure returns synchronously ("state":"FAILED") and the
# task never reaches the overlord — polling would only ever report
# "Cannot find any task". Bail out immediately with the real error.
case "$resp" in
  *'"state":"FAILED"'*)
    echo "MSQ ingest rejected at plan time; response:"
    echo "$resp"
    docker compose -f "$druid_dir/docker-compose.yml" logs || true
    exit 1
    ;;
esac
# MSQ jobs are asynchronous; poll the task until it succeeds or fails.
# The /status endpoint reports "status":"SUCCESS|FAILED" (statusCode only
# appears in the full report, not here). If the submission returned a
# bare query id, the overlord knows it as query-<id>, so retry prefixed.
ok=""
for i in $(seq 1 60); do
  status=$(curl -sS "http://localhost:8888/druid/indexer/v1/task/$task_id/status" || true)
  echo "poll $i: $status"
  if grep -q "Cannot find any task" <<<"$status"; then
    case "$task_id" in
      query-*) : ;;
      *) task_id="query-$task_id"; continue ;;
    esac
  fi
  case "$status" in
    *'"status":"SUCCESS"'*) ok=1; break ;;
    *'"status":"FAILED"'* | *'"statusCode":"FAILED"'*)
      echo "MSQ ingest task $task_id FAILED; status payload:"
      echo "$status"
      docker compose -f "$druid_dir/docker-compose.yml" logs || true
      exit 1
      ;;
  esac
  sleep 5
done
if [ -z "$ok" ]; then
  echo "MSQ ingest task $task_id never reached SUCCESS; last status:"
  echo "$status"
  docker compose -f "$druid_dir/docker-compose.yml" logs || true
  exit 1
fi
# Segment handoff lags task success; wait until the broker serves rows.
# No curl -f: a 400 body carries the broker's actual error message.
cnt=""
for _ in $(seq 1 30); do
  body=$(curl -sS -X POST -H 'Content-Type: application/json' \
    -d '{"query":"SELECT COUNT(*) AS c FROM ggsql_live_test"}' \
    http://localhost:8082/druid/v2/sql || true)
  # The || true matters: with pipefail, grep finding no "c":N in an
  # error/empty body would kill the script silently via set -e.
  cnt=$(printf '%s' "$body" | grep -o '"c":[0-9]*' | cut -d: -f2 || true)
  [ "$cnt" = "8" ] && break
  sleep 5
done
if [ "$cnt" != "8" ]; then
  echo "datasource never became queryable (count=$cnt); last broker response:"
  echo "$body"
  docker compose -f "$druid_dir/docker-compose.yml" logs || true
  exit 1
fi
# tls=false: the Foundry driver defaults to https for the broker URL
# (its README: "druid://localhost:8888?tls=false"); the compose cluster
# serves plain HTTP.
export_uri druid "druid://localhost:8082?tls=false"
