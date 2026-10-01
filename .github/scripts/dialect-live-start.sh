#!/usr/bin/env bash
# Start the container for one live-dialect-test backend and publish its
# connection URI as GGSQL_TEST_URI_<BACKEND> in $GITHUB_ENV. One matrix leg
# of .github/workflows/dialect-live.yml invokes this with its backend name.
#
# Health checks poll until the backend actually accepts connections; on
# timeout the container logs are dumped to make CI failures debuggable.
set -euo pipefail

backend="$1"
uri=""

# wait_for <name> <cmd...>: poll cmd every 5s (up to ~7.5 min) until it
# exits 0.
wait_for() {
  local name="$1"
  shift
  for _ in $(seq 1 90); do
    if "$@" >/dev/null 2>&1; then
      echo "$name is up"
      return 0
    fi
    sleep 5
  done
  echo "$name failed to start"
  docker logs db || true
  exit 1
}

port_open() { (exec 3<>"/dev/tcp/localhost/$1") 2>/dev/null; }

case "$backend" in
  postgres)
    docker run -d --name db \
      -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ggsql \
      -p 5432:5432 postgres:17
    wait_for postgres docker exec db pg_isready -U postgres
    uri="postgres://postgres:postgres@localhost:5432/ggsql"
    ;;
  clickhouse)
    # Passwordless 'default' only authenticates from localhost, but
    # port-mapped connections arrive from the docker bridge IP.
    docker run -d --name db \
      -e CLICKHOUSE_PASSWORD=clickhouse \
      -p 8123:8123 clickhouse/clickhouse-server:25.8
    wait_for clickhouse curl -sf http://localhost:8123/ping
    # The driver speaks HTTP and ignores URL userinfo: credentials go in
    # query params, which ggsql maps to the username/password options.
    uri="clickhouse://localhost:8123?username=default&password=clickhouse"
    ;;
  trino)
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
    uri="trino://test@localhost:8080/memory/default?SSL=false"
    ;;
  mysql)
    docker run -d --name db \
      -e MYSQL_ROOT_PASSWORD=mysql -e MYSQL_DATABASE=ggsql \
      -p 3306:3306 mysql:8.4
    # mysqladmin ping answers over the unix socket before init finishes and
    # TCP is serving; a real TCP query against the ggsql database only
    # succeeds once the entrypoint's init/restart cycle is complete.
    wait_for mysql docker exec db mysql -uroot -pmysql --protocol=TCP -h127.0.0.1 -e "SELECT 1" ggsql
    uri="mysql://root:mysql@localhost:3306/ggsql"
    ;;
  mariadb)
    docker run -d --name db \
      -e MARIADB_ROOT_PASSWORD=mysql -e MARIADB_DATABASE=ggsql \
      -p 3306:3306 mariadb:11
    wait_for mariadb docker exec db mariadb -uroot -pmysql --protocol=TCP -h127.0.0.1 -e "SELECT 1" ggsql
    uri="mariadb://root:mysql@localhost:3306/ggsql"
    ;;
  mssql)
    # Password meets the SQL Server complexity rules (3 of 4 categories).
    docker run -d --name db \
      -e ACCEPT_EULA=Y -e MSSQL_SA_PASSWORD=Ggsql_test1 \
      -p 1433:1433 mcr.microsoft.com/mssql/server:2022-latest
    wait_for mssql docker exec db /opt/mssql-tools18/bin/sqlcmd \
      -C -U sa -P Ggsql_test1 -Q "SELECT 1" -l 2
    # TrustServerCertificate: the container's TLS cert is self-signed.
    uri="mssql://sa:Ggsql_test1@localhost:1433/master?TrustServerCertificate=true"
    ;;
  exasol)
    docker run -d --name db --privileged \
      -p 8563:8563 exasol/docker-db:2026.1.2
    wait_for exasol port_open 8563
    # The port opens long before the database finishes its startup stages
    # (the driver was getting "TLS error: Connection reset by peer" on a
    # fixed sleep). A completed TLS handshake is the earliest reliable
    # readiness signal; the server resets connections until then.
    wait_for exasol-tls bash -c 'echo | timeout 5 openssl s_client -connect localhost:8563 >/dev/null 2>&1'
    uri="exasol://sys:exasol@localhost:8563/?tls=true&validateservercertificate=0"
    ;;
  redshift)
    # Pseudo-leg: the Foundry "redshift" ADBC driver is the PostgreSQL
    # driver, so a stock PostgreSQL container exercises RedshiftDialect
    # end-to-end over a wire-compatible server (ggsql rewrites the
    # redshift:// scheme to postgres:// for the driver).
    docker run -d --name db \
      -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ggsql \
      -p 5432:5432 postgres:17
    wait_for redshift docker exec db pg_isready -U postgres
    uri="redshift://postgres:postgres@localhost:5432/ggsql"
    ;;
  odbc)
    # Generic-ODBC leg: the same PostgreSQL container as the postgres leg,
    # but the URI's ?reader=odbc forces ggsql's ODBC fallback with psqlODBC
    # (installed by the workflow step for this leg), exercising OdbcReader
    # and the connection-string synthesis instead of the ADBC driver.
    docker run -d --name db \
      -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ggsql \
      -p 5432:5432 postgres:17
    wait_for odbc docker exec db pg_isready -U postgres
    uri="postgres://postgres:postgres@localhost:5432/ggsql?reader=odbc&Driver={PostgreSQL Unicode}"
    ;;
  monetdb)
    # MonetDB-over-ODBC leg: no usable ADBC driver exists, so the URI's
    # ?reader=odbc forces the ODBC fallback with the MonetDB ODBC driver and
    # the ggsql-monetdb DSN (both registered by the workflow step for this
    # leg). The DSN carries Host/Port/Database because the driver's
    # connection-string keywords differ from the generic Server= the URI
    # synthesis would emit.
    docker run -d --name db \
      -e MDB_CREATE_DBS=ggsql -e MDB_DB_ADMIN_PASS=monetdb \
      -p 50000:50000 monetdb/monetdb:Dec2025-SP3
    wait_for monetdb port_open 50000
    # Readiness plus preflight in one: poll with isql through the real
    # driver and DSN. Keeps unixODBC diagnostics visible in the log (the
    # Rust test otherwise reports an opaque IM005) and covers the daemon
    # accepting connections before the database is fully started.
    ok=0
    for _ in $(seq 1 24); do
      if isql -v ggsql-monetdb monetdb monetdb <<< 'SELECT 1;'; then
        ok=1
        break
      fi
      sleep 5
    done
    if [ "$ok" != 1 ]; then
      echo "monetdb ODBC preflight failed"
      exit 1
    fi
    uri="monetdb://monetdb:monetdb@localhost:50000/ggsql?reader=odbc&DSN=ggsql-monetdb"
    ;;
  oracle)
    # Oracle-over-ODBC leg: no usable ADBC driver exists without a Columnar
    # commercial license (the Foundry "oracle" driver is private), so the
    # URI's ?reader=odbc forces the ODBC fallback with the Oracle Instant
    # Client ODBC driver and the ggsql-oracle DSN (both registered by the
    # workflow step for this leg). gvenzl/oracle-xe faststart ships a
    # pre-built database, cutting startup from ~10 min to ~1-2; it is the
    # de-facto CI image and free under Oracle's Free Use Terms and
    # Conditions. APP_USER creates a plain schema user so the battery never
    # runs as SYS; XEPDB1 is the XE pluggable database.
    docker run -d --name db \
      -e ORACLE_PASSWORD=Ggsql_test1 \
      -e APP_USER=ggsql -e APP_USER_PASSWORD=Ggsql_test1 \
      -p 1521:1521 gvenzl/oracle-xe:21.3.0-slim-faststart
    # The image ships a healthcheck script doing a real sqlplus round-trip.
    wait_for oracle docker exec db healthcheck.sh
    # Readiness plus preflight in one, like the monetdb leg: poll with isql
    # through the real driver and DSN so unixODBC diagnostics stay visible.
    ok=0
    for _ in $(seq 1 24); do
      if isql -v ggsql-oracle ggsql Ggsql_test1 <<< 'SELECT 1 FROM dual;'; then
        ok=1
        break
      fi
      sleep 5
    done
    if [ "$ok" != 1 ]; then
      echo "oracle ODBC preflight failed"
      # unixODBC's "Can't open lib ... file not found" usually means a
      # NEEDED library of the driver failed to resolve, not that the driver
      # itself is missing — ldd names the culprit.
      driver_so=$(sed -n 's/^Driver=\(\/opt.*\)/\1/p' /etc/odbcinst.ini | head -1)
      echo "--- ldd $driver_so:"
      ldd "$driver_so" || true
      exit 1
    fi
    uri="oracle://ggsql:Ggsql_test1@localhost:1521/XEPDB1?reader=odbc&DSN=ggsql-oracle"
    ;;
  druid)
    # Druid leg: a nano-quickstart single server (all services in one JVM)
    # queried through the Foundry druid ADBC driver (installed with --pre by
    # the workflow step — it is prerelease-only). Druid has no DDL or plain
    # INSERT, so the datasource is created by an MSQ ingestion job submitted
    # to the router (8888) and polled to completion; the driver then talks
    # SQL to the broker (8082). DruidDialect requires_cache, so the battery
    # runs through ggsql's automatic sqlite cache wrap.
    # The 37.x image is distroless (busybox + static bash; no /usr/bin/env,
    # no perl), so the classic all-in-one supervise quickstart cannot run
    # inside it. The vendor-supported path is the docker-compose cluster:
    # zookeeper, postgres metadata, and one container per Druid service at
    # micro-quickstart sizing (~6 GiB total). MSQ ingest goes through the
    # router (8888); the ADBC driver talks to the broker (8082); overlord
    # and middleManager must also be up before MSQ tasks will run.
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
    # The row data comes from an inline EXTERN source: a UNION ALL of
    # literal SELECTs is NOT plannable by MSQ ("Union operation is only
    # supported between regular tables") because each literal select plans
    # as an inline datasource. Inline CSV with a header row names the
    # columns, but the planner still requires an explicit signature as
    # EXTERN's third argument ("EXTERN requires either a [signature] value
    # or an EXTEND clause"). All extern columns are strings, so CAST the
    # numerics in the SELECT.
    payload=$(python3 - <<'EOF'
import json
rows = "\n".join(
    f"{i},{i + 0.5},{'a' if i % 2 == 1 else 'b'}" for i in range(1, 9)
)
inline = json.dumps({"type": "inline", "data": "id,val,grp\n" + rows})
csv_fmt = json.dumps({"type": "csv", "findColumnsFromHeader": True})
sig = json.dumps([
    {"name": "id", "type": "STRING"},
    {"name": "val", "type": "STRING"},
    {"name": "grp", "type": "STRING"},
])
query = f"""
INSERT INTO ggsql_live_test
SELECT TIMESTAMP '2020-01-01 00:00:00' AS __time,
       CAST(id AS BIGINT) AS id,
       CAST(val AS DOUBLE) AS val,
       grp
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
    uri="druid://localhost:8082?tls=false"
    ;;
  sqlite)
    # Embedded reader — no server, no container, no dbc driver.
    uri="sqlite://:memory:"
    ;;
  duckdb)
    # Embedded reader — no server, no container, no dbc driver.
    uri="duckdb://memory"
    ;;
  datafusion)
    # In-process through the Foundry datafusion driver (installed by the
    # workflow's dbc step) — no server, no container. The test builds its
    # own datafusion:// reader and is gated on this flag rather than a URI.
    echo "GGSQL_TEST_DATAFUSION=1" >> "$GITHUB_ENV"
    echo "exported GGSQL_TEST_DATAFUSION"
    exit 0
    ;;
  *)
    echo "unknown backend: $backend" >&2
    exit 1
    ;;
esac

var="GGSQL_TEST_URI_$(echo "$backend" | tr '[:lower:]' '[:upper:]')"
echo "$var=$uri" >> "$GITHUB_ENV"
echo "exported $var"
