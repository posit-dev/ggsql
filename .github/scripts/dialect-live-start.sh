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
    wait_for trino bash -c 'curl -sf http://localhost:8080/v1/info | grep -q "\"starting\":false"'
    # The driver defaults to HTTPS; SSL=false selects plain HTTP.
    uri="trino://test@localhost:8080/memory/default?SSL=false"
    ;;
  mysql)
    docker run -d --name db \
      -e MYSQL_ROOT_PASSWORD=mysql -e MYSQL_DATABASE=ggsql \
      -p 3306:3306 mysql:8.4
    wait_for mysql docker exec db mysqladmin ping -uroot -pmysql --silent
    uri="mysql://root:mysql@localhost:3306/ggsql"
    ;;
  mariadb)
    docker run -d --name db \
      -e MARIADB_ROOT_PASSWORD=mysql -e MARIADB_DATABASE=ggsql \
      -p 3306:3306 mariadb:11
    wait_for mariadb docker exec db mariadb-admin ping -uroot -pmysql --silent
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
    # The port opens before the database finishes its startup stages.
    sleep 30
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
    # The daemon accepts connections before the database is fully started.
    sleep 10
    uri="monetdb://monetdb:monetdb@localhost:50000/ggsql?reader=odbc&DSN=ggsql-monetdb"
    ;;
  druid)
    # Druid leg: a nano-quickstart single server (all services in one JVM)
    # queried through the Foundry druid ADBC driver (installed with --pre by
    # the workflow step — it is prerelease-only). Druid has no DDL or plain
    # INSERT, so the datasource is created by an MSQ ingestion job submitted
    # to the router (8888) and polled to completion; the driver then talks
    # SQL to the broker (8082). DruidDialect requires_cache, so the battery
    # runs through ggsql's automatic sqlite cache wrap.
    docker run -d --name db \
      -e DRUID_SINGLE_NODE_CONF=nano-quickstart \
      -p 8888:8888 -p 8082:8082 apache/druid:37.0.0
    wait_for druid curl -sf http://localhost:8082/status/health
    # MSQ INSERT: every Druid datasource needs a __time column; one shared
    # timestamp suffices. PARTITIONED BY ALL puts everything in one segment.
    payload=$(python3 - <<'EOF'
import json
query = """
INSERT INTO ggsql_live_test
SELECT TIMESTAMP '2020-01-01 00:00:00' AS __time, 1 AS id, 1.5 AS val, 'a' AS grp UNION ALL
SELECT TIMESTAMP '2020-01-01 00:00:00', 2, 2.5, 'b' UNION ALL
SELECT TIMESTAMP '2020-01-01 00:00:00', 3, 3.5, 'a' UNION ALL
SELECT TIMESTAMP '2020-01-01 00:00:00', 4, 4.5, 'b' UNION ALL
SELECT TIMESTAMP '2020-01-01 00:00:00', 5, 5.5, 'a' UNION ALL
SELECT TIMESTAMP '2020-01-01 00:00:00', 6, 6.5, 'b' UNION ALL
SELECT TIMESTAMP '2020-01-01 00:00:00', 7, 7.5, 'a' UNION ALL
SELECT TIMESTAMP '2020-01-01 00:00:00', 8, 8.5, 'b'
PARTITIONED BY ALL
"""
print(json.dumps({"query": query, "context": {"engine": "msq-task"}}))
EOF
)
    task_id=$(curl -sf -X POST -H 'Content-Type: application/json' \
      -d "$payload" http://localhost:8888/druid/v2/sql \
      | sed -n 's/.*"taskId":"\([^"]*\)".*/\1/p')
    if [ -z "$task_id" ]; then
      echo "MSQ ingest submission returned no taskId"
      docker logs db || true
      exit 1
    fi
    # MSQ jobs are asynchronous; poll the task until it succeeds or fails.
    for _ in $(seq 1 60); do
      status=$(curl -sf "http://localhost:8888/druid/indexer/v1/task/$task_id/status" || true)
      case "$status" in
        *'"status":"SUCCESS"'*) break ;;
        *'"statusCode":"FAILED"'*)
          echo "MSQ ingest task $task_id FAILED"
          docker logs db || true
          exit 1
          ;;
      esac
      sleep 5
    done
    # Segment handoff lags task success; wait until the broker serves rows.
    for _ in $(seq 1 30); do
      cnt=$(curl -sf -X POST -H 'Content-Type: application/json' \
        -d '{"query":"SELECT COUNT(*) AS c FROM ggsql_live_test"}' \
        http://localhost:8082/druid/v2/sql \
        | grep -o '"c":[0-9]*' | cut -d: -f2 || true)
      [ "$cnt" = "8" ] && break
      sleep 5
    done
    if [ "$cnt" != "8" ]; then
      echo "datasource never became queryable (count=$cnt)"
      docker logs db || true
      exit 1
    fi
    uri="druid://localhost:8082"
    ;;
  sqlite)
    # Embedded reader — no server, no container, no dbc driver.
    uri="sqlite://:memory:"
    ;;
  duckdb)
    # Embedded reader — no server, no container, no dbc driver.
    uri="duckdb://memory"
    ;;
  *)
    echo "unknown backend: $backend" >&2
    exit 1
    ;;
esac

var="GGSQL_TEST_URI_$(echo "$backend" | tr '[:lower:]' '[:upper:]')"
echo "$var=$uri" >> "$GITHUB_ENV"
echo "exported $var"
