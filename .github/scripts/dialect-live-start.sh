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
