#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

# Generic-ODBC leg: the same PostgreSQL container as the postgres leg,
# but the URI's ?reader=odbc forces ggsql's ODBC fallback with psqlODBC
# (installed by the workflow step for this leg), exercising OdbcReader
# and the connection-string synthesis instead of the ADBC driver.
docker run -d --name db \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ggsql \
  -p 5432:5432 postgres:17
wait_for odbc docker exec db pg_isready -U postgres
export_uri odbc "postgres://postgres:postgres@localhost:5432/ggsql?reader=odbc&Driver={PostgreSQL Unicode}"
