#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

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
export_uri monetdb "monetdb://monetdb:monetdb@localhost:50000/ggsql?reader=odbc&DSN=ggsql-monetdb"
