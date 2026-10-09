#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

# Oracle-over-ODBC leg: no usable ADBC driver exists without a Columnar
# commercial license (the Foundry "oracle" driver is private), so the
# URI's ?reader=odbc forces the ODBC fallback with the Oracle Instant
# Client ODBC driver (registered by the workflow step for this leg).
# gvenzl/oracle-xe faststart ships a pre-built database, cutting startup
# from ~10 min to ~1-2; it is the de-facto CI image and free under
# Oracle's Free Use Terms and Conditions. APP_USER creates a plain schema
# user so the battery never runs as SYS; XEPDB1 is the XE pluggable
# database.
docker run -d --name db \
  -e ORACLE_PASSWORD=Ggsql_test1 \
  -e APP_USER=ggsql -e APP_USER_PASSWORD=Ggsql_test1 \
  -p 1521:1521 gvenzl/oracle-xe:21.3.0-slim-faststart
# The image ships a healthcheck script doing a real sqlplus round-trip.
wait_for oracle docker exec db healthcheck.sh
# Readiness plus preflight in one, like the monetdb leg, with isql -k
# (SQLDriverConnect — the API ggsql uses). The string is DSN-less:
# unixODBC's DSN attribute mapping never delivers DBQ to this driver
# (every DBQ spelling fails with ORA-12162 via DSN while the identical
# values connect DSN-less), so both the preflight and the ggsql URI
# carry Driver + full TNS descriptor directly. This mirrors what
# connection.rs synthesizes from the URI below.
ok=0
for _ in $(seq 1 24); do
  if isql -v -k "Driver=Oracle;DBQ=(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=localhost)(PORT=1521))(CONNECT_DATA=(SERVICE_NAME=XEPDB1)));UID=ggsql;PWD=Ggsql_test1" <<< 'SELECT 1 FROM dual;'; then
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
  # Fallback probe: bypass the DSN entirely with a full TNS descriptor.
  # If this connects while the DSN form does not, the ini attribute
  # mapping (not the connect string) is at fault.
  echo "--- fallback probe: driver path + full descriptor:"
  isql -v -k "Driver=$driver_so;DBQ=(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=localhost)(PORT=1521))(CONNECT_DATA=(SERVICE_NAME=XEPDB1)));UID=ggsql;PWD=Ggsql_test1" \
    <<< 'SELECT 1 FROM dual;' || true
  exit 1
fi
# Driver + DBQ in the URI: connection.rs turns this into
# "Driver={Oracle};UID=...;PWD=...;DBQ=(...)" (a DBQ= param suppresses
# the Server/Port/Database synthesis). No DSN is involved.
export_uri oracle "oracle://ggsql:Ggsql_test1@localhost:1521/XEPDB1?reader=odbc&Driver={Oracle}&DBQ=(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=localhost)(PORT=1521))(CONNECT_DATA=(SERVICE_NAME=XEPDB1)))"
