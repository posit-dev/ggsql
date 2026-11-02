#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

# Password meets the SQL Server complexity rules (3 of 4 categories).
docker run -d --name db \
  -e ACCEPT_EULA=Y -e MSSQL_SA_PASSWORD=Ggsql_test1 \
  -p 1433:1433 mcr.microsoft.com/mssql/server:2022-latest
wait_for mssql docker exec db /opt/mssql-tools18/bin/sqlcmd \
  -C -U sa -P Ggsql_test1 -Q "SELECT 1" -l 2
# TrustServerCertificate: the container's TLS cert is self-signed.
export_uri mssql "mssql://sa:Ggsql_test1@localhost:1433/master?TrustServerCertificate=true"
