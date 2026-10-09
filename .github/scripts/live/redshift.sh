#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

# Pseudo-leg: the Foundry "redshift" ADBC driver is the PostgreSQL
# driver, so a stock PostgreSQL container exercises RedshiftDialect
# end-to-end over a wire-compatible server (ggsql rewrites the
# redshift:// scheme to postgres:// for the driver).
docker run -d --name db \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ggsql \
  -p 5432:5432 postgres:17
wait_for redshift docker exec db pg_isready -U postgres
export_uri redshift "redshift://postgres:postgres@localhost:5432/ggsql"
