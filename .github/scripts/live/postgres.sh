#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

docker run -d --name db \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ggsql \
  -p 5432:5432 postgres:17
wait_for postgres docker exec db pg_isready -U postgres
export_uri postgres "postgres://postgres:postgres@localhost:5432/ggsql"
