#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

# Passwordless 'default' only authenticates from localhost, but
# port-mapped connections arrive from the docker bridge IP.
docker run -d --name db \
  -e CLICKHOUSE_PASSWORD=clickhouse \
  -p 8123:8123 clickhouse/clickhouse-server:25.8
wait_for clickhouse curl -sf http://localhost:8123/ping
# The driver speaks HTTP and ignores URL userinfo: credentials go in
# query params, which ggsql maps to the username/password options.
export_uri clickhouse "clickhouse://localhost:8123?username=default&password=clickhouse"
