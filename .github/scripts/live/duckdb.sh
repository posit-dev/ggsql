#!/usr/bin/env bash
# Embedded reader — no server, no container, no dbc driver.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

export_uri duckdb "duckdb://memory"
