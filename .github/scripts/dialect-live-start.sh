#!/usr/bin/env bash
# Start the container for one live-dialect-test backend and publish its
# connection URI as GGSQL_TEST_URI_<BACKEND> in $GITHUB_ENV. One matrix leg
# of .github/workflows/dialect-live.yml invokes this with its backend name;
# the work itself lives in the per-backend script under live/, which keeps
# each backend's setup (and its quirks) reviewable on its own.
set -euo pipefail

backend="${1:?usage: dialect-live-start.sh <backend>}"
script="$(dirname "$0")/live/$backend.sh"
if [ ! -f "$script" ]; then
  echo "unknown backend: $backend" >&2
  exit 1
fi
exec bash "$script"
