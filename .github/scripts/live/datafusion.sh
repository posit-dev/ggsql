#!/usr/bin/env bash
# In-process through the Foundry datafusion driver (installed by the
# workflow's dbc step) — no server, no container. The test builds its own
# datafusion:// reader and is gated on this flag rather than a URI.
set -euo pipefail

echo "GGSQL_TEST_DATAFUSION=1" >> "$GITHUB_ENV"
echo "exported GGSQL_TEST_DATAFUSION"
