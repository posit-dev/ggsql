#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

docker run -d --name db --privileged \
  -p 8563:8563 exasol/docker-db:2026.1.2
wait_for exasol port_open 8563
# The port opens long before the database finishes its startup stages
# (the driver was getting "TLS error: Connection reset by peer" on a
# fixed sleep). A completed TLS handshake is the earliest reliable
# readiness signal; the server resets connections until then.
wait_for exasol-tls bash -c 'echo | timeout 5 openssl s_client -connect localhost:8563 >/dev/null 2>&1'
export_uri exasol "exasol://sys:exasol@localhost:8563/?tls=true&validateservercertificate=0"
