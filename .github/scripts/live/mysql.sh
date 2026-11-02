#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

docker run -d --name db \
  -e MYSQL_ROOT_PASSWORD=mysql -e MYSQL_DATABASE=ggsql \
  -p 3306:3306 mysql:8.4
# mysqladmin ping answers over the unix socket before init finishes and
# TCP is serving; a real TCP query against the ggsql database only
# succeeds once the entrypoint's init/restart cycle is complete.
wait_for mysql docker exec db mysql -uroot -pmysql --protocol=TCP -h127.0.0.1 -e "SELECT 1" ggsql
export_uri mysql "mysql://root:mysql@localhost:3306/ggsql"
