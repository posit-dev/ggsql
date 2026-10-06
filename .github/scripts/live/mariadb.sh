#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

docker run -d --name db \
  -e MARIADB_ROOT_PASSWORD=mysql -e MARIADB_DATABASE=ggsql \
  -p 3306:3306 mariadb:11
wait_for mariadb docker exec db mariadb -uroot -pmysql --protocol=TCP -h127.0.0.1 -e "SELECT 1" ggsql
export_uri mariadb "mariadb://root:mysql@localhost:3306/ggsql"
