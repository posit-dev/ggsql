# Shared helpers for the per-backend live-test start scripts. Sourced by
# each live/<backend>.sh; not executed directly.
#
# Health checks poll until the backend actually accepts connections; on
# timeout the container logs are dumped to make CI failures debuggable.

# wait_for <name> <cmd...>: poll cmd every 5s (up to ~7.5 min) until it
# exits 0.
wait_for() {
  local name="$1"
  shift
  for _ in $(seq 1 90); do
    if "$@" >/dev/null 2>&1; then
      echo "$name is up"
      return 0
    fi
    sleep 5
  done
  echo "$name failed to start"
  docker logs db || true
  exit 1
}

port_open() { (exec 3<>"/dev/tcp/localhost/$1") 2>/dev/null; }

# Publish the connection URI as GGSQL_TEST_URI_<BACKEND> in $GITHUB_ENV.
export_uri() {
  local backend="$1" uri="$2"
  local var="GGSQL_TEST_URI_$(echo "$backend" | tr '[:lower:]' '[:upper:]')"
  echo "$var=$uri" >> "$GITHUB_ENV"
  echo "exported $var"
}
