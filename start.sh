#!/bin/bash
# Starts Duckling (from the host-mounted binary at /duckling) then Uvicorn.
# The ECS task mounts the EC2 host path /opt/duckling to /duckling read-only,
# so the binary and its shared libraries are pre-built by the EC2 user-data script.
set -euo pipefail

DUCKLING_BIN="/duckling/duckling-example-exe"
DUCKLING_URL="http://127.0.0.1:8000/parse"
DUCKLING_TIMEOUT=120

if [ ! -x "$DUCKLING_BIN" ]; then
  echo "ERROR: $DUCKLING_BIN not found or not executable. EC2 user-data bootstrap may not have completed."
  exit 1
fi

export LD_LIBRARY_PATH="/duckling/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "==> Starting Duckling"
"$DUCKLING_BIN" &
DUCKLING_PID=$!

echo "==> Waiting up to ${DUCKLING_TIMEOUT}s for Duckling to accept requests"
ELAPSED=0
until curl -sf -X POST "$DUCKLING_URL" \
    -d 'locale=en_GB&text=tomorrow&dims=["time"]' > /dev/null 2>&1; do
  if ! kill -0 "$DUCKLING_PID" 2>/dev/null; then
    echo "ERROR: Duckling process exited unexpectedly"
    exit 1
  fi
  if [ "$ELAPSED" -ge "$DUCKLING_TIMEOUT" ]; then
    echo "ERROR: Duckling did not become ready within ${DUCKLING_TIMEOUT}s"
    kill "$DUCKLING_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 2
  ELAPSED=$((ELAPSED + 2))
done
echo "==> Duckling ready after ${ELAPSED}s"

echo "==> Starting Uvicorn"
exec /opt/scheduler0-nlp/bin/uvicorn app:app --host 0.0.0.0 --port 8080
