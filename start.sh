#!/bin/bash
# Container entrypoint for the intent-classifier ECS task.
#
# Duckling runs on the EC2 HOST (not inside the container) because the GHC
# threaded RTS does not handle time-dimension parsing reliably inside Docker's
# execution environment. The host writes its private IP to /opt/duckling/host_ip
# (mounted at /duckling/host_ip read-only); start.sh reads it to build the URL.
#
# After Duckling is reachable, Uvicorn is started using the Python venv baked
# into the golden AMI and mounted at /opt/scheduler0-nlp.
set -euo pipefail

HOST_IP_FILE="/duckling/host_ip"
DUCKLING_TIMEOUT="${DUCKLING_TIMEOUT:-120}"

# ── Resolve Duckling host ─────────────────────────────────────────────────────
if [ -f "$HOST_IP_FILE" ] && [ -s "$HOST_IP_FILE" ]; then
  HOST_IP=$(tr -d '[:space:]' < "$HOST_IP_FILE")
  DUCKLING_URL="http://${HOST_IP}:8000/parse"
  echo "==> Duckling host IP from file: ${HOST_IP}"
else
  # Fallback: try EC2 instance metadata (available in awsvpc mode via ECS proxy)
  HOST_IP=$(curl -sf --max-time 5 http://169.254.169.254/latest/meta-data/local-ipv4 2>/dev/null || echo "")
  if [ -n "$HOST_IP" ]; then
    DUCKLING_URL="http://${HOST_IP}:8000/parse"
    echo "==> Duckling host IP from metadata: ${HOST_IP}"
  else
    DUCKLING_URL="http://127.0.0.1:8000/parse"
    echo "==> WARNING: Could not determine host IP; falling back to 127.0.0.1"
  fi
fi

export DUCKLING_URL

# ── Wait for Duckling ─────────────────────────────────────────────────────────
echo "==> Waiting up to ${DUCKLING_TIMEOUT}s for Duckling at ${DUCKLING_URL}"
ELAPSED=0
until curl -sf --connect-timeout 5 --max-time 10 -X POST "$DUCKLING_URL" \
    -d 'locale=en_GB&text=tomorrow&dims=["time"]' > /dev/null 2>&1; do
  if [ "$ELAPSED" -ge "$DUCKLING_TIMEOUT" ]; then
    echo "ERROR: Duckling did not become ready within ${DUCKLING_TIMEOUT}s"
    exit 1
  fi
  sleep 2
  ELAPSED=$((ELAPSED + 2))
done
echo "==> Duckling ready after ${ELAPSED}s"

# ── Start Uvicorn ─────────────────────────────────────────────────────────────
echo "==> Starting Uvicorn"
exec /opt/scheduler0-nlp/bin/uvicorn app:app --host 0.0.0.0 --port 8080
