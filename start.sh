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

# Accept a dotted IPv4 address that can be interpolated into a URL.
# Link-local (169.254.0.0/16) is rejected: it includes the instance metadata
# service and the ECS task-credential endpoint. Leading zeros are rejected so
# values such as 0177.0.0.1 cannot redirect the parser.
is_safe_ipv4() {
  local ip="$1" o1 o2 o3 o4 octet dec
  [[ "$ip" =~ ^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]] || return 1
  o1="${BASH_REMATCH[1]}"
  o2="${BASH_REMATCH[2]}"
  o3="${BASH_REMATCH[3]}"
  o4="${BASH_REMATCH[4]}"
  for octet in "$o1" "$o2" "$o3" "$o4"; do
    [[ "$octet" =~ ^0[0-9]+$ ]] && return 1
    dec=$((10#$octet))
    [ "$dec" -le 255 ] || return 1
  done
  if [ "$o1" -eq 0 ] || [ "$o1" -ge 224 ]; then
    return 1
  fi
  if [ "$o1" -eq 169 ] && [ "$o2" -eq 254 ]; then
    return 1
  fi
  return 0
}

# IMDSv2 only. A tokenless request is the SSRF path into instance credentials.
fetch_imds_local_ipv4() {
  local token
  token=$(curl -sf --max-time 2 -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)
  if [ -z "$token" ]; then
    return 0
  fi
  curl -sf --max-time 5 \
    -H "X-aws-ec2-metadata-token: ${token}" \
    "http://169.254.169.254/latest/meta-data/local-ipv4" || true
}

# ── Resolve Duckling host ─────────────────────────────────────────────────────
HOST_IP=""
if [ -f "$HOST_IP_FILE" ] && [ -s "$HOST_IP_FILE" ]; then
  CANDIDATE=$(tr -d '[:space:]' < "$HOST_IP_FILE")
  if is_safe_ipv4 "$CANDIDATE"; then
    HOST_IP="$CANDIDATE"
    echo "==> Duckling host IP from file: ${HOST_IP}"
  else
    echo "==> WARNING: Ignoring host IP file; value is not a usable IPv4 address"
  fi
fi

if [ -z "$HOST_IP" ]; then
  CANDIDATE=$(fetch_imds_local_ipv4)
  CANDIDATE=$(printf '%s' "$CANDIDATE" | tr -d '[:space:]')
  if is_safe_ipv4 "$CANDIDATE"; then
    HOST_IP="$CANDIDATE"
    echo "==> Duckling host IP from metadata: ${HOST_IP}"
  fi
fi

if [ -z "$HOST_IP" ]; then
  HOST_IP="127.0.0.1"
  echo "==> WARNING: Could not determine host IP; falling back to 127.0.0.1"
fi

DUCKLING_URL="http://${HOST_IP}:8000/parse"
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
exec /opt/scheduler0-nlp/bin/uvicorn app:app \
  --host 0.0.0.0 \
  --port 8080 \
  --no-server-header \
  --limit-concurrency 64 \
  --timeout-keep-alive 5
