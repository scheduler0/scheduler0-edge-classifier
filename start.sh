#!/bin/bash
# Container entrypoint for the intent-classifier ECS task.
#
# Duckling runs on the EC2 HOST (not inside the container) because the GHC
# threaded RTS does not handle time-dimension parsing reliably inside Docker's
# execution environment. The host writes its private IP to /opt/duckling/host_ip
# (mounted at /duckling/host_ip read-only); start.sh reads it to build the URL.
#
# After Duckling is reachable, Uvicorn is started using the Python venv baked
# into the golden AMI and mounted at /opt/scheduler0-nlp. The server drops to
# the unprivileged appuser when that venv is readable.
set -euo pipefail

HOST_IP_FILE="/duckling/host_ip"
DUCKLING_TIMEOUT="${DUCKLING_TIMEOUT:-120}"

# Only an RFC1918 address is accepted from the host file or IMDS.
# Link-local (169.254.0.0/16, which includes the metadata service), loopback,
# public addresses, and leading-zero octets are rejected so a poisoned host_ip
# file cannot redirect Duckling traffic.
is_safe_host_ip() {
  local ip="$1"
  [[ "$ip" =~ ^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]] || return 1
  local o1="${BASH_REMATCH[1]}" o2="${BASH_REMATCH[2]}" o3="${BASH_REMATCH[3]}" o4="${BASH_REMATCH[4]}"
  local octet
  for octet in "$o1" "$o2" "$o3" "$o4"; do
    [[ "$octet" =~ ^0[0-9]+$ ]] && return 1
    ((10#$octet <= 255)) || return 1
  done
  if ((10#$o1 == 10)); then
    return 0
  fi
  if ((10#$o1 == 172 && 10#$o2 >= 16 && 10#$o2 <= 31)); then
    return 0
  fi
  if ((10#$o1 == 192 && 10#$o2 == 168)); then
    return 0
  fi
  return 1
}

if [ "${1:-}" = "--check-host-ip" ]; then
  if is_safe_host_ip "${2:-}"; then
    exit 0
  fi
  exit 1
fi

if ! [[ "$DUCKLING_TIMEOUT" =~ ^[0-9]+$ ]] || ((10#$DUCKLING_TIMEOUT < 1 || 10#$DUCKLING_TIMEOUT > 600)); then
  DUCKLING_TIMEOUT=120
fi

read_metadata_ipv4() {
  local token ip
  token=$(curl -sf --max-time 2 --max-redirs 0 -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)
  if [ -n "$token" ]; then
    ip=$(curl -sf --max-time 5 --max-redirs 0 \
      -H "X-aws-ec2-metadata-token: ${token}" \
      http://169.254.169.254/latest/meta-data/local-ipv4 || true)
    if is_safe_host_ip "$ip"; then
      printf '%s\n' "$ip"
      return 0
    fi
  fi
  # IMDSv1 only as a last resort, and only for the private IPv4 address.
  # The response is still passed through is_safe_host_ip.
  ip=$(curl -sf --max-time 5 --max-redirs 0 http://169.254.169.254/latest/meta-data/local-ipv4 || true)
  if is_safe_host_ip "$ip"; then
    printf '%s\n' "$ip"
    return 0
  fi
  return 1
}

# ── Resolve Duckling host ─────────────────────────────────────────────────────
HOST_IP=""
if [ -f "$HOST_IP_FILE" ] && [ -s "$HOST_IP_FILE" ]; then
  CANDIDATE=$(tr -d '[:space:]' < "$HOST_IP_FILE")
  if is_safe_host_ip "$CANDIDATE"; then
    HOST_IP="$CANDIDATE"
    echo "==> Duckling host IP from file: ${HOST_IP}"
  else
    echo "==> WARNING: Ignoring unsafe value in ${HOST_IP_FILE}"
  fi
fi

if [ -z "$HOST_IP" ]; then
  if HOST_IP=$(read_metadata_ipv4); then
    echo "==> Duckling host IP from metadata: ${HOST_IP}"
  else
    HOST_IP=""
  fi
fi

if [ -n "$HOST_IP" ]; then
  DUCKLING_URL="http://${HOST_IP}:8000/parse"
else
  DUCKLING_URL="http://127.0.0.1:8000/parse"
  echo "==> WARNING: Could not determine host IP; falling back to 127.0.0.1"
fi

export DUCKLING_URL
export PYTHONDONTWRITEBYTECODE=1

# ── Wait for Duckling ─────────────────────────────────────────────────────────
echo "==> Waiting up to ${DUCKLING_TIMEOUT}s for Duckling at ${DUCKLING_URL}"
ELAPSED=0
until curl -sf --connect-timeout 5 --max-time 10 --max-redirs 0 -X POST "$DUCKLING_URL" \
    -d 'locale=en_GB&text=tomorrow&dims=["time"]' > /dev/null 2>&1; do
  if [ "$ELAPSED" -ge "$((10#$DUCKLING_TIMEOUT))" ]; then
    echo "ERROR: Duckling did not become ready within ${DUCKLING_TIMEOUT}s"
    exit 1
  fi
  sleep 2
  ELAPSED=$((ELAPSED + 2))
done
echo "==> Duckling ready after ${ELAPSED}s"

# ── Start Uvicorn ─────────────────────────────────────────────────────────────
start_uvicorn() {
  if [ "$(id -u)" -eq 0 ] && id appuser >/dev/null 2>&1 && command -v setpriv >/dev/null 2>&1; then
    if setpriv --reuid=appuser --regid=appuser --init-groups -- test -x /opt/scheduler0-nlp/bin/uvicorn \
      && setpriv --reuid=appuser --regid=appuser --init-groups -- test -r /app/app.py; then
      echo "==> Starting Uvicorn as appuser"
      export HOME=/home/appuser
      exec setpriv --reuid=appuser --regid=appuser --init-groups -- \
        /opt/scheduler0-nlp/bin/uvicorn app:app --host 0.0.0.0 --port 8080
    fi
    echo "==> WARNING: appuser cannot read the NLP venv or app; starting as root"
  fi
  echo "==> Starting Uvicorn"
  exec /opt/scheduler0-nlp/bin/uvicorn app:app --host 0.0.0.0 --port 8080
}

start_uvicorn
