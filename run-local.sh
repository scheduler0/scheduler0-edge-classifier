#!/usr/bin/env bash
# Run the intent classifier locally.
#
# Starts Duckling in Docker (rasa/duckling image, container port 8000), waits
# for it to accept requests, then launches Uvicorn from the local .venv with
# --reload.
#
# Usage:
#   ./run-local.sh
#
# Environment variables:
#   PORT              – Uvicorn port                 (default: 5000)
#   DUCKLING_PORT     – host port for Duckling       (default: 8000)
#   DUCKLING_TIMEOUT  – seconds to wait for Duckling (default: 60)
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-5000}"
DUCKLING_PORT="${DUCKLING_PORT:-8000}"
DUCKLING_TIMEOUT="${DUCKLING_TIMEOUT:-60}"
DUCKLING_CONTAINER="scheduler0-duckling"
DUCKLING_URL="http://127.0.0.1:${DUCKLING_PORT}/parse"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -x .venv/bin/uvicorn ]; then
  echo "ERROR: .venv/bin/uvicorn not found. Set up the venv first:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  echo "  python -m spacy download en_core_web_sm"
  exit 1
fi

if [ "$PORT" = "$DUCKLING_PORT" ]; then
  echo "ERROR: PORT and DUCKLING_PORT both set to ${PORT}; they must differ."
  exit 1
fi

# ── Start Duckling (Docker) ───────────────────────────────────────────────────
if curl -sf --max-time 2 -X POST "$DUCKLING_URL" \
    -d 'locale=en_GB&text=tomorrow&dims=["time"]' > /dev/null 2>&1; then
  echo "==> Duckling already responding at ${DUCKLING_URL}"
else
  if ! command -v docker > /dev/null 2>&1; then
    echo "ERROR: Duckling is not running and Docker is not installed."
    exit 1
  fi

  if [ "$(docker ps -q -f name="^${DUCKLING_CONTAINER}$")" ]; then
    echo "==> Duckling container already running (still warming up)"
  elif [ "$(docker ps -aq -f name="^${DUCKLING_CONTAINER}$")" ]; then
    echo "==> Starting existing Duckling container"
    docker start "$DUCKLING_CONTAINER" > /dev/null
  else
    echo "==> Launching Duckling container (rasa/duckling, port ${DUCKLING_PORT})"
    # rasa/duckling listens on 8000 inside the container
    docker run -d --name "$DUCKLING_CONTAINER" \
      -p "${DUCKLING_PORT}:8000" rasa/duckling > /dev/null
  fi

  echo "==> Waiting up to ${DUCKLING_TIMEOUT}s for Duckling at ${DUCKLING_URL}"
  ELAPSED=0
  until curl -sf --connect-timeout 2 --max-time 5 -X POST "$DUCKLING_URL" \
      -d 'locale=en_GB&text=tomorrow&dims=["time"]' > /dev/null 2>&1; do
    if [ "$ELAPSED" -ge "$DUCKLING_TIMEOUT" ]; then
      echo "ERROR: Duckling did not become ready within ${DUCKLING_TIMEOUT}s"
      echo "Check logs: docker logs ${DUCKLING_CONTAINER}"
      exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
  done
  echo "==> Duckling ready after ${ELAPSED}s"
fi

# ── Start Uvicorn ─────────────────────────────────────────────────────────────
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN > /dev/null 2>&1; then
  echo "ERROR: Port ${PORT} is already in use."
  if [ "$PORT" = "5000" ]; then
    echo "On macOS, AirPlay Receiver often binds 5000 (ControlCenter)."
    echo "Disable it: System Settings → General → AirDrop & Handoff → AirPlay Receiver"
    echo "Or run with a free port: PORT=5001 ./run-local.sh"
  fi
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN || true
  exit 1
fi

export DUCKLING_URL
echo "==> Starting Uvicorn on http://127.0.0.1:${PORT} (Ctrl-C to stop)"
echo "    Duckling container '${DUCKLING_CONTAINER}' keeps running; stop it with:"
echo "    docker stop ${DUCKLING_CONTAINER}"
exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port "$PORT" --reload
