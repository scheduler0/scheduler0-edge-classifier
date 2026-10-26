#!/usr/bin/env bash
# Installs Duckling and the Python NLP stack on Amazon Linux 2023 (or yum-based).
#
# Usage:
#   ./install_scheduler0_nlp.sh              # interactive, writes to stdout
#   ./install_scheduler0_nlp.sh --ci         # suitable for EC2 user-data; writes
#                                            #   a bootstrap marker on success and
#                                            #   logs to $LOG_FILE
#
# Environment variables (all have defaults):
#   DUCKLING_DIR  – where to clone/build Duckling  (default: $HOME/workspace/duckling)
#   NLP_DIR       – Python virtualenv root          (default: /opt/scheduler0-nlp)
#   DUCKLING_PORT – Duckling HTTP port              (default: 8000)
#   DUCKLING_EXPORT_DIR – where to copy the binary and libs for ECS mounting
#                                                   (default: /opt/duckling)
#   LOG_FILE      – log destination (--ci mode)     (default: /var/log/scheduler0-intent-classifier-bootstrap.log)
#   BOOTSTRAP_MARKER – file written on success      (default: /var/lib/scheduler0-intent-classifier/bootstrap-complete)
#
# Stack and Duckling are pinned. The Haskell installer is a release tarball
# checked with SHA-256. Duckling is fetched at DUCKLING_REV rather than
# whatever master happens to be.

set -Eeuo pipefail
# HOME is not set in the EC2 user-data environment; default to /root so the
# ${HOME:-...} expansions below don't trip the -u (nounset) flag.
export HOME="${HOME:-/root}"

CI_MODE=false
for arg in "$@"; do
  [ "$arg" = "--ci" ] && CI_MODE=true
done

DUCKLING_DIR="${DUCKLING_DIR:-$HOME/workspace/duckling}"
NLP_DIR="${NLP_DIR:-/opt/scheduler0-nlp}"
DUCKLING_PORT="${DUCKLING_PORT:-8000}"
if ! [[ "$DUCKLING_PORT" =~ ^[0-9]+$ ]] || ((10#$DUCKLING_PORT < 1 || 10#$DUCKLING_PORT > 65535)); then
  echo "ERROR: DUCKLING_PORT must be an integer from 1 to 65535."
  exit 1
fi
DUCKLING_EXPORT_DIR="${DUCKLING_EXPORT_DIR:-/opt/duckling}"
LOG_FILE="${LOG_FILE:-/var/log/scheduler0-intent-classifier-bootstrap.log}"
BOOTSTRAP_MARKER="${BOOTSTRAP_MARKER:-/var/lib/scheduler0-intent-classifier/bootstrap-complete}"
# facebook/duckling main at the time this pin was recorded.
DUCKLING_REV="59a13ff87b1aa8be6b93d387244f8636b26185c5"
STACK_VERSION="3.11.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if $CI_MODE; then
  mkdir -p "$(dirname "$LOG_FILE")"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "==> Detecting package manager"
if command -v dnf >/dev/null 2>&1; then
  PM="dnf"
else
  PM="yum"
fi

echo "==> Updating system packages"
sudo "$PM" update -y

echo "==> Installing build tools"
sudo "$PM" groupinstall -y "Development Tools" || true

echo "==> Installing system dependencies"
sudo "$PM" install -y \
  gmp-devel \
  zlib-devel \
  ncurses-devel \
  libffi-devel \
  openssl-devel \
  readline-devel \
  bzip2-devel \
  sqlite-devel \
  pcre-devel \
  pkgconfig \
  perl \
  tar \
  xz \
  make \
  gcc \
  gcc-c++ \
  git \
  wget \
  which \
  python3 \
  python3-devel
# curl-minimal is pre-installed on Amazon Linux 2023 EC2 and conflicts with the
# full curl package; skip it — curl-minimal provides the curl command we need.
# python3-pip is not reliably in the AL2023 dnf repos; bootstrap via ensurepip.
python3 -m ensurepip --upgrade

install_pinned_stack() {
  local arch sha tmp name
  case "$(uname -m)" in
    x86_64|amd64)
      arch="linux-x86_64"
      sha="1fda71e657cd8d355625cc66b61b352699279dfee2664c014a392163bd19a952"
      ;;
    aarch64|arm64)
      arch="linux-aarch64"
      sha="1617ae9976a5cd38ad4daec583b026b589eb45d5482afb045cd4ca8c8d0de6d0"
      ;;
    *)
      echo "ERROR: no pinned Stack ${STACK_VERSION} build for $(uname -m)"
      exit 1
      ;;
  esac
  name="stack-${STACK_VERSION}-${arch}"
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '${tmp}'" RETURN
  echo "==> Downloading Stack ${STACK_VERSION} (${arch})"
  curl -fsSL \
    "https://github.com/commercialhaskell/stack/releases/download/v${STACK_VERSION}/${name}.tar.gz" \
    -o "${tmp}/stack.tar.gz"
  echo "${sha}  ${tmp}/stack.tar.gz" | sha256sum -c -
  tar -xzf "${tmp}/stack.tar.gz" -C "${tmp}"
  if [ ! -f "${tmp}/${name}/stack" ]; then
    echo "ERROR: Stack archive did not contain ${name}/stack"
    exit 1
  fi
  sudo install -m 0755 "${tmp}/${name}/stack" /usr/local/bin/stack
}

install_pinned_spacy_model() {
  local tmp
  tmp="$(mktemp)"
  # shellcheck disable=SC2064
  trap "rm -f '${tmp}'" RETURN
  curl -fsSL \
    -o "$tmp" \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
  echo "86cc141f63942d4b2c5fcee06630fd6f904788d2f0ab005cce45aadb8fb73889  ${tmp}" | sha256sum -c -
  python -m pip install "$tmp"
}

install_duckling_service() {
  # Snap's example server binds 0.0.0.0 unless --address is set. The container
  # reaches Duckling on the host private IP, so the service listens only on
  # that RFC1918 address and not on a public interface.
  local unit bind_script
  unit="/etc/systemd/system/duckling.service"
  bind_script="${DUCKLING_EXPORT_DIR}/duckling-bind.sh"
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "==> systemd is not available; Duckling was not registered as a service"
    return 0
  fi
  if ! id duckling >/dev/null 2>&1; then
    # Home is not the export directory. useradd must not take ownership of
    # the directory that holds the root-owned bind script.
    sudo useradd --system --home-dir /nonexistent --shell /sbin/nologin duckling
  fi
  sudo tee "$bind_script" >/dev/null <<'EOF'
#!/bin/bash
set -euo pipefail
PORT="${DUCKLING_PORT:-8000}"
EXPORT_DIR="${DUCKLING_EXPORT_DIR:-/opt/duckling}"
HOST_IP_FILE="${EXPORT_DIR}/host_ip"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || ((10#$PORT < 1 || 10#$PORT > 65535)); then
  echo "DUCKLING_PORT is not a TCP port" >&2
  exit 1
fi

is_rfc1918() {
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

private_ip() {
  local ip
  local candidates=""
  candidates="$(hostname -I 2>/dev/null || true)"
  if [ -z "$candidates" ] && command -v ip >/dev/null 2>&1; then
    candidates="$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1)"
  fi
  for ip in $candidates; do
    if is_rfc1918 "$ip"; then
      printf '%s\n' "$ip"
      return 0
    fi
  done
  return 1
}

write_host_ip() {
  local ip tmp
  ip="$(private_ip)" || {
    echo "No RFC1918 address available for Duckling" >&2
    exit 1
  }
  tmp="$(mktemp "${EXPORT_DIR}/host_ip.XXXXXX")"
  printf '%s\n' "$ip" > "$tmp"
  chmod 644 "$tmp"
  mv -f "$tmp" "$HOST_IP_FILE"
}

case "${1:-}" in
  --write-host-ip)
    write_host_ip
    ;;
  --serve)
    if [ ! -s "$HOST_IP_FILE" ]; then
      write_host_ip
    fi
    ip="$(tr -d '[:space:]' < "$HOST_IP_FILE")"
    is_rfc1918 "$ip" || {
      echo "Refusing to bind Duckling: host_ip is not an RFC1918 address" >&2
      exit 1
    }
    exec "${EXPORT_DIR}/duckling-example-exe" \
      -b "$ip" -p "$PORT" --no-access-log --no-error-log
    ;;
  *)
    echo "usage: duckling-bind.sh --write-host-ip|--serve" >&2
    exit 1
    ;;
esac
EOF
  sudo chown root:root "$bind_script" "$DUCKLING_EXPORT_DIR"
  sudo chmod 755 "$bind_script" "$DUCKLING_EXPORT_DIR"
  sudo chmod -R a+rX "$DUCKLING_EXPORT_DIR"
  sudo tee "$unit" >/dev/null <<EOF
[Unit]
Description=Duckling Time Parser
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=duckling
Group=duckling
WorkingDirectory=${DUCKLING_EXPORT_DIR}
Environment=LD_LIBRARY_PATH=${DUCKLING_EXPORT_DIR}/lib
Environment=DUCKLING_PORT=${DUCKLING_PORT}
Environment=DUCKLING_EXPORT_DIR=${DUCKLING_EXPORT_DIR}
ExecStartPre=+${DUCKLING_EXPORT_DIR}/duckling-bind.sh --write-host-ip
ExecStart=${DUCKLING_EXPORT_DIR}/duckling-bind.sh --serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable duckling
  sudo systemctl restart duckling
  echo "==> Duckling service listens on the host RFC1918 address only"
}

echo "==> Installing Haskell Stack if missing"
if ! command -v stack >/dev/null 2>&1; then
  install_pinned_stack
fi

echo "==> Stack version"
stack --version

echo "==> Cloning Duckling at ${DUCKLING_REV}"
mkdir -p "$HOME/workspace"

if [ ! -d "$DUCKLING_DIR/.git" ]; then
  git clone https://github.com/facebook/duckling.git "$DUCKLING_DIR"
fi
git -C "$DUCKLING_DIR" fetch --depth 1 origin "$DUCKLING_REV"
git -C "$DUCKLING_DIR" checkout --detach "$DUCKLING_REV"

echo "==> Building Duckling (this takes 30–90 minutes on a cold instance)"
cd "$DUCKLING_DIR"
stack setup
stack build

echo "==> Creating Python virtualenv"
python3 -m venv "$NLP_DIR"

echo "==> Installing Python NLP dependencies"
source "$NLP_DIR/bin/activate"
# Pin the installer tools. An unpinned "pip install --upgrade pip" would
# execute whatever release PyPI serves at bootstrap time.
python -m pip install --disable-pip-version-check \
  'pip==26.2.2' \
  'setuptools==84.2.1' \
  'wheel==0.48.0'
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
  python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
else
  # The bootstrap script is sometimes downloaded on its own. Keep the same
  # lower bounds as requirements.txt so this path cannot float to an older set.
  python -m pip install \
    'fastapi>=0.135.1' \
    'starlette>=1.3.1' \
    'uvicorn[standard]>=0.30.0' \
    'python-multipart>=0.0.31' \
    'requests>=2.33.0' \
    'urllib3>=2.8.0' \
    'spacy>=3.7.0,<3.8.0'
  install_pinned_spacy_model
fi

mkdir -p "$NLP_DIR/app"

echo "==> Creating Duckling smoke test"
cat > "$NLP_DIR/app/duckling_smoke_test.py" <<'PY'
import requests

r = requests.post(
    "http://127.0.0.1:8000/parse",
    data={
        "locale": "en_GB",
        "text": "Remind me every Monday at 9am",
        "dims": '["time","duration"]',
    },
    timeout=5,
)

print(r.status_code)
print(r.text[:1000])
r.raise_for_status()
PY

echo "==> Creating spaCy smoke test"
cat > "$NLP_DIR/app/spacy_smoke_test.py" <<'PY'
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Can you send me a digest every Friday?")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.head.text)
PY

echo "==> Starting Duckling temporarily for smoke tests (port $DUCKLING_PORT)"
cd "$DUCKLING_DIR"

DUCKLING_STARTED_HERE=false
if ! pgrep -f "duckling-example-exe" >/dev/null 2>&1; then
  # Bind the smoke-test process to loopback. The example server listens on
  # every interface unless --address is set.
  nohup stack exec -- duckling-example-exe \
    -b 127.0.0.1 -p "$DUCKLING_PORT" --no-access-log --no-error-log \
    > "$HOME/duckling.log" 2>&1 &
  DUCKLING_PID=$!
  DUCKLING_STARTED_HERE=true
  echo "Waiting for Duckling to accept connections..."
  for i in $(seq 1 60); do
    if curl -sf -X POST "http://127.0.0.1:${DUCKLING_PORT}/parse" \
        -d 'locale=en_GB&text=tomorrow&dims=["time"]' > /dev/null 2>&1; then
      echo "Duckling ready"
      break
    fi
    sleep 3
  done
else
  echo "Duckling already running"
fi

echo "==> Testing spaCy"
source "$NLP_DIR/bin/activate"
python "$NLP_DIR/app/spacy_smoke_test.py"

echo "==> Testing Duckling"
python "$NLP_DIR/app/duckling_smoke_test.py"

if $DUCKLING_STARTED_HERE; then
  echo "==> Stopping temporary Duckling instance"
  kill "$DUCKLING_PID" 2>/dev/null || true
  wait "$DUCKLING_PID" 2>/dev/null || true
fi

if $CI_MODE; then
  echo "==> Exporting Duckling binary and shared libraries to $DUCKLING_EXPORT_DIR"
  DUCKLING_BIN=$(cd "$DUCKLING_DIR" && stack exec -- which duckling-example-exe 2>/dev/null \
    || find "$DUCKLING_DIR/.stack-work" -name "duckling-example-exe" -type f | head -1)

  if [ -z "$DUCKLING_BIN" ]; then
    echo "ERROR: Could not locate duckling-example-exe binary"
    exit 1
  fi

  mkdir -p "$DUCKLING_EXPORT_DIR/lib"
  cp "$DUCKLING_BIN" "$DUCKLING_EXPORT_DIR/duckling-example-exe"
  chmod +x "$DUCKLING_EXPORT_DIR/duckling-example-exe"

  # Copy shared library dependencies so the binary runs inside a container.
  ldd "$DUCKLING_EXPORT_DIR/duckling-example-exe" \
    | awk '/=> \// { print $3 }' \
    | while read -r lib; do
        cp "$lib" "$DUCKLING_EXPORT_DIR/lib/" 2>/dev/null || true
      done

  install_duckling_service

  echo "==> Writing bootstrap marker $BOOTSTRAP_MARKER"
  mkdir -p "$(dirname "$BOOTSTRAP_MARKER")"
  echo "bootstrap-complete" > "$BOOTSTRAP_MARKER"
fi

echo
echo "Done."
echo
if ! $CI_MODE; then
  echo "Useful commands:"
  echo "  source $NLP_DIR/bin/activate"
  echo "  cd $DUCKLING_DIR && stack exec -- duckling-example-exe -b 127.0.0.1 -p $DUCKLING_PORT"
  echo "  tail -f $HOME/duckling.log"
  echo "  python $NLP_DIR/app/spacy_smoke_test.py"
  echo "  python $NLP_DIR/app/duckling_smoke_test.py"
fi
