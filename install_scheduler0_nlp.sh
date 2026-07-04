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
#   NLP_DIR       – Python virtualenv root          (default: $HOME/scheduler0-nlp)
#   DUCKLING_PORT – Duckling HTTP port              (default: 8000)
#   DUCKLING_EXPORT_DIR – where to copy the binary and libs for ECS mounting
#                                                   (default: /opt/duckling)
#   LOG_FILE      – log destination (--ci mode)     (default: /var/log/scheduler0-intent-classifier-bootstrap.log)
#   BOOTSTRAP_MARKER – file written on success      (default: /var/lib/scheduler0-intent-classifier/bootstrap-complete)

set -Eeuo pipefail
# HOME is not set in the EC2 user-data environment; default to /root so the
# ${HOME:-...} expansions below don't trip the -u (nounset) flag.
export HOME="${HOME:-/root}"

CI_MODE=false
for arg in "$@"; do
  [ "$arg" = "--ci" ] && CI_MODE=true
done

DUCKLING_DIR="${DUCKLING_DIR:-$HOME/workspace/duckling}"
NLP_DIR="${NLP_DIR:-$HOME/scheduler0-nlp}"
DUCKLING_PORT="${DUCKLING_PORT:-8000}"
DUCKLING_EXPORT_DIR="${DUCKLING_EXPORT_DIR:-/opt/duckling}"
LOG_FILE="${LOG_FILE:-/var/log/scheduler0-intent-classifier-bootstrap.log}"
BOOTSTRAP_MARKER="${BOOTSTRAP_MARKER:-/var/lib/scheduler0-intent-classifier/bootstrap-complete}"

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
python3 -m pip install --upgrade pip --no-cache-dir

echo "==> Installing Haskell Stack if missing"
if ! command -v stack >/dev/null 2>&1; then
  curl -sSL https://get.haskellstack.org/ | sh
fi

echo "==> Stack version"
stack --version

echo "==> Cloning Duckling"
mkdir -p "$HOME/workspace"

if [ ! -d "$DUCKLING_DIR/.git" ]; then
  git clone https://github.com/facebook/duckling.git "$DUCKLING_DIR"
else
  echo "Duckling repo already exists at $DUCKLING_DIR"
fi

echo "==> Building Duckling (this takes 30–90 minutes on a cold instance)"
cd "$DUCKLING_DIR"
stack setup
stack build

echo "==> Creating Python virtualenv"
python3 -m venv "$NLP_DIR"

echo "==> Installing Python NLP dependencies"
source "$NLP_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install requests spacy
python -m spacy download en_core_web_sm

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
  nohup stack exec duckling-example-exe > "$HOME/duckling.log" 2>&1 &
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
  echo "  cd $DUCKLING_DIR && stack exec duckling-example-exe"
  echo "  tail -f $HOME/duckling.log"
  echo "  python $NLP_DIR/app/spacy_smoke_test.py"
  echo "  python $NLP_DIR/app/duckling_smoke_test.py"
fi
