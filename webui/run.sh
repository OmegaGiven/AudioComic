#!/usr/bin/env bash
# AudioComic Studio dev server.
#   webui/run.sh            -> http://0.0.0.0:8971
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

PY="${PY:-$REPO/.venv-web/bin/python}"
if [ ! -x "$PY" ]; then
  "${PYTHON:-python3}" -m venv "$REPO/.venv-web"
  PY="$REPO/.venv-web/bin/python"
  "$PY" -m pip -q install --upgrade pip
  "$PY" -m pip -q install -r "$HERE/requirements.txt"
fi
exec "$PY" -m uvicorn webui.app:app --host 0.0.0.0 --port "${PORT:-8971}" "$@"
