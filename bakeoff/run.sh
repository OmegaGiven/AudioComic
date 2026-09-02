#!/usr/bin/env bash
# TTS bake-off runner.
#
#   ./bakeoff/run.sh                 # tier-1 engines: piper kokoro chatterbox
#   ./bakeoff/run.sh all             # everything, incl. orpheus + vibevoice
#   ./bakeoff/run.sh kokoro orpheus  # just these
#   ./bakeoff/run.sh page            # only (re)build the comparison page
#
# Each engine gets its own venv under bakeoff/.venvs/<engine> so their deps
# never collide. If `uv` is on PATH it's used (and pins Python 3.12, since
# the ML stack has no 3.14 wheels yet); otherwise falls back to python -m venv.
# Models download on first run (cached by HuggingFace after). Reference voice
# clips are optional: drop wavs in bakeoff/refs/ named to match passage.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENVS="$HERE/.venvs"
PYPIN="${PYPIN:-3.12}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TIER1=(piper kokoro chatterbox)
TIER2=(orpheus vibevoice)

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[[ -x "$UV" ]] || UV=""

declare -A DEPS=(
  [piper]=""
  [kokoro]="kokoro soundfile numpy"
  [chatterbox]="chatterbox-tts torchaudio"
  [orpheus]="transformers snac soundfile numpy accelerate"
  [vibevoice]="vibevoice-community soundfile accelerate"
)
declare -A NEEDS_TORCH=([kokoro]=1 [chatterbox]=1 [orpheus]=1 [vibevoice]=1)

want=()
case "${1:-tier1}" in
  ""|tier1)  want=("${TIER1[@]}") ;;
  all)       want=("${TIER1[@]}" "${TIER2[@]}") ;;
  page)      want=() ;;
  *)         want=("$@") ;;
esac

mkdir -p "$HERE/refs" "$HERE/out"

mk_venv() {
  local venv="$1"
  if [[ -n "$UV" ]]; then
    "$UV" venv -q --python "$PYPIN" "$venv"
  else
    python3 -m venv "$venv"
    "$venv/bin/pip" -q install --upgrade pip
  fi
}
pip_install() {
  local venv="$1"; shift
  if [[ -n "$UV" ]]; then
    VIRTUAL_ENV="$venv" "$UV" pip install -q "$@"
  else
    "$venv/bin/pip" -q install "$@"
  fi
}

run_engine() {
  local eng="$1"
  local venv="$VENVS/$eng"
  echo "=== $eng ==="
  if [[ ! -x "$venv/bin/python" ]]; then
    mk_venv "$venv"
    if [[ -n "${NEEDS_TORCH[$eng]:-}" ]]; then
      pip_install "$venv" --index-url "$TORCH_INDEX" torch || pip_install "$venv" torch
    fi
    if [[ -n "${DEPS[$eng]}" ]]; then
      # shellcheck disable=SC2086
      pip_install "$venv" ${DEPS[$eng]}
    fi
  fi
  ( cd "$HERE" && PYTHONPATH="$HERE" "$venv/bin/python" "engines/${eng}_engine.py" )
}

for eng in "${want[@]}"; do
  if run_engine "$eng"; then :; else
    echo "!!! $eng failed -- continuing with the rest" >&2
  fi
done

( cd "$HERE" && PYTHONPATH="$HERE" python3 build_page.py )

echo
echo "Done. Open:  $HERE/out/index.html"
