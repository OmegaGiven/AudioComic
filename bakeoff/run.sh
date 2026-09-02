#!/usr/bin/env bash
# TTS bake-off runner.
#
#   ./bakeoff/run.sh                 # tier-1 engines: piper kokoro chatterbox
#   ./bakeoff/run.sh all             # everything, incl. orpheus + vibevoice
#   ./bakeoff/run.sh kokoro orpheus  # just these
#   ./bakeoff/run.sh page            # only (re)build the comparison page
#
# Each engine gets its own venv under bakeoff/.venvs/<engine> so their deps
# never collide. Models download on first run (cached by HuggingFace after).
# Reference voice clips are optional: drop wavs in bakeoff/refs/ named to match
# passage.json ("black_hand.wav", "hal_jordan.wav", "mera.wav", "narrator.wav").
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENVS="$HERE/.venvs"
PYBASE="${PYTHON:-python3}"
TIER1=(piper kokoro chatterbox)
TIER2=(orpheus vibevoice)

declare -A DEPS=(
  [piper]=""                                             # uses ~/.local/opt/piper
  [kokoro]="kokoro soundfile numpy"
  [chatterbox]="chatterbox-tts torchaudio"
  [orpheus]="transformers snac torch soundfile numpy accelerate"
  [vibevoice]="vibevoice-community soundfile accelerate"
)

want=()
case "${1:-tier1}" in
  ""|tier1)  want=("${TIER1[@]}") ;;
  all)       want=("${TIER1[@]}" "${TIER2[@]}") ;;
  page)      want=() ;;
  *)         want=("$@") ;;
esac

mkdir -p "$HERE/refs" "$HERE/out"

run_engine() {
  local eng="$1"
  local venv="$VENVS/$eng"
  echo "=== $eng ==="
  if [[ ! -d "$venv" ]]; then
    "$PYBASE" -m venv "$venv"
    "$venv/bin/pip" -q install --upgrade pip
    if [[ -n "${DEPS[$eng]}" ]]; then
      # torch first so the others resolve against the installed CUDA build
      if [[ "${DEPS[$eng]}" == *torch* ]]; then
        "$venv/bin/pip" -q install torch --index-url https://download.pytorch.org/whl/cu124 || \
          "$venv/bin/pip" -q install torch
      fi
      # shellcheck disable=SC2086
      "$venv/bin/pip" -q install ${DEPS[$eng]}
    fi
  fi
  ( cd "$HERE" && "$venv/bin/python" "engines/$eng.py" )
}

for eng in "${want[@]}"; do
  if run_engine "$eng"; then :; else
    echo "!!! $eng failed -- continuing with the rest" >&2
  fi
done

# build the page with whatever engine venv is handy (needs only stdlib + _common)
PAGE_PY="$PYBASE"
for e in "${TIER1[@]}" "${TIER2[@]}"; do
  [[ -x "$VENVS/$e/bin/python" ]] && PAGE_PY="$VENVS/$e/bin/python" && break
done
( cd "$HERE" && "$PAGE_PY" build_page.py )

echo
echo "Done. Open:  $HERE/out/index.html"
