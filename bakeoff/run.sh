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
  [chatterbox]="chatterbox-tts soundfile"
  [orpheus]="transformers snac soundfile numpy accelerate"
  [vibevoice]="vibevoice-community soundfile accelerate"
)
# these import torch and run on the GPU -> need the cu128 (Blackwell/sm_120) build
declare -A NEEDS_TORCH=([chatterbox]=1 [orpheus]=1 [vibevoice]=1)
declare -A NEEDS_TORCHAUDIO=([chatterbox]=1 [vibevoice]=1)
# per-engine fixups run after deps are installed ($V = venv path)
declare -A POSTSETUP=(
  [kokoro]='"$V/bin/python" -m spacy download en_core_web_sm'
)

want=()
case "${1:-tier1}" in
  ""|tier1)  want=("${TIER1[@]}") ;;
  all)       want=("${TIER1[@]}" "${TIER2[@]}") ;;
  page)      want=() ;;
  *)         want=("$@") ;;
esac

mkdir -p "$HERE/refs" "$HERE/out" "$VENVS"

mk_venv() {
  local venv="$1"
  if [[ -n "$UV" ]]; then
    "$UV" venv -q --clear --python "$PYPIN" "$venv"
  else
    rm -rf "$venv"
    python3 -m venv "$venv"
    "$venv/bin/pip" -q install --upgrade pip
  fi
}
pip_install() {
  local venv="$1"; shift
  if [[ -n "$UV" ]]; then
    "$UV" pip install -q --python "$venv/bin/python" "$@"
  else
    "$venv/bin/pip" -q install "$@"
  fi
}

run_engine() {
  local eng="$1"
  local venv="$VENVS/$eng"
  echo "=== $eng ==="
  if [[ ! -e "$venv/.ready" ]]; then
    mk_venv "$venv"
    pip_install "$venv" pip setuptools wheel
    if [[ -n "${DEPS[$eng]}" ]]; then
      # shellcheck disable=SC2086
      pip_install "$venv" ${DEPS[$eng]}
    fi
    if [[ -n "${NEEDS_TORCH[$eng]:-}" ]]; then
      # engine deps often pull a PyPI torch that lacks Blackwell kernels;
      # force the cu128 build in last so it wins.
      local ta=""
      [[ -n "${NEEDS_TORCHAUDIO[$eng]:-}" ]] && ta="torchaudio"
      # shellcheck disable=SC2086
      pip_install "$venv" --upgrade --force-reinstall --index-url "$TORCH_INDEX" torch $ta
      "$venv/bin/python" - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.version.cuda, "ok", torch.cuda.is_available())
sys.exit(0 if torch.cuda.is_available() else 1)
PY
    fi
    local V="$venv"
    [[ -n "${POSTSETUP[$eng]:-}" ]] && eval "${POSTSETUP[$eng]}"
    touch "$venv/.ready"
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
