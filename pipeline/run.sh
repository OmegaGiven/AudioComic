#!/usr/bin/env bash
# ComicDB pipeline, end to end.
#   pipeline/run.sh <issue.cbz> <work_dir> <out.wav> [--from <phase>]
#   VISION=claude pipeline/run.sh ...   -- use the Claude API instead of the
#                                          local qwen2.5vl model (needs
#                                          ANTHROPIC_API_KEY; real per-page cost)
# phases (VISION=local, default): segment transcribe identify resolve redescribe assemble narrate render publish
# phases (VISION=claude):         segment claude_extract assemble narrate render publish
set -euo pipefail

CBZ="$1"; WORK="$2"; OUT="$3"; shift 3
FROM="segment"
[ "${1:-}" = "--from" ] && FROM="$2"
VISION="${VISION:-local}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

SEG_PY="${SEG_PY:-$REPO/.venv312/bin/python}"        # Pillow + opencv (Kumiko + vision)
MAGI_PY="${MAGI_PY:-$REPO/.venv-magi/bin/python}"    # transformers + torch 2.8
KOKORO_PY="${KOKORO_PY:-$REPO/bakeoff/.venvs/kokoro/bin/python}"
OLLAMA="${OLLAMA:-docker exec ollama ollama}"
[ -x "$SEG_PY" ] || SEG_PY=python3

order=(segment transcribe identify resolve redescribe assemble narrate render publish)
started=0
run_phase() { [ "$1" = "$FROM" ] && started=1; [ "$started" = 1 ]; }

if run_phase segment; then
  echo "== segment =="; "$SEG_PY" -m pipeline.segment "$CBZ" "$WORK"
fi
if [ "$VISION" = "claude" ]; then
  if run_phase claude_extract; then
    echo "== claude_extract =="
    [ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "ANTHROPIC_API_KEY not set" >&2; exit 3; }
    PYTHONPATH="$REPO" "$SEG_PY" -m pipeline.claude_extract "$WORK"
  fi
else
  if run_phase transcribe; then
    echo "== transcribe =="; PYTHONPATH="$REPO" "$SEG_PY" -m pipeline.transcribe "$WORK"
  fi
  if run_phase identify; then
    echo "== identify (Magi) =="
    for m in $($OLLAMA ps 2>/dev/null | awk 'NR>1{print $1}'); do $OLLAMA stop "$m" || true; done
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONPATH="$REPO" "$MAGI_PY" -m pipeline.identify "$WORK"
  fi
  if run_phase resolve; then
    echo "== resolve =="; PYTHONPATH="$REPO" "$SEG_PY" -m pipeline.resolve "$WORK"
  fi
  if run_phase redescribe; then
    echo "== redescribe =="; PYTHONPATH="$REPO" "$SEG_PY" -m pipeline.redescribe "$WORK"
  fi
fi
if run_phase assemble; then
  echo "== assemble =="; PYTHONPATH="$REPO" "$SEG_PY" -m pipeline.assemble "$WORK"
fi
if run_phase narrate; then
  echo "== narrate =="
  # free the VRAM the vision model held through redescribe so Ollama can
  # load the text model -- otherwise every narrate call silently gets an
  # empty response and the page keeps its rough deterministic narration
  for m in $($OLLAMA ps 2>/dev/null | awk 'NR>1{print $1}'); do $OLLAMA stop "$m" || true; done
  PYTHONPATH="$REPO" "$SEG_PY" -m pipeline.narrate "$WORK" || echo "  narrate skipped (not fatal)"
fi
if run_phase render; then
  TTS_ENGINE="${TTS_ENGINE:-kokoro}"
  if [ "$TTS_ENGINE" = "dia" ]; then
    echo "== render (Dia, GPU) =="
    DIA_PY="${DIA_PY:-$REPO/.venv-dia/bin/python}"
    for m in $($OLLAMA ps 2>/dev/null | awk 'NR>1{print $1}'); do $OLLAMA stop "$m" || true; done
    PYTHONPATH="$REPO" "$DIA_PY" "$REPO/scripts/04_tts_render_dia.py" "$WORK" "$OUT"
  else
    echo "== render (Kokoro, CPU) =="
    CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
      "$KOKORO_PY" "$REPO/scripts/04_tts_render_kokoro.py" "$WORK" "$OUT"
  fi
fi
if run_phase publish; then
  echo "== publish (media library) =="
  PYTHONPATH="$REPO" "$SEG_PY" -m pipeline.publish "$WORK" "$OUT" || echo "  publish failed (not fatal)"
fi
echo "Done -> $OUT"
