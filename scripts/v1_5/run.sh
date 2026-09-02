#!/usr/bin/env bash
# Pipeline v1.5 end to end.
#   scripts/v1_5/run.sh <issue.cbz> <work_dir> <out.wav>
#
# 1  extract + panel segmentation (Kumiko)          -> manifest.json
# 2  per-panel transcribe + brief scene (qwen3-vl)  -> transcript.json
# 3  deterministic assembly (no LLM)                -> narrative.json
# 4  multi-voice TTS (Kokoro, CPU)                  -> out.wav
set -euo pipefail

CBZ="$1"; WORK="$2"; OUT="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

SEG_PY="${SEG_PY:-$REPO/.venv312/bin/python}"          # Pillow + opencv (Kumiko)
KOKORO_PY="${KOKORO_PY:-$REPO/bakeoff/.venvs/kokoro/bin/python}"
[ -x "$SEG_PY" ] || SEG_PY=python3

mkdir -p "$(dirname "$OUT")" "$WORK"

echo "== 1: extract + segment =="
"$SEG_PY" "$REPO/scripts/01_extract_and_segment.py" "$CBZ" "$WORK"

echo "== 2: transcribe + scene (qwen3-vl) =="
PYTHONPATH="$REPO" "$SEG_PY" "$HERE/2_transcribe.py" "$WORK"

echo "== 3: assemble (deterministic) =="
PYTHONPATH="$REPO" "$SEG_PY" "$HERE/3_assemble.py" "$WORK"

echo "== 4: TTS (Kokoro, CPU) =="
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
  "$KOKORO_PY" "$REPO/scripts/04_tts_render_kokoro.py" "$WORK" "$OUT"

echo "Done -> $OUT"
