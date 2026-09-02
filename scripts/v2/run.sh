#!/usr/bin/env bash
# Pipeline v2 end to end.
#   scripts/v2/run.sh <issue.cbz> <work_dir> <out.wav> [char_bank_dir]
#
# Stage A (Magi) and stages B/C (Ollama) can't share the 16 GB card, so this
# unloads Ollama models before A and lets Ollama reload for B/C. Stage D
# (Kokoro) runs on CPU.
#
# Expects venvs:
#   $MAGI_PY   - python with transformers + torch 2.8 + torchvision 0.23  (Magi)
#   $KOKORO_PY - python with kokoro + soundfile                            (stage D)
set -euo pipefail

CBZ="$1"; WORK="$2"; OUT="$3"; BANK="${4:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

MAGI_PY="${MAGI_PY:-$REPO/.venv-magi/bin/python}"
KOKORO_PY="${KOKORO_PY:-$REPO/bakeoff/.venvs/kokoro/bin/python}"
# B/C only need PIL + stdlib (they shell out to curl); reuse the pipeline venv
BC_PY="${BC_PY:-$REPO/.venv312/bin/python}"
[ -x "$BC_PY" ] || BC_PY=python3
OLLAMA="${OLLAMA:-docker exec ollama ollama}"

mkdir -p "$(dirname "$OUT")" "$WORK"

echo "== A: structure (Magi) =="
for m in $($OLLAMA ps 2>/dev/null | awk 'NR>1{print $1}'); do $OLLAMA stop "$m" || true; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$MAGI_PY" "$HERE/a_structure_magi.py" "$CBZ" "$WORK" $BANK

echo "== B: describe (VLM) =="
PYTHONPATH="$REPO" "$BC_PY" "$HERE/b_describe.py" "$WORK"

echo "== C: narrative (Nemo) =="
PYTHONPATH="$REPO" "$BC_PY" "$HERE/c_narrative.py" "$WORK"

echo "== D: TTS (Kokoro, CPU) =="
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
  "$KOKORO_PY" "$REPO/scripts/04_tts_render_kokoro.py" "$WORK" "$OUT"

echo "Done -> $OUT"
