#!/usr/bin/env bash
# ComicDB pipeline, end to end.
#   pipeline/run.sh <issue.cbz> <work_dir> <out.wav> [--from <phase>]
# phases: segment transcribe identify resolve redescribe assemble render
set -euo pipefail

CBZ="$1"; WORK="$2"; OUT="$3"; shift 3
FROM="segment"
[ "${1:-}" = "--from" ] && FROM="$2"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

SEG_PY="${SEG_PY:-$REPO/.venv312/bin/python}"        # Pillow + opencv (Kumiko + vision)
MAGI_PY="${MAGI_PY:-$REPO/.venv-magi/bin/python}"    # transformers + torch 2.8
KOKORO_PY="${KOKORO_PY:-$REPO/bakeoff/.venvs/kokoro/bin/python}"
OLLAMA="${OLLAMA:-docker exec ollama ollama}"
[ -x "$SEG_PY" ] || SEG_PY=python3

order=(segment transcribe identify resolve redescribe assemble render)
started=0
run_phase() { [ "$1" = "$FROM" ] && started=1; [ "$started" = 1 ]; }

if run_phase segment; then
  echo "== segment =="; "$SEG_PY" -m pipeline.segment "$CBZ" "$WORK"
fi
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
if run_phase assemble; then
  echo "== assemble =="; PYTHONPATH="$REPO" "$SEG_PY" -m pipeline.assemble "$WORK"
fi
if run_phase render; then
  echo "== render (Kokoro, CPU) =="
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
    "$KOKORO_PY" "$REPO/scripts/04_tts_render_kokoro.py" "$WORK" "$OUT"
fi
echo "Done -> $OUT"
