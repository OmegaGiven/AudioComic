#!/usr/bin/env python3
"""02_vision_analyze.py <work_dir>

Slow phase: per-panel vision analysis (scene description + dialogue OCR +
character attribution) via qwen3-vl:8b. Resumable -- checkpoints after every
panel to panel_analysis.json, skips panels already done on rerun.

Make sure your GPU has enough free VRAM for the vision model before starting
a long run (unload/close anything else using it first) -- this script talks
to Ollama directly with no VRAM-contention handling of its own. Because it
checkpoints after every single panel, if something else on the machine
claims the GPU mid-run and a call fails, rerunning the script just redoes
the interrupted panel, not the whole batch.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3-vl:8b"

PROMPT_TEMPLATE = """Describe this comic panel in 2-3 sentences: the scene, characters present, and their actions/expressions. Name specific characters if you recognize them.
Then transcribe every piece of dialogue or caption text visible in the panel, in reading order, formatted exactly as:
SPEAKER: text
or
CAPTION: text
(if no clear speaker, e.g. narration boxes)

Respond directly with the description and transcription -- do not explain your reasoning process."""


def analyze_panel(image_path: str) -> dict:
    import base64
    img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    payload = {
        "model": MODEL,
        "prompt": PROMPT_TEMPLATE,
        "images": [img_b64],
        "stream": False,
        "think": False,
        "options": {
            "num_predict": 1500,
            # Default context (262144) is wildly oversized for one image +
            # a short prompt, and grows the KV cache every consecutive call
            # within the same long-lived server process -- confirmed cause
            # of a real 18x throughput collapse (9 t/s -> 0.5 t/s) over a
            # ~90 min run (2026-08-25). 4096 is generous for this workload
            # and keeps cache pressure flat across hundreds of calls.
            # 4096 was too tight -- a single large panel crop alone can
            # tokenize to ~4000 image tokens (confirmed: prompt_eval_count
            # 4013 on a 1824x2494 crop), leaving no room for any response
            # and silently truncating to empty output. 16384 covers the
            # worst-case image + prompt + full 1500-token response with
            # margin, while still being far below the 262144 default that
            # caused the original throughput collapse.
            "num_ctx": 16384,
        },
    }
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", "300", OLLAMA_URL, "--data-binary", "@-"],
            input=json.dumps(payload), capture_output=True, text=True, timeout=310,
        )
    except subprocess.TimeoutExpired:
        return {"text": "", "error": "subprocess timeout (curl exceeded 310s)"}

    try:
        d = json.loads(r.stdout.strip().split("\n")[0])
    except Exception as e:
        return {"text": "", "error": f"parse failure: {e}", "raw": r.stdout[:500]}

    text = d.get("response", "").strip()
    if not text:
        # Model didn't honor think:false and used the whole budget on
        # reasoning -- the useful content is still in there, just unformatted.
        thinking = d.get("thinking", "")
        text = thinking.strip()
    return {
        "text": text,
        "done_reason": d.get("done_reason"),
        "eval_count": d.get("eval_count"),
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: 02_vision_analyze.py <work_dir>", file=sys.stderr)
        sys.exit(2)

    work_dir = Path(sys.argv[1])
    manifest = json.load(open(work_dir / "manifest.json"))
    results_path = work_dir / "panel_analysis.json"

    results = {}
    if results_path.exists():
        results = json.load(open(results_path))

    all_panels = []
    for page in manifest["pages"]:
        for panel in page["panels"]:
            key = f"page{page['page_index']:03d}_panel{panel['panel_index']:02d}"
            all_panels.append((key, panel["file"]))

    # A key existing in results is only "done" if it actually produced text --
    # error stubs from a prior crashed/timed-out run must be retried, not
    # silently accepted as complete (real bug hit in the first run: 6 panels
    # saved as {"text": "", "error": ...} were never going to be retried).
    todo = [(k, f) for k, f in all_panels if not results.get(k, {}).get("text")]
    print(f"{len(all_panels)} total panels, {len(todo)} remaining.")

    for i, (key, file) in enumerate(todo):
        t0 = time.time()
        try:
            result = analyze_panel(file)
        except Exception as e:
            # Never let one bad panel kill an hours-long batch job -- log it,
            # move on, it'll retry on the next run same as a timeout does.
            result = {"text": "", "error": f"unhandled exception: {e}"}

        results[key] = result
        json.dump(results, open(results_path, "w"), indent=2)

        elapsed = time.time() - t0
        status = "OK" if result.get("text") else f"FAILED: {result.get('error')}"
        print(f"[{i+1}/{len(todo)}] {key} done in {elapsed:.0f}s "
              f"({result.get('eval_count', '?')} tokens, reason={result.get('done_reason')}) {status}")

        # Periodic model refresh as a safety net against any residual
        # cache-growth degradation beyond what num_ctx=4096 already fixes --
        # cheap insurance for a run that's going to take hours regardless.
        if (i + 1) % 20 == 0:
            print("  Refreshing model (periodic unload/reload)...")
            subprocess.run(
                ["curl", "-s", "-X", "POST", OLLAMA_URL,
                 "-d", json.dumps({"model": MODEL, "keep_alive": 0})],
                capture_output=True, timeout=30,
            )
            time.sleep(2)

    print(f"All done. Results: {results_path}")


if __name__ == "__main__":
    main()
