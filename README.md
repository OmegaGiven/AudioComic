# AudioComic

A local, self-hosted pipeline that converts a comic book (`.cbr`/`.cbz`) into
a multi-voice audiobook. No cloud APIs -- everything runs on local models via
[Ollama](https://ollama.com) and [Piper TTS](https://github.com/rhasspy/piper).

## Pipeline

1. **`01_extract_and_segment.py`** -- extracts the archive, detects and
   filters non-page junk files (e.g. release-group credit images), detects
   double-page spreads by aspect ratio, and segments every page into panels
   (in reading order) using [Kumiko](https://github.com/njean42/kumiko).
2. **`02_vision_analyze.py`** -- per-panel vision analysis (scene
   description, character identification, dialogue/caption OCR) via a local
   vision-language model (`qwen3-vl:8b` by default). Resumable --
   checkpoints after every panel.
3. **`03_narrative.py`** -- turns the raw per-panel analysis into flowing,
   in-story audiobook narration (not a dry panel-by-panel recap) via a local
   LLM (`devstral:24b` by default), page by page. Includes a fidelity check
   that retries a page if it looks like real content got dropped.
4. **`04_tts_render.py`** -- multi-voice text-to-speech rendering via Piper.
   Classifies each named character's gender and age-tone via the LLM and
   assigns a consistent voice for the whole issue (saved to
   `voice_map.json`, reviewable/editable by hand), then concatenates every
   segment into one final audiobook file.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally, with a vision-capable model
  (`ollama pull qwen3-vl:8b`) and a general-purpose model for narrative/
  classification (`ollama pull devstral:24b` -- or substitute any model that
  fits your hardware; update `MODEL`/`GENDER_MODEL` at the top of each
  script).
- [Kumiko](https://github.com/njean42/kumiko) cloned locally for panel
  segmentation:
  ```
  git clone https://github.com/njean42/kumiko.git tools/kumiko
  ```
- [Piper TTS](https://github.com/rhasspy/piper) with one or more voice
  models downloaded (see [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)
  on Hugging Face). The default voice pool in `04_tts_render.py` expects:
  `lessac`, `amy`, `kristin`, `ljspeech`, `hfc_female` (medium quality) for
  narrator/female voices, and `joe`, `ryan`, `sam`, `norman`, `bryce`
  (medium quality) for male voices -- adjust `VOICE_POOLS` to whatever
  voices you have.
- `unrar` (for `.cbr`) and/or `unzip` (for `.cbz`).
- `pip install -r requirements.txt`

## Usage

```bash
python3 01_extract_and_segment.py path/to/issue.cbr work/issue-01
python3 02_vision_analyze.py work/issue-01
python3 03_narrative.py work/issue-01
python3 04_tts_render.py work/issue-01 output/issue-01.wav
```

Each phase is resumable -- rerunning a script picks up where it left off
(tracked in `manifest.json`, `panel_analysis.json`, `narrative.json`, and
`voice_map.json` inside the work directory) rather than redoing completed
work.

## Notes

- The vision-analysis and narrative-generation phases are the slow parts
  (real GPU inference per panel/page) -- expect anywhere from tens of
  minutes to a few hours for a full issue depending on your hardware and
  panel count. TTS rendering (Piper, CPU-only) is fast by comparison.
- This is a local-first, no-paid-services pipeline by design -- everything
  runs on hardware you control.
