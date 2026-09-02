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

## Setup

Requires Python 3.11+, [Ollama](https://ollama.com), and `unrar`/`unzip` for
archive extraction. Everything below is copy-pasteable on Linux; adjust
package-manager commands for your OS where noted.

### 1. Clone this repo and set up a virtualenv

```bash
git clone https://github.com/OmegaGiven/AudioComic.git
cd AudioComic
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Archive extraction tools

```bash
# Debian/Ubuntu
sudo apt install unrar unzip
# Arch Linux
sudo pacman -S unrar unzip
# macOS (Homebrew)
brew install unrar
```

### 3. Kumiko (panel segmentation)

```bash
git clone --depth 1 https://github.com/njean42/kumiko.git tools/kumiko
```

### 4. Ollama models

```bash
# If you don't have Ollama installed yet:
curl -fsSL https://ollama.com/install.sh | sh

# Vision model (scene/dialogue/character analysis per panel)
ollama pull qwen3-vl:8b

# Narrative + voice-classification model
ollama pull devstral:24b
```

Either model can be swapped for something else your hardware runs better --
update `MODEL`/`GENDER_MODEL` at the top of `02_vision_analyze.py`,
`03_narrative.py`, and `04_tts_render.py`.

### 5. Piper TTS + voices

```bash
mkdir -p ~/.local/opt/piper && cd ~/.local/opt/piper
curl -sL -o piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
tar xzf piper.tar.gz --strip-components=1
rm piper.tar.gz

mkdir -p voices && cd voices
for voice in lessac amy kristin ljspeech hfc_female joe ryan sam norman bryce; do
  curl -sL -o "en_US-${voice}-medium.onnx" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/${voice}/medium/en_US-${voice}-medium.onnx"
  curl -sL -o "en_US-${voice}-medium.onnx.json" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/${voice}/medium/en_US-${voice}-medium.onnx.json"
done
```

This installs to `~/.local/opt/piper` -- the path `04_tts_render.py` expects
by default (`PIPER_BIN`/`VOICES_DIR` at the top of that file). Installing
elsewhere just means updating those two constants. The voice list above
matches `VOICE_POOLS` in the script; swap in any other
[Piper voice](https://huggingface.co/rhasspy/piper-voices) you prefer, and
update `VOICE_POOLS` to match.

## Usage

```bash
python3 01_extract_and_segment.py path/to/issue.cbr work/issue-01
python3 02_vision_analyze.py work/issue-01
python3 03_narrative.py work/issue-01
python3 04_tts_render.py work/issue-01 output/issue-01.wav          # Piper
# or, the Kokoro variant (more natural, needs `pip install -r requirements-kokoro.txt`):
python3 04_tts_render_kokoro.py work/issue-01 output/issue-01.wav
```

Stage 4 has two implementations kept in parallel so their output can be
compared: **`04_tts_render.py`** (Piper, the original) and
**`04_tts_render_kokoro.py`** (Kokoro-82M, chosen in the TTS bake-off --
more natural prosody, still fast, plus onomatopoeia cleanup via
`panelspeak`). They share the `voice_map.json` format but store different
voice ids, so use separate work dirs (or delete `voice_map.json`) when
switching engines on the same issue.

Each phase is resumable -- rerunning a script picks up where it left off
(tracked in `manifest.json`, `panel_analysis.json`, `narrative.json`, and
`voice_map.json` inside the work directory) rather than redoing completed
work.

## Development

`panelspeak/` holds the deterministic logic that wraps the model calls
(onomatopoeia lexicon, panel-text classification, vocalization attribution,
the emotion-hint line format). It has no model or network dependency.

```bash
pip install -r requirements-dev.txt
pytest                       # unit + classification + corpus + regression
pytest -m llm tests/eval     # opt-in model-quality checks, needs a live stack
```

`tests/README.md` explains the layout and, importantly, the workflow for
**adding a regression case every time a new comic surfaces a problem** --
that's how the tool is expected to improve.

## Notes

- The vision-analysis and narrative-generation phases are the slow parts
  (real GPU inference per panel/page) -- expect anywhere from tens of
  minutes to a few hours for a full issue depending on your hardware and
  panel count. TTS rendering (Piper, CPU-only) is fast by comparison.
- This is a local-first, no-paid-services pipeline by design -- everything
  runs on hardware you control.
