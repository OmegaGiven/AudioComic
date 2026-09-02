# TTS bake-off

Renders one fixed passage (`passage.json`) through several local TTS engines
and builds a single side-by-side comparison page so we can pick the one that
sounds least robotic and carries emotion + onomatopoeia best.

## Run it (on the box with the GPU)

```bash
./bakeoff/run.sh            # tier 1: piper, kokoro, chatterbox
./bakeoff/run.sh all        # + orpheus, vibevoice (heavier)
./bakeoff/run.sh page       # rebuild the page from existing renders
open bakeoff/out/index.html
```

Each engine installs into its own venv under `bakeoff/.venvs/`. Models
download on first run and are cached by HuggingFace afterwards. `ffmpeg` on
PATH keeps the page small (audio -> opus); without it the page still builds
but is much larger.

## Reference voice clips (optional but recommended)

The cloning engines (Chatterbox, VibeVoice) sound far better with a 6-15s
clean reference clip per speaker. Drop wavs in `bakeoff/refs/`:

| file | speaker | suggested source |
|---|---|---|
| `narrator.wav`    | NARRATOR   | a calm audiobook narrator sample |
| `black_hand.wav`  | BLACK HAND | a low, quiet, menacing read |
| `hal_jordan.wav`  | HAL JORDAN | an urgent, heroic read |
| `mera.wav`        | MERA       | a sharp, commanding female read |

Without refs, each engine uses its own built-in voice for every speaker
(still a valid quality comparison, just not distinct characters).

## The candidates

| engine | license | ~VRAM | emotion | cloning | why it's here |
|---|---|---|---|---|---|
| **piper**      | MIT       | CPU     | none | none | current pipeline baseline -- the bar to beat |
| **kokoro**     | Apache-2  | 2-3 GB  | none | none | fast/tiny; the "draft preview" tier |
| **chatterbox** | MIT       | 6-8 GB  | exaggeration knob | zero-shot | best per-character voice bank + emotion dial |
| **orpheus**    | Apache-2  | 8-12 GB | inline `<sigh>`/`<laugh>` tags | voice-by-name | native non-verbal tags = onomatopoeia |
| **vibevoice**  | MIT       | 7-24 GB | tags + turn-taking | reference clips | whole passage in ONE pass -> cross-line prosody |

## Output

```
bakeoff/out/
  index.html            <- open this
  <engine>/
    full.wav / .opus     full passage
    seg00.wav ...        per line
    manifest.json
```

The page scores each engine on naturalness, emotion, character
distinctiveness, onomatopoeia handling, seams, and speed/VRAM.

## Passage

`passage.json` is fixed on purpose -- keep it stable so renders stay
comparable across engines and across time. It packs neutral narration, three
distinct voices, a menacing line, an urgent line, a shout, a pained scream
(`Aaaah!`), and a weary `*sigh*` line, so every axis gets exercised in ~110
words.
