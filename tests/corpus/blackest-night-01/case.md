# blackest-night-01

First comic run through the pipeline (Aug 2026). Output was "decent" but
surfaced two issues that this corpus locks down:

1. **Onomatopoeia handling.** Sound-effect lettering ("KRAKKA", grave-dirt
   SFX) and character vocalizations ("tsk" from Black Hand, pained "aaah"
   from a Black Lantern victim) were being fed to TTS as literal letters or
   dropped. Vocalizations should attach to the speaking character; ambient
   SFX should not be spoken as dialogue.

2. **Voice robotic-ness / flat affect** -- tracked separately in the TTS
   bake-off, not here.

## About the labels

The rows in `elements.jsonl` below are **hand-authored seed cases** built
from the issue's actual scenes. They encode the behaviour we want and are
safe to keep green. As real panels get labelled from
`work/bn01-full/panel_analysis.json`, add them here -- especially any case
where the current classifier gets it *wrong* (mark `expect_*` with the
correct answer and let the test fail until the logic is fixed; that's the
workflow).
