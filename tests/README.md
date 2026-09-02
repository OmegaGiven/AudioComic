# Tests

The pipeline is model-driven and non-deterministic. The strategy is to push
as much logic as possible into deterministic, pure functions (in
`panelspeak/`), test those hard, and keep the model calls behind a thin seam
that can be faked.

```
pip install -r requirements-dev.txt
pytest                      # everything except model/network tests
pytest -m llm tests/eval    # opt-in model-quality checks (needs a live stack)
```

## Layout

| Dir | What | In CI |
|---|---|---|
| `tests/unit/` | Characterization tests for the existing `scripts/` functions. Lock current behaviour so the refactor is safe. | yes |
| `tests/onomatopoeia/` | The lexicon, element classifier, and vocalization attribution (`panelspeak.onomatopoeia` / `classify` / `attribution`). | yes |
| `tests/format/` | Emotion-hint line format and vocalization merging (`panelspeak.emotion`). | yes |
| `tests/providers/` | The `VisionModel` / `ChatModel` / `TTSProvider` contract, run against fakes always and real endpoints when env vars are set. | yes (fakes) |
| `tests/corpus/` | **The regression corpus.** Labelled panel elements from real comics. Grows over time. | yes |
| `tests/regression/` | One test per already-fixed bug. | yes |
| `tests/eval/` | Model-quality evaluations. Slow, opt-in, never gate a merge. | no |

## When a new comic reveals a problem

This is expected — every issue we run finds something. The workflow:

1. **If it's a mis-classification or mis-attribution** (a sound effect read as
   dialogue, a `tsk` put on the wrong character):
   - `mkdir tests/corpus/<comic-slug>/`
   - add a `case.md` describing what broke
   - add lines to `tests/corpus/<comic-slug>/elements.jsonl` — one per text
     element, with `expect_kind` / `expect_speaker` set to the **correct**
     answer (schema in `tests/corpus/_schema.py`)
   - run `pytest tests/corpus`. If the new rows fail, the classifier is
     wrong — fix `panelspeak/` until they pass. If they already pass, good:
     you've just locked the behaviour in.

2. **If it's a new crash or a wrong-output bug in a script**: fix it, then add
   a test to `tests/regression/test_known_bugs.py` that fails without the fix.

3. **If it's model quality** (flat narration, wrong emotion, robotic TTS): that
   belongs in `tests/eval/` as a measurement, and in the bake-off notes — not
   as a hard gate.

## Accuracy floors

`tests/corpus/test_corpus.py` has `KIND_ACCURACY_FLOOR` and
`SPEAKER_ACCURACY_FLOOR`. Raise them as the corpus and the classifiers
mature. They're deliberately below 100% so that individually-unlabelled
elements can't be silently broken by a sweeping change, while still allowing
a few known-hard cases to be marked expected-fail.
