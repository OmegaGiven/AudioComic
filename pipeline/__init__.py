"""AudioComic pipeline, ComicDB architecture.

One accumulating record per issue (`comic.json`) that every phase writes into;
identity and naming are decided once, at the end, with full evidence. See the
ComicDB Architecture design doc.

Phases: segment -> transcribe -> identify -> resolve -> assemble -> render.
Each is idempotent against the DB; only `transcribe` costs model inference,
and only when the model or prompt version changed for a given panel.
"""
