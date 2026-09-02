"""Collect every bakeoff/out/<engine>/manifest.json into one self-contained
comparison page: bakeoff/out/index.html.

Audio is embedded as base64 data URIs (opus if ffmpeg produced it, else wav)
so the page is a single file you can open anywhere or hand off for review.
"""

from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

from _common import BAKEOFF_DIR, OUT_DIR

ENGINE_ORDER = ["piper", "kokoro", "chatterbox", "orpheus", "vibevoice"]


def _data_uri(path: Path) -> str:
    mime = "audio/ogg" if path.suffix == ".opus" else "audio/wav"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _audio(engine_dir: Path, opus: str | None, wav: str | None) -> str:
    name = opus or wav
    if not name:
        return '<span class="missing">&mdash;</span>'
    p = engine_dir / name
    if not p.exists():
        return '<span class="missing">&mdash;</span>'
    return f'<audio controls preload="none" src="{_data_uri(p)}"></audio>'


def main() -> None:
    manifests = []
    for name in ENGINE_ORDER:
        mf = OUT_DIR / name / "manifest.json"
        if mf.exists():
            manifests.append((name, json.loads(mf.read_text())))
    for mf in sorted(OUT_DIR.glob("*/manifest.json")):
        if mf.parent.name not in ENGINE_ORDER:
            manifests.append((mf.parent.name, json.loads(mf.read_text())))

    if not manifests:
        sys.exit("no engine output found -- run some engines first")

    passage = json.loads((BAKEOFF_DIR / "passage.json").read_text())
    segs = passage["segments"]

    # per-segment matrix rows
    seg_rows = ""
    for i, s in enumerate(segs):
        cells = ""
        for engine, m in manifests:
            edir = OUT_DIR / engine
            entry = next((e for e in m["segments"] if e["idx"] == i), None)
            if entry and (entry.get("opus") or entry.get("wav")):
                cells += f'<td>{_audio(edir, entry.get("opus"), entry.get("wav"))}</td>'
            elif not m.get("per_segment_available", True):
                cells += '<td><span class="missing">full only</span></td>'
            else:
                cells += '<td><span class="missing">&mdash;</span></td>'
        tag = f' <span class="tag">{html.escape(s["kind"])}</span>' if s.get("kind") else ""
        emo = f' <span class="emo">{html.escape(s.get("emotion",""))}</span>'
        seg_rows += f"""
      <tr>
        <th scope="row">
          <span class="spk">{html.escape(s["speaker"])}</span>{emo}{tag}
          <span class="line">{html.escape(s["text"])}</span>
        </th>
        {cells}
      </tr>"""

    full_cells = ""
    engine_head = ""
    for engine, m in manifests:
        edir = OUT_DIR / engine
        note = html.escape(str(m.get("meta", {}).get("note", "")))
        dur = m.get("full_duration_s", "?")
        engine_head += f'<th><span class="en">{html.escape(engine)}</span><span class="meta">{note} &middot; {dur}s</span></th>'
        full_cells += f'<td>{_audio(edir, m.get("full_opus"), m.get("full_wav"))}</td>'

    doc = _TEMPLATE.format(
        engine_head=engine_head,
        full_cells=full_cells,
        seg_rows=seg_rows,
        passage_title=html.escape(passage.get("title", "TTS bake-off")),
        n_engines=len(manifests),
    )
    out = OUT_DIR / "index.html"
    out.write_text(doc)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size_mb:.1f} MB, {len(manifests)} engines)")
    if size_mb > 14:
        print("  WARNING: >14 MB -- too big to publish as an artifact. "
              "Install ffmpeg so audio compresses to opus, or drop an engine.")


_TEMPLATE = """<title>TTS Bake-off</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --paper:#f5f2ea; --surface:#fffefb; --ink:#1a1714; --ink-soft:#4a443c;
    --rule:#d9d3c4; --accent:#1e4fd8; --accent-soft:#e7ecfc; --spot:#d1216f;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --paper:#14120f; --surface:#1d1a16; --ink:#efe9dd; --ink-soft:#b3aa98;
      --rule:#35302a; --accent:#8fa8ff; --accent-soft:#23263a; --spot:#f06baa;
    }}
  }}
  :root[data-theme="dark"] {{
    --paper:#14120f; --surface:#1d1a16; --ink:#efe9dd; --ink-soft:#b3aa98;
    --rule:#35302a; --accent:#8fa8ff; --accent-soft:#23263a; --spot:#f06baa;
  }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--paper); color:var(--ink); margin:0;
    font-family:"Newsreader","Iowan Old Style",Georgia,serif; font-size:17px; line-height:1.55; }}
  .wrap {{ max-width:min(1400px, 96vw); margin:0 auto; padding:2.5rem 1.25rem 5rem; }}
  h1 {{ font-family:"Barlow Condensed","Arial Narrow",sans-serif; font-weight:800;
    text-transform:uppercase; letter-spacing:.01em; font-size:clamp(2.2rem,5vw,3.4rem);
    margin:0 0 .2rem; }}
  .sub {{ color:var(--ink-soft); font-size:1.05rem; margin:0 0 2rem; max-width:60ch; }}
  .rubric {{ background:var(--surface); border:1px solid var(--rule); border-left:4px solid var(--spot);
    border-radius:4px; padding:1rem 1.25rem; margin:0 0 2.5rem; font-size:.95rem; }}
  .rubric b {{ font-family:"Barlow Condensed",sans-serif; text-transform:uppercase; letter-spacing:.04em; }}
  .rubric ul {{ margin:.4rem 0 0; padding-left:1.2rem; }}
  .scroll {{ overflow-x:auto; border:1px solid var(--rule); border-radius:6px; background:var(--surface); }}
  table {{ border-collapse:collapse; width:100%; }}
  caption {{ text-align:left; font-family:"Barlow Condensed",sans-serif; font-weight:700;
    text-transform:uppercase; letter-spacing:.05em; padding:.9rem 1rem .5rem; color:var(--ink-soft); }}
  th, td {{ padding:.7rem .8rem; border-bottom:1px solid var(--rule); text-align:left; vertical-align:top; }}
  thead th {{ position:sticky; top:0; background:var(--surface); border-bottom:2px solid var(--ink);
    font-family:"Barlow Condensed",sans-serif; }}
  thead th .en {{ display:block; font-weight:800; text-transform:uppercase; font-size:1.05rem; }}
  thead th .meta {{ display:block; font-weight:400; font-size:.72rem; color:var(--ink-soft);
    font-family:"IBM Plex Mono",ui-monospace,monospace; max-width:22ch; line-height:1.3; margin-top:.2rem; }}
  tbody th {{ font-weight:400; max-width:34ch; }}
  tbody th .spk {{ font-family:"Barlow Condensed",sans-serif; font-weight:700; text-transform:uppercase;
    letter-spacing:.03em; color:var(--accent); }}
  tbody th .emo {{ font-family:"IBM Plex Mono",monospace; font-size:.68rem; text-transform:uppercase;
    letter-spacing:.06em; color:var(--spot); }}
  tbody th .tag {{ font-family:"IBM Plex Mono",monospace; font-size:.62rem; text-transform:uppercase;
    background:var(--accent-soft); padding:.05em .4em; border-radius:3px; color:var(--ink-soft); }}
  tbody th .line {{ display:block; margin-top:.3rem; color:var(--ink); }}
  .full-row td, .full-row th {{ background:var(--accent-soft); border-bottom:2px solid var(--ink); }}
  audio {{ width:230px; max-width:38vw; height:34px; }}
  .missing {{ color:var(--ink-soft); font-family:"IBM Plex Mono",monospace; font-size:.8rem; }}
  a {{ color:var(--accent); }}
  .foot {{ margin-top:2rem; color:var(--ink-soft); font-size:.85rem;
    font-family:"IBM Plex Mono",monospace; }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap">

<div class="wrap">
  <h1>TTS Bake-off</h1>
  <p class="sub">{passage_title}. Same passage, {n_engines} engines. Listen down a column for one engine's whole take; listen across a row to compare a single line.</p>

  <div class="rubric">
    <b>Score each engine 1&ndash;5 on:</b>
    <ul>
      <li><b>Naturalness</b> &mdash; does it sound like a person reading, or a machine</li>
      <li><b>Emotion</b> &mdash; is the menace / urgency / pain / weariness actually there</li>
      <li><b>Character distinctiveness</b> &mdash; are the three voices clearly different people</li>
      <li><b>Onomatopoeia</b> &mdash; the <code>Aaaah!</code> and the <code>*sigh*</code> line: convincing, or read as letters</li>
      <li><b>Seams</b> &mdash; does the full passage flow, or lurch between clips</li>
      <li><b>Speed / VRAM</b> &mdash; render time and memory (see per-engine notes)</li>
    </ul>
  </div>

  <div class="scroll">
    <table>
      <caption>Full passage &mdash; then line by line</caption>
      <thead>
        <tr><th>Segment</th>{engine_head}</tr>
      </thead>
      <tbody>
        <tr class="full-row"><th scope="row"><span class="spk">Whole passage</span><span class="line">start to finish, one clip</span></th>{full_cells}</tr>
        {seg_rows}
      </tbody>
    </table>
  </div>

  <p class="foot">Generated by bakeoff/build_page.py &middot; AudioComic</p>
</div>
"""


if __name__ == "__main__":
    main()
