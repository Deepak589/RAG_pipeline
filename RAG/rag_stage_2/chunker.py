"""Stage 2 — parent-child chunking over sections.json -> chunks.json.

Parent-child retrieval: embed and match *small* child chunks (precise),
but hand the *parent section* to the LLM (full context). Small children
sharpen retrieval; the parent restores the surrounding context a lone
120-word window would lose.

  parent = a section from sections.json          (id: "source#section_idx")
  child  = a word-window inside that section      (id: "source#sec#child")

sections.json stays the parent store (no text duplication). This writes
chunks.json = the child records, each pointing back at its parent_id.

Run:  python chunker.py            # build chunks.json + size report
"""

import json
from collections import Counter
from pathlib import Path

SECTIONS_PATH = Path(__file__).parent / "sections.json"
OUT_PATH = Path(__file__).parent / "chunks.json"

CHILD_SIZE = 120      # words per child chunk (retrieval unit)
CHILD_OVERLAP = 20    # words shared between consecutive children
MIN_TAIL = 15         # drop trailing fragments shorter than this


def _child_windows(words):
    """Yield overlapping fixed-size word windows, skipping tiny tails."""
    step = CHILD_SIZE - CHILD_OVERLAP
    for start in range(0, len(words), step):
        piece = words[start:start + CHILD_SIZE]
        if start > 0 and len(piece) < MIN_TAIL:   # keep a short lone section
            continue
        yield " ".join(piece)


def build_children(sections):
    """Split each section into child chunks tagged with parent_id."""
    children = []
    for sec in sections:
        parent_id = f"{sec['source']}#{sec['section_idx']}"
        words = sec["text"].split()
        for ci, text in enumerate(_child_windows(words)):
            children.append({
                "id": f"{parent_id}#{ci}",
                "parent_id": parent_id,
                "source": sec["source"],
                "title": sec["title"],
                "page_start": sec["page_start"],
                "page_end": sec["page_end"],
                "text": text,
            })
    return children


def main():
    if not SECTIONS_PATH.exists():
        raise SystemExit(f"{SECTIONS_PATH.name} missing — run pdf_extractor.py first")
    sections = json.loads(SECTIONS_PATH.read_text())
    children = build_children(sections)
    OUT_PATH.write_text(json.dumps(children, indent=2, ensure_ascii=False))

    # size report — sanity-check child word counts + children per parent
    wc = [len(c["text"].split()) for c in children]
    per_parent = Counter(c["parent_id"] for c in children)
    buckets = Counter(min(w // 30 * 30, 120) for w in wc)
    print(f"Parents (sections): {len(sections)}")
    print(f"Children (chunks):  {len(children)}")
    print(f"Children/parent:    min {min(per_parent.values())}  "
          f"max {max(per_parent.values())}  "
          f"avg {len(children) / len(per_parent):.1f}")
    print(f"Child word counts:  min {min(wc)}  max {max(wc)}  "
          f"avg {sum(wc) / len(wc):.0f}")
    print("Word-count histogram (bucket -> count):")
    for b in sorted(buckets):
        print(f"  {b:>3}+ : {buckets[b]}")
    print(f"\nWrote {len(children)} children -> {OUT_PATH.name}")


if __name__ == "__main__":
    main()
