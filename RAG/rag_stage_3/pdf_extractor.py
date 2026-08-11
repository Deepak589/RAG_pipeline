"""Stage 3 — resized-parent extractor. Imports the frozen stage-2 extractor,
reuses its detection primitives, and reshapes parents: split IEEE-subhead
giants, merge tiny stubs, for a tighter char_len band.

Stage 2 is never modified — it is the A/B baseline. This module loads it by
explicit file path (both files are named pdf_extractor.py, so a plain
`import pdf_extractor` from here would grab THIS file).

Run:  python pdf_extractor.py            # extract + resize -> stage_3/sections.json
      python pdf_extractor.py --selftest # unit-test resize logic (no PDF load)
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
STAGE2 = HERE.parent / "rag_stage_2"
OUT_PATH = HERE / "sections.json"

# Second-tier IEEE subhead: "A. Retrieval Source", "B. Indexing". Letters only —
# numbered list items ("1) New Modules") are body content, not peer headings.
SUBHEAD_RE = re.compile(r"^\s*([A-Z])\.?\s+([A-Z][A-Za-z][^\n]{0,50})$")

GIANT_CHARS = 5000    # sections larger than this are split on subheads
STUB_CHARS = 500      # sections smaller than this are merged into a sibling


def _load_s2():
    """Load the frozen stage-2 extractor as module 's2_extractor' by path."""
    spec = importlib.util.spec_from_file_location(
        "s2_extractor", STAGE2 / "pdf_extractor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S2 = _load_s2()


# ------------------------------------------------------------ subhead detection

def _is_subhead(m, last_letter):
    """True if a SUBHEAD_RE match is a real subhead, not a body fragment.

    Guards: no trailing sentence punctuation; letter must climb over the last
    subhead letter seen (rejects a body line that merely starts "A ...").
    """
    marker, text = m.group(1), m.group(2).strip()
    if text[-1:] in ",;:":
        return False
    if last_letter is not None and marker <= last_letter:
        return False
    return True


def _split_giant(sec):
    """Split one section on valid subheads into a list of fragments. The first
    fragment (the head) keeps the section title; each subhead starts a new
    fragment titled by the subhead line. All fragments inherit the section's
    page range. No valid subhead -> returns [sec] unchanged."""
    lines = sec["text"].splitlines()
    frags, title, buf = [], sec["title"], []
    last_letter = None

    def flush(t, b):
        if b:
            frags.append({
                "title": t,
                "page_start": sec["page_start"], "page_end": sec["page_end"],
                "text": "\n".join(b).strip(),
            })

    for line in lines:
        m = SUBHEAD_RE.match(line)
        if m and _is_subhead(m, last_letter):
            flush(title, buf)
            title, buf = line.strip(), [line]
            last_letter = m.group(1)
        else:
            buf.append(line)
    flush(title, buf)
    return frags if len(frags) > 1 else [dict(sec)]


def _absorb(dst, src):
    """Fold src's text into dst in place and widen dst's page range."""
    dst["text"] = (dst["text"] + "\n" + src["text"]).strip()
    dst["page_start"] = min(dst["page_start"], src["page_start"])
    dst["page_end"] = max(dst["page_end"], src["page_end"])


def _merge_within_group(g):
    """Fold sub-STUB fragments into a neighbor WITHIN the same group (a giant's
    own split parts). A tiny non-head part folds into the preceding part; a
    tiny head folds FORWARD into the next part, which is then promoted to the
    group anchor but keeps the group's real title (the head's), not the
    subhead's — the merged fragment is still "the section", just missing its
    own intro body. Never crosses a group (top-section) boundary."""
    if len(g) == 1:
        return g
    out = [g[0]]
    for frag in g[1:]:
        if len(frag["text"]) < STUB_CHARS:
            _absorb(out[-1], frag)
        else:
            out.append(frag)
    if len(out) > 1 and len(out[0]["text"]) < STUB_CHARS:   # tiny head -> forward
        head_title = out[0]["title"]
        _absorb(out[1], out[0])
        out[1]["title"] = head_title
        out = out[1:]
    return out


def _merge_tiny_groups(groups):
    """Fold a whole tiny group (a runt raw section, single fragment under
    STUB_CHARS) into the previous group's last fragment. This is the only
    cross-section merge, reserved for identity-less runts (e.g. a 69-char
    orphan). A leading tiny group with no predecessor is kept as-is."""
    out = []
    for g in groups:
        if len(g) == 1 and len(g[0]["text"]) < STUB_CHARS and out:
            _absorb(out[-1][-1], g[0])
        else:
            out.append(g)
    return out


def resize_sections(raw):
    """Reshape raw sections into a tighter char_len band, preserving the
    original section grouping. Split giants on subheads (Move 1); merge stubs
    within a group, then fold runt groups into the previous section (Move 2).
    Each fragment carries `_gid` = its source section's original index, so the
    first fragment of group g keeps the int id `g` and its subheads get
    `g#A`, `g#B`, ... (letters reset per group)."""
    groups = []
    for i, sec in enumerate(raw):
        frags = _split_giant(sec) if len(sec["text"]) > GIANT_CHARS else [dict(sec)]
        frags = _merge_within_group(frags)
        for f in frags:
            f["_gid"] = i
        groups.append(frags)
    groups = _merge_tiny_groups(groups)

    out = []
    for g in groups:
        for j, frag in enumerate(g):
            gid = frag["_gid"]
            frag["section_idx"] = gid if j == 0 else f"{gid}#{chr(65 + j - 1)}"
            out.append(frag)
    return out


# --------------------------------------------------------------- extraction

def extract_doc(path):
    """Extract one PDF, resize its sections (split giants, merge stubs), and
    build records. resize_sections has already set each fragment's
    `section_idx` (int for a group head, `g#A` for subheads)."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    pages = S2._pages_text(pdf)
    raw = S2._sections_from_regex(pages)
    resized = resize_sections(raw)

    records = []
    for sec in resized:
        if len(sec["text"]) < 40:
            continue
        sec.pop("_gid", None)                      # internal tag, not persisted
        idx = sec.pop("section_idx")
        records.append({
            "source": path.name, "section_idx": idx,
            **sec, "char_len": len(sec["text"]),
        })
    return records


# ------------------------------------------------------------------- selftest

def _selftest():
    """Unit-test resize logic on hand-built sections (no PDF/model load)."""
    ok = True

    def check(cond, label):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and cond

    # --- _is_subhead guards ---
    m = SUBHEAD_RE.match("A. Retrieval Source")
    check(m is not None and _is_subhead(m, None), "letter subhead A accepted")
    m = SUBHEAD_RE.match("A. Retrieval Source")
    check(not _is_subhead(m, "A"), "letter repeat A after A rejected")
    m = SUBHEAD_RE.match("B. Indexing Optimization")
    check(m is not None and _is_subhead(m, "A"), "letter climb A->B accepted")
    m = SUBHEAD_RE.match("2019). REALM achieves new state of the art")
    check(m is None or not _is_subhead(m, None), "citation year 2019) rejected")
    m = SUBHEAD_RE.match("2) Metadata Attachments")
    check(m is None, "numbered list item 2) not treated as subhead")
    m = SUBHEAD_RE.match("Section, describing")
    check(m is None or not _is_subhead(m, None), "trailing comma rejected")

    # --- _split_giant (head has real intro text) ---
    giant = {
        "title": "III RETRIEVAL", "page_start": 3, "page_end": 8,
        "text": "III RETRIEVAL\n" + "intro body words " * 60
                + "\nA. Retrieval Source\n" + "y " * 3000
                + "\nB. Indexing Optimization\n" + "z " * 3000,
    }
    parts = _split_giant(giant)
    check(len(parts) == 3, f"giant splits into 3 parts (got {len(parts)})")
    check(parts[0]["title"] == "III RETRIEVAL", "head keeps giant title")
    check(parts[1]["title"] == "A. Retrieval Source", "part 1 titled by subhead")
    check(all(p["page_start"] == 3 and p["page_end"] == 8 for p in parts),
          "splits inherit giant page range")

    # --- _merge_within_group: tiny non-head part folds into previous ---
    grp = [
        {"title": "H", "page_start": 1, "page_end": 1, "text": "h" * 800},
        {"title": "A. sub", "page_start": 1, "page_end": 1, "text": "a" * 800},
        {"title": "B. tiny", "page_start": 2, "page_end": 2, "text": "b" * 100},
    ]
    mg = _merge_within_group(grp)
    check(len(mg) == 2, f"tiny part folds into previous (got {len(mg)})")
    check("b" * 100 in mg[1]["text"], "tiny B folded into A")

    # --- _merge_within_group: tiny head folds forward, next promoted ---
    grp2 = [
        {"title": "III RETRIEVAL", "page_start": 3, "page_end": 3, "text": "III RETRIEVAL"},
        {"title": "A. Retrieval Source", "page_start": 3, "page_end": 5, "text": "a" * 900},
    ]
    mg2 = _merge_within_group(grp2)
    check(len(mg2) == 1, "tiny head folds forward")
    check("III RETRIEVAL" in mg2[0]["text"] and mg2[0]["title"] == "III RETRIEVAL",
          "promoted fragment keeps the real section title")

    # --- group-aware id assignment via resize_sections ---
    resized = resize_sections([{"title": "0 front", "page_start": 1, "page_end": 1,
                                "text": "f" * 800}, giant])
    ids = [r["section_idx"] for r in resized]
    check(ids[0] == 0, "non-giant group keeps int id 0")
    check(ids[1] == 1, "giant head keeps int id 1 (group index)")
    check(ids[2] == "1#A" and ids[3] == "1#B", f"subheads get 1#A/1#B (got {ids[2:]})")

    # --- _merge_tiny_groups: runt whole-section folds into previous group ---
    groups = [[{"title": "1 A", "page_start": 1, "page_end": 1, "text": "a" * 800}],
              [{"title": "2 runt", "page_start": 1, "page_end": 1, "text": "r" * 100}]]
    tg = _merge_tiny_groups(groups)
    check(len(tg) == 1 and "r" * 100 in tg[0][0]["text"],
          "runt group folds into previous section")
    return ok


# ------------------------------------------------------------------------ main

def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    pdfs = sorted(S2.DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {S2.DOCS_DIR}")
    all_records = []
    for path in pdfs:
        all_records.extend(extract_doc(path))
    OUT_PATH.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))
    print(f"Wrote {len(all_records)} sections -> {OUT_PATH.name}")


if __name__ == "__main__":
    main()
