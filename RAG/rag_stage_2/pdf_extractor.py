"""Stage 2 — extract PDFs in docs/ into sections.json for later chunking.

Extract-only pass: PDF -> per-page text -> sections. NO chunking yet; the
goal is to verify extraction + section detection quality before building
parent-child chunks on top.

Section detection: regex on numbered headings, uniform for every doc.
Matches Arabic ('1 Introduction', '2.1 Method') and Roman ('II. Overview',
IEEE survey style). TOC bookmarks were tried but only give page-level
granularity, so multiple headings on one page collapse into duplicate
sections; in-text heading regex is char-precise and dup-free.

Output: sections.json  (list of section records) + a sample dump to stdout.
Run:    python pdf_extractor.py            # extract + dump samples
        python pdf_extractor.py --dump 3   # show first 3 sections per doc
"""

import json
import re
import sys
from pathlib import Path

import pypdfium2 as pdfium

DOCS_DIR = Path(__file__).parent.parent / "docs"   # shared corpus at RAG/ root
OUT_PATH = Path(__file__).parent / "sections.json"

# Numbered heading on its own line. Arabic ("1 Introduction", "2.1 Related
# Work", "3. Method") or Roman ("II. Overview", IEEE survey style). Requires
# a letter word after the number, rejecting page nums / equation refs.
HEADING_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*|[IVXL]+)\.?\s+([A-Z][A-Za-z][^\n]{0,60})$"
)

# End-of-content marker. Everything from the bibliography onward is dropped:
# reference lists have no answerable content and their year/URL lines
# false-match HEADING_RE, dumping the whole list into one noisy section.
REF_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.I)


# ------------------------------------------------------------- text extraction

def _pages_text(pdf):
    """Return list of page strings, index 0 = page 1."""
    pages = []
    for i in range(len(pdf)):
        tp = pdf[i].get_textpage()
        pages.append(tp.get_text_range())
    return pages


# ---------------------------------------------------------- section detection

def _is_heading(m, last_bare_int):
    """Reject HEADING_RE matches that are really footnotes / body fragments.

    Two guards, both must pass:
    1. No trailing sentence punctuation — real headings don't end in ,;:
       but a footnote sentence-fragment ("3 Note that we still fine-tune...,")
       does.
    2. Increasing bare integer — a bare-integer heading number N (no dot,
       Arabic) must exceed the last bare-integer section opened. A second "3"
       after "3 Approach" is a footnote marker, not section 3 again. Dotted
       subsections ("3.1") and Roman numerals skip this check.
    """
    number, text = m.group(1), m.group(2).strip()
    if text[-1:] in ",;:":
        return False
    if number.isdigit():                       # bare integer (no dot, Arabic)
        n = int(number)
        if last_bare_int is not None and n <= last_bare_int:
            return False
    return True


def _split_sections(lines):
    """Split (page, line) pairs into sections at each numbered heading."""
    sections, title, page_start, buf = [], None, lines[0][0], []
    last_page = lines[-1][0]
    last_bare_int = None

    def flush(end_page):
        if buf:
            sections.append({
                "title": title or "(front matter)",
                "page_start": page_start, "page_end": end_page,
                "text": "\n".join(buf).strip(),
            })

    for pnum, line in lines:
        m = HEADING_RE.match(line)
        if m and _is_heading(m, last_bare_int):
            flush(pnum)
            title = f"{m.group(1)} {m.group(2).strip()}"
            page_start, buf = pnum, [line]
            if m.group(1).isdigit():
                last_bare_int = int(m.group(1))
        else:
            buf.append(line)
    flush(last_page)
    return sections


def _sections_from_regex(pages):
    """Sections split on numbered headings, truncated at the references list."""
    # Track which page each line came from so sections carry page numbers.
    lines = []
    for pnum, page in enumerate(pages, start=1):
        for line in page.splitlines():
            if REF_RE.match(line):
                return _split_sections(lines)   # stop before bibliography
            lines.append((pnum, line))
    return _split_sections(lines)


def extract_doc(path):
    """Extract one PDF into section records via heading regex."""
    pdf = pdfium.PdfDocument(str(path))
    pages = _pages_text(pdf)
    raw = _sections_from_regex(pages)

    records = []
    for i, sec in enumerate(raw):
        if len(sec["text"]) < 40:              # drop empty/heading-only slivers
            continue
        records.append({
            "source": path.name, "section_idx": i,
            **sec, "char_len": len(sec["text"]),
        })
    return records


# ------------------------------------------------------------------------ main

def main():
    dump_n = 2
    if len(sys.argv) > 2 and sys.argv[1] == "--dump":
        dump_n = int(sys.argv[2])

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {DOCS_DIR}")

    all_records = []
    for path in pdfs:
        recs = extract_doc(path)
        all_records.extend(recs)
        total_chars = sum(r["char_len"] for r in recs)
        print(f"\n=== {path.name[:55]}")
        print(f"    sections={len(recs)}  chars={total_chars:,}")
        for r in recs[:dump_n]:
            snippet = " ".join(r["text"][:120].split())
            print(f"    §{r['section_idx']:>2} p{r['page_start']}-{r['page_end']}"
                  f"  \"{r['title'][:45]}\"  [{r['char_len']} ch]")
            print(f"        {snippet}...")

    OUT_PATH.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(all_records)} sections from {len(pdfs)} PDFs "
          f"-> {OUT_PATH.name}")


if __name__ == "__main__":
    main()
