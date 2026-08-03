# Stage 3 Chunk-Resize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stage-3 chunking strategy that resizes parent sections (split IEEE-subhead giants, merge tiny stubs) into a ~600–5000 char band, then A/B it against stage 2 with the existing eval.

**Architecture:** Import-and-extend. `rag_stage_3/pdf_extractor.py` loads the frozen stage-2 extractor by explicit file path (`importlib`), reuses its detection primitives, and adds a `resize_sections()` pass (split-then-merge). `rag_stage_3/chunker.py` imports stage-2 `build_children` and windows the stage-3 sections. `run_eval.py` gains a `--stage3` flag. Stage 2 is never modified.

**Tech Stack:** Python 3.12, numpy, sentence-transformers, pypdfium2. Framework-free tests (inline `--selftest`, matching repo convention — no pytest).

## Global Constraints

- **Do not modify anything under `RAG/rag_stage_2/`** — it is a frozen A/B baseline. All new code lives in `RAG/rag_stage_3/` or is an additive flag in `RAG/rag_stage_2/eval/run_eval.py` (the eval harness is shared infra, additive-only).
- **No test framework** — repo is deliberately framework-free (`run_eval.py` docstring: "numpy + stdlib"). Tests are `selftest_*()` functions invoked via `--selftest`, mirroring `run_eval.selftest_metrics`.
- **Resize thresholds:** giants `char_len > 5000`; stubs `char_len < 500`; target band ~600–5000.
- **SUBHEAD_RE** (verbatim): `r"^\s*([A-Z]|\d+\))\.?\s+([A-Z][A-Za-z][^\n]{0,50})$"`.
- **Subhead guards, all must pass:** (1) text not ending in `,;:`; (2) `[A-Z]` form: letter climbs over last letter in this giant; (3) `\d+)` form: number ≤9 AND climbs over last number since the most recent letter; (4) short Title-Case (enforced by regex).
- **Id scheme:** head keeps original int idx `3`; splits append letter `3#A`, `3#B`. Merge drops the absorbed stub's idx. `parent_id = f"{source}#{section_idx}"` already yields `source#3#A` for a string idx — no downstream change.
- **User handles commits** — the "Commit" steps below are written for completeness, but DO NOT run `git commit` yourself; leave staged changes for the user unless they say otherwise.
- Run all commands from `RAG/rag_stage_3/` unless noted (or `RAG/rag_stage_2/eval/` for eval).

---

### Task 1: Stage-3 extractor scaffold — import stage-2 primitives

Create the stage-3 extractor that loads the frozen stage-2 module by explicit path and re-exports its primitives, with a trivial passthrough `extract_doc` (no resize yet). Proves the import-extend wiring works before any resize logic.

**Files:**
- Create: `RAG/rag_stage_3/pdf_extractor.py`

**Interfaces:**
- Consumes: stage-2 `RAG/rag_stage_2/pdf_extractor.py` module members `_pages_text`, `_sections_from_regex`, `DOCS_DIR` (loaded via importlib).
- Produces: `_load_s2()` → stage-2 module object; module-level `S2` bound to it; `OUT_PATH` (stage_3/sections.json); `extract_doc(path)` → `list[dict]` records with keys `source, section_idx, title, page_start, page_end, text, char_len`.

- [ ] **Step 1: Write the import-loader + passthrough extractor**

```python
"""Stage 3 — resized-parent extractor. Imports the frozen stage-2 extractor,
reuses its detection primitives, and (in later tasks) reshapes parents:
split IEEE-subhead giants, merge tiny stubs, for a tighter char_len band.

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


def _load_s2():
    """Load the frozen stage-2 extractor as module 's2_extractor' by path."""
    spec = importlib.util.spec_from_file_location(
        "s2_extractor", STAGE2 / "pdf_extractor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S2 = _load_s2()


def extract_doc(path):
    """Extract one PDF into section records (passthrough; resize added later)."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    pages = S2._pages_text(pdf)
    raw = S2._sections_from_regex(pages)   # [{title, page_start, page_end, text}]

    records = []
    for i, sec in enumerate(raw):
        if len(sec["text"]) < 40:
            continue
        records.append({
            "source": path.name, "section_idx": i,
            **sec, "char_len": len(sec["text"]),
        })
    return records
```

- [ ] **Step 2: Write a smoke check confirming the import resolves**

Run:
```bash
cd RAG/rag_stage_3 && python -c "import pdf_extractor as e; print('S2 loaded:', hasattr(e.S2, '_sections_from_regex'), hasattr(e.S2, '_pages_text'))"
```
Expected: `S2 loaded: True True`

- [ ] **Step 3: Run the extractor end-to-end, confirm it matches stage-2 counts**

Since `extract_doc` is still a passthrough, stage-3 output must equal stage-2's 111 sections. Add a temporary `main()` (kept for later tasks):

```python
def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)   # _selftest defined in later tasks
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
```

Note: `_selftest` is referenced but not yet defined — do NOT run `--selftest` until Task 4. The plain run below does not touch that branch.

Run:
```bash
cd RAG/rag_stage_3 && python pdf_extractor.py
```
Expected: `Wrote 111 sections -> sections.json` (equals stage-2's section count — passthrough parity).

- [ ] **Step 4: Commit** (stage the files; the user runs the actual commit)

```bash
git add RAG/rag_stage_3/pdf_extractor.py
# user commits: "feat(stage3): extractor scaffold importing stage-2 primitives"
```

---

### Task 2: `_is_subhead` guard + `resize_sections` split pass (Move 1)

Add the subhead detector with all guards, and the split half of `resize_sections`. Wire split into `extract_doc`. Merge comes in Task 3.

**Files:**
- Modify: `RAG/rag_stage_3/pdf_extractor.py`

**Interfaces:**
- Consumes: `SUBHEAD_RE`; raw sections `[{title, page_start, page_end, text}]`.
- Produces:
  - `SUBHEAD_RE` (module constant).
  - `_is_subhead(m, last_letter, last_num) -> bool` — guard check for a `SUBHEAD_RE` match.
  - `_split_giant(sec) -> list[dict]` — split one giant section on subheads into `[{title, page_start, page_end, text}]` fragments (page range inherited); returns `[sec]` unchanged if no valid subhead found.
  - `resize_sections(raw) -> list[dict]` — apply split to every section (`char_len > 5000` only), returning reshaped `[{title, page_start, page_end, text}]` in order. (Merge added in Task 3.)

- [ ] **Step 1: Write the failing selftest cases for the subhead guard + split**

Add to `pdf_extractor.py` (these call functions that don't exist yet → fail):

```python
def _selftest():
    """Unit-test resize logic on hand-built sections (no PDF/model load)."""
    ok = True

    def check(cond, label):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and cond

    # --- _is_subhead guards ---
    m = SUBHEAD_RE.match("A. Retrieval Source")
    check(m is not None and _is_subhead(m, None, None), "letter subhead A accepted")
    m = SUBHEAD_RE.match("A. Retrieval Source")
    check(not _is_subhead(m, "A", None), "letter repeat A after A rejected")
    m = SUBHEAD_RE.match("B. Indexing Optimization")
    check(m is not None and _is_subhead(m, "A", None), "letter climb A->B accepted")
    m = SUBHEAD_RE.match("2019). REALM achieves new state of the art")
    check(m is None or not _is_subhead(m, None, None), "citation year 2019) rejected")
    m = SUBHEAD_RE.match("2) Metadata Attachments")
    check(m is not None and _is_subhead(m, "C", 1), "number climb 1->2 accepted")
    m = SUBHEAD_RE.match("Section, describing")
    check(m is None or not _is_subhead(m, None, None), "trailing comma rejected")

    # --- _split_giant ---
    giant = {
        "title": "III RETRIEVAL", "page_start": 3, "page_end": 8,
        "text": "intro body line\n" + "x " * 3000
                + "\nA. Retrieval Source\n" + "y " * 3000
                + "\nB. Indexing Optimization\n" + "z " * 3000,
    }
    parts = _split_giant(giant)
    check(len(parts) == 3, f"giant splits into 3 parts (got {len(parts)})")
    check(parts[0]["title"] == "III RETRIEVAL", "head keeps giant title")
    check(parts[1]["title"] == "A. Retrieval Source", "part 1 titled by subhead")
    check(all(p["page_start"] == 3 and p["page_end"] == 8 for p in parts),
          "splits inherit giant page range")

    # small section untouched by split
    small = {"title": "2 Method", "page_start": 1, "page_end": 1, "text": "short " * 50}
    check(resize_sections([small]) == [small], "sub-5000 section not split")
    return ok
```

- [ ] **Step 2: Run selftest to verify it fails**

Run:
```bash
cd RAG/rag_stage_3 && python pdf_extractor.py --selftest
```
Expected: FAIL — `NameError: name '_is_subhead' is not defined` (or `SUBHEAD_RE`).

- [ ] **Step 3: Implement `SUBHEAD_RE`, `_is_subhead`, `_split_giant`, split-only `resize_sections`**

Add near the top (after `OUT_PATH`):

```python
# Second-tier IEEE subhead: "A. Retrieval Source", "B. Indexing", "2) Metadata".
SUBHEAD_RE = re.compile(r"^\s*([A-Z]|\d+\))\.?\s+([A-Z][A-Za-z][^\n]{0,50})$")

GIANT_CHARS = 5000    # sections larger than this are split on subheads
```

Add the functions:

```python
def _is_subhead(m, last_letter, last_num):
    """True if a SUBHEAD_RE match is a real subhead, not a body fragment.

    Guards: no trailing sentence punctuation; letter form must climb over the
    last letter; number form must be <=9 (rejects citation years) and climb
    over the last number since the most recent letter subhead.
    """
    marker, text = m.group(1), m.group(2).strip()
    if text[-1:] in ",;:":
        return False
    if marker.isalpha():                          # "A", "B", ...
        if last_letter is not None and marker <= last_letter:
            return False
    else:                                         # "\d+)"
        n = int(marker[:-1])                      # strip the ")"
        if n > 9:
            return False
        if last_num is not None and n <= last_num:
            return False
    return True


def _split_giant(sec):
    """Split one section on valid subheads. Head keeps the section title; each
    subhead starts a new fragment titled by the subhead line. All fragments
    inherit the section's page range. Fragments are tagged `_split` (False for
    the head, True for subhead parts) so id assignment never has to re-guess
    from the title. No valid subhead -> returns [sec] (untagged)."""
    lines = sec["text"].splitlines()
    frags, title, buf, is_part = [], sec["title"], [], False
    last_letter, last_num = None, None

    def flush(t, b, split):
        if b:
            frags.append({
                "title": t,
                "page_start": sec["page_start"], "page_end": sec["page_end"],
                "text": "\n".join(b).strip(), "_split": split,
            })

    for line in lines:
        m = SUBHEAD_RE.match(line)
        if m and _is_subhead(m, last_letter, last_num):
            flush(title, buf, is_part)
            title, buf, is_part = line.strip(), [line], True
            marker = m.group(1)
            if marker.isalpha():
                last_letter, last_num = marker, None   # numbers reset per letter
            else:
                last_num = int(marker[:-1])
        else:
            buf.append(line)
    flush(title, buf, is_part)
    return frags if len(frags) > 1 else [sec]


def resize_sections(raw):
    """Reshape raw sections: split giants on subheads. (Merge added in Task 3.)"""
    out = []
    for sec in raw:
        if len(sec["text"]) > GIANT_CHARS:
            out.extend(_split_giant(sec))
        else:
            out.append(sec)
    return out
```

- [ ] **Step 4: Run selftest to verify it passes**

Run:
```bash
cd RAG/rag_stage_3 && python pdf_extractor.py --selftest
```
Expected: all lines `[PASS]`, exit 0.

- [ ] **Step 5: Commit** (stage; user commits)

```bash
git add RAG/rag_stage_3/pdf_extractor.py
# user commits: "feat(stage3): subhead split pass (Move 1) with guards"
```

---

### Task 3: Merge stubs (Move 2) + id assignment, wire into `extract_doc`

Add the merge half of `resize_sections`, assign the `3` / `3#A` id scheme, and wire the full resize into `extract_doc`. After this the extractor produces the real resized `sections.json`.

**Files:**
- Modify: `RAG/rag_stage_3/pdf_extractor.py`

**Interfaces:**
- Consumes: split output from Task 2 `resize_sections`.
- Produces:
  - `STUB_CHARS = 500` constant.
  - `_merge_stubs(secs) -> list[dict]` — fold any section with `len(text) < 500` into the nearest sibling (prefer preceding, else following), concatenating text and widening page range.
  - Updated `resize_sections(raw)` — split then merge.
  - Updated `extract_doc` — assigns `section_idx` per the head/`#A` scheme over resized sections.

- [ ] **Step 1: Extend the selftest with merge + id cases (they fail)**

Append to `_selftest()` before `return ok`:

```python
    # --- _merge_stubs ---
    a = {"title": "1 A", "page_start": 1, "page_end": 2, "text": "a" * 800}
    stub = {"title": "1.1 tiny", "page_start": 2, "page_end": 2, "text": "b" * 100}
    b = {"title": "2 B", "page_start": 3, "page_end": 4, "text": "c" * 800}
    merged = _merge_stubs([a, stub, b])
    check(len(merged) == 2, f"stub merged away (got {len(merged)} sections)")
    check("b" * 100 in merged[0]["text"], "stub text folded into preceding sibling")
    check(merged[0]["page_end"] == 2, "merge widens page range")
    lead_stub = {"title": "0 s", "page_start": 1, "page_end": 1, "text": "z" * 100}
    m2 = _merge_stubs([lead_stub, a])
    check(len(m2) == 1 and "z" * 100 in m2[0]["text"],
          "leading stub folds into following sibling")

    # --- id assignment via split (integration through extract-style build) ---
    # A giant that splits -> head idx int, splits get letter suffix.
    ids = _assign_ids(_split_giant(giant))
    check(ids[0] == 0, "split head keeps int idx 0")
    check(ids[1] == "0#A" and ids[2] == "0#B", f"splits get 0#A/0#B (got {ids[1:]})")
```

Note: `giant` is defined earlier in `_selftest` (Task 2). `_assign_ids` is introduced here.

- [ ] **Step 2: Run selftest to verify the new cases fail**

Run:
```bash
cd RAG/rag_stage_3 && python pdf_extractor.py --selftest
```
Expected: earlier cases PASS, new ones FAIL — `NameError: name '_merge_stubs' is not defined`.

- [ ] **Step 3: Implement merge, id assignment, and finish `resize_sections` + `extract_doc`**

Add constant near `GIANT_CHARS`:

```python
STUB_CHARS = 500      # sections smaller than this are merged into a sibling
```

Add functions:

```python
def _merge_stubs(secs):
    """Fold sections under STUB_CHARS into the nearest sibling (prefer the
    preceding one; if the stub is first, the following one). Concatenates text
    and widens the page range. Sibling means adjacent in `secs`, which is
    already per-source (extract_doc runs this on one document's sections)."""
    out = []
    for sec in secs:
        if len(sec["text"]) < STUB_CHARS and out:     # fold into preceding
            tgt = out[-1]
            tgt["text"] = (tgt["text"] + "\n" + sec["text"]).strip()
            tgt["page_start"] = min(tgt["page_start"], sec["page_start"])
            tgt["page_end"] = max(tgt["page_end"], sec["page_end"])
        else:                                         # keep (incl. leading stub)
            out.append(sec)
    # second pass: fold any leading stub (no preceding sibling above) forward
    if len(out) > 1 and len(out[0]["text"]) < STUB_CHARS:
        nxt = out[1]
        nxt["text"] = (out[0]["text"] + "\n" + nxt["text"]).strip()
        nxt["page_start"] = min(out[0]["page_start"], nxt["page_start"])
        nxt["page_end"] = max(out[0]["page_end"], nxt["page_end"])
        out = out[1:]
    return out


def _assign_ids(secs):
    """Assign section ids over resized sections: a split head + its lettered
    parts share the head's int index (3, '3#A', '3#B'); every non-split
    section just gets its running int index. A part is identified by its
    `_split` flag (set by _split_giant), never by re-guessing from the title —
    so a real top-level heading like 'V. Augmentation' is never mistaken for a
    subhead part."""
    ids, head = [], -1
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i, sec in enumerate(secs):
        if sec.get("_split") and ids:
            ids.append(f"{head}#{next(letters)}")
        else:
            head = i
            letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            ids.append(head)
    return ids
```

Update `resize_sections` to add the merge pass:

```python
def resize_sections(raw):
    """Reshape raw sections: split giants on subheads, then merge tiny stubs."""
    split = []
    for sec in raw:
        if len(sec["text"]) > GIANT_CHARS:
            split.extend(_split_giant(sec))
        else:
            split.append(sec)
    return _merge_stubs(split)
```

Update `extract_doc` to resize + assign ids:

```python
def extract_doc(path):
    """Extract one PDF, resize its sections (split giants, merge stubs), and
    build records with the head/#A id scheme."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    pages = S2._pages_text(pdf)
    raw = S2._sections_from_regex(pages)
    resized = resize_sections(raw)
    ids = _assign_ids(resized)

    records = []
    for idx, sec in zip(ids, resized):
        if len(sec["text"]) < 40:
            continue
        sec.pop("_split", None)                    # internal tag, not persisted
        records.append({
            "source": path.name, "section_idx": idx,
            **sec, "char_len": len(sec["text"]),
        })
    return records
```

- [ ] **Step 4: Run selftest to verify all cases pass**

Run:
```bash
cd RAG/rag_stage_3 && python pdf_extractor.py --selftest
```
Expected: all `[PASS]`, exit 0.

- [ ] **Step 5: Build real stage-3 sections.json and inspect the band**

Run:
```bash
cd RAG/rag_stage_3 && python pdf_extractor.py && python -c "
import json
s=json.loads(open('sections.json').read())
cl=sorted(x['char_len'] for x in s)
print('sections:',len(s),' giants>5000:',sum(1 for c in cl if c>5000),' stubs<500:',sum(1 for c in cl if c<500))
print('min/max:',cl[0],cl[-1])
print('sample split ids:',[x['section_idx'] for x in s if isinstance(x['section_idx'],str)][:6])
"
```
Expected: more sections than 111 (giants split), fewer stubs (<9), giants>5000 reduced (IEEE ones gone; non-IEEE ReAct/DPR may remain), and split ids like `['2#A','2#B',...]` present. Record the numbers — they are the Move-1/2 evidence.

- [ ] **Step 6: Commit** (stage; user commits)

```bash
git add RAG/rag_stage_3/pdf_extractor.py RAG/rag_stage_3/sections.json
# user commits: "feat(stage3): merge stubs (Move 2) + id scheme, emit sections.json"
```

---

### Task 4: Stage-3 chunker — window resized sections into chunks.json

Create the thin stage-3 chunker that imports stage-2 `build_children` and runs it on the stage-3 sections.

**Files:**
- Create: `RAG/rag_stage_3/chunker.py`

**Interfaces:**
- Consumes: stage-2 `RAG/rag_stage_2/chunker.py` member `build_children(sections) -> list[dict]` (loaded via importlib); `rag_stage_3/sections.json` from Task 3.
- Produces: `RAG/rag_stage_3/chunks.json` (child records, same shape as stage 2, `parent_id` pointing at resized parents).

- [ ] **Step 1: Write the stage-3 chunker**

```python
"""Stage 3 — window resized sections into chunks.json.

Thin wrapper: imports the frozen stage-2 chunker's `build_children` (same
120-word / 20-overlap windowing) and runs it on the RESIZED stage-3 sections.
No window logic is duplicated. Stage 2 is untouched.

Run:  python chunker.py     # build stage_3/chunks.json + size report
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
STAGE2 = HERE.parent / "rag_stage_2"
SECTIONS_PATH = HERE / "sections.json"
OUT_PATH = HERE / "chunks.json"


def _load_s2_chunker():
    spec = importlib.util.spec_from_file_location(
        "s2_chunker", STAGE2 / "chunker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if not SECTIONS_PATH.exists():
        raise SystemExit(f"{SECTIONS_PATH.name} missing — run pdf_extractor.py first")
    s2 = _load_s2_chunker()
    sections = json.loads(SECTIONS_PATH.read_text())
    children = s2.build_children(sections)
    OUT_PATH.write_text(json.dumps(children, indent=2, ensure_ascii=False))

    per_parent = Counter(c["parent_id"] for c in children)
    print(f"Parents (resized sections): {len(sections)}")
    print(f"Children (chunks):          {len(children)}")
    print(f"Children/parent: min {min(per_parent.values())} "
          f"max {max(per_parent.values())} avg {len(children)/len(per_parent):.1f}")
    print(f"Wrote {len(children)} children -> {OUT_PATH.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build chunks.json and verify parent_ids resolve**

Run:
```bash
cd RAG/rag_stage_3 && python chunker.py && python -c "
import json
secs={f\"{s['source']}#{s['section_idx']}\" for s in json.loads(open('sections.json').read())}
kids=json.loads(open('chunks.json').read())
missing={k['parent_id'] for k in kids}-secs
print('children:',len(kids),' orphan parent_ids:',len(missing))
print('max children/parent should have dropped vs stage-2 (giants split)')
"
```
Expected: `orphan parent_ids: 0` (every child points at a real resized parent), children count printed.

- [ ] **Step 3: Commit** (stage; user commits)

```bash
git add RAG/rag_stage_3/chunker.py RAG/rag_stage_3/chunks.json
# user commits: "feat(stage3): chunker windows resized sections into chunks.json"
```

---

### Task 5: Eval `--stage3` flag + stage-3 ranker builder

Add a `--stage3` flag to `run_eval.py` that scores the stage-3 files through the existing metrics, and honors it in `--selftest` label validation.

**Files:**
- Modify: `RAG/rag_stage_2/eval/run_eval.py`

**Interfaces:**
- Consumes: `rag_stage_3/sections.json`, `rag_stage_3/chunks.json`; `parent_child_rag.ParentChildIndex`, `parent_child_rag.load_index`; existing `rank_parent_child(index, query)`.
- Produces: `build_parent_child_ranker_stage3() -> (name, rank_fn)`; `--stage3` CLI flag; `load_stage3_parents() -> dict[pid -> record]` used by both the ranker and `validate_labels`.

- [ ] **Step 1: Add the stage-3 loaders + ranker builder**

Add near `build_parent_child_ranker` (after it):

```python
STAGE3_DIR = STAGE_DIR.parent / "rag_stage_3"


def load_stage3_parents():
    """parent_id -> record from stage_3/sections.json (resized parents)."""
    secs = json.loads((STAGE3_DIR / "sections.json").read_text())
    return {f"{s['source']}#{s['section_idx']}": s for s in secs}


def build_parent_child_ranker_stage3():
    """Load a ParentChildIndex over the RESIZED stage-3 files."""
    import parent_child_rag as pc
    children = json.loads((STAGE3_DIR / "chunks.json").read_text())
    parents = load_stage3_parents()
    index = pc.load_index(children, parents, cache_path=STAGE3_DIR / ".dense_cache_pc.npz")
    print(f"stage-3 parent-child index: {len(children)} children -> {len(parents)} parents\n")
    return "parent_child_s3", lambda q: rank_parent_child(index, q)
```

Note: `pc.load_index(children, parents, cache_path=...)` — confirm the signature accepts `cache_path` (it does: `def load_index(children, parents, cache_path=CACHE_PATH)`). The stage-3 cache is separate so it never collides with stage-2's.

- [ ] **Step 2: Wire the `--stage3` flag into `main` and `validate_labels`**

In `main()`, add the argument (after `--no-save`):

```python
    ap.add_argument("--stage3", action="store_true",
                    help="score the resized stage-3 chunker instead of stage-2")
```

Replace the retriever-selection block:

```python
    if args.stage3:
        name, rank_fn = build_parent_child_ranker_stage3()
    elif args.retriever == "reranked":
        sys.path.insert(0, str(STAGE_DIR.parent / "rag_stage_3"))
        from reranker import build_reranked_ranker
        name, rank_fn = build_reranked_ranker(args.rerank_depth)
    else:
        name, rank_fn = build_parent_child_ranker()
```

Make `validate_labels` honor `--stage3` — change its signature and call:

```python
def validate_labels(questions, stage3=False):
    """Every relevant_parent_id must exist in the parent store."""
    if stage3:
        parents = load_stage3_parents()
    else:
        import parent_child_rag as pc
        parents = pc.load_parents()
    ...
```

And in the `--selftest` branch of `main`:

```python
        l_ok = validate_labels(questions, stage3=args.stage3)
```

- [ ] **Step 3: Validate stage-3 labels (catch merge-orphaned gold ids)**

Run:
```bash
cd RAG/rag_stage_2/eval && python run_eval.py --selftest --stage3
```
Expected: metric unit tests PASS. Label validation either prints `labels OK` OR lists `parent_ids [that] do not exist` — those are gold ids orphaned by a merge. If any are listed, STOP and go to Step 4; otherwise skip to Step 5.

- [ ] **Step 4: (Only if Step 3 listed orphaned ids) remap gold labels**

For each listed `qid: pid`, open `RAG/rag_stage_2/eval/qa.json`, find that question, and repoint the stale `relevant_parent_ids` entry to the surviving parent that now contains the answer text. Find it:

```bash
cd RAG/rag_stage_3 && python -c "
import json
q='<the answer keyword or phrase>'
for s in json.loads(open('sections.json').read()):
    if q.lower() in s['text'].lower():
        print(s['source']+'#'+str(s['section_idx']), repr(s['title'][:40]))
"
```
Edit `qa.json`: replace the orphaned id with the printed one. Re-run Step 3 until `labels OK`.

Note: qa.json is under `rag_stage_2/eval/` but is the **shared** gold set for the harness (not stage-2 pipeline code) — editing gold labels to track a new chunking is expected and does not violate the "don't touch stage_2 pipeline" rule.

- [ ] **Step 5: Run the stage-3 eval end-to-end**

Run:
```bash
cd RAG/rag_stage_2/eval && python run_eval.py --stage3
```
Expected: a full metrics table prints (ALL_POSITIVE / factual / paraphrase rows + negatives separation) and `saved -> results/parent_child_s3_<stamp>.json`.

- [ ] **Step 6: Commit** (stage; user commits)

```bash
git add RAG/rag_stage_2/eval/run_eval.py RAG/rag_stage_2/eval/qa.json
# user commits: "feat(eval): --stage3 flag scores resized chunker"
```

---

### Task 6: A/B comparison + verdict

Run baseline and stage-3 evals back to back, compare, and record the verdict.

**Files:**
- Modify: none (measurement only). Optionally append findings to the spec or a results note.

**Interfaces:**
- Consumes: `run_eval.py` (stage-2 default and `--stage3`).

- [ ] **Step 1: Run both evals and capture the tables**

Run:
```bash
cd RAG/rag_stage_2/eval && echo "=== STAGE 2 BASELINE ===" && python run_eval.py --k 1 3 5 8 && echo "=== STAGE 3 RESIZED ===" && python run_eval.py --stage3 --k 1 3 5 8
```
Expected: two tables. Compare `recall@k`, `hit@k`, `mrr` for ALL_POSITIVE / factual / paraphrase, plus the negatives separation.

- [ ] **Step 2: Record the verdict**

Write a 3–5 line summary: did stage-3 recall@k / hit@k / MRR beat stage-2? By how much? Did the negatives separation hold? Did the char_len band actually tighten (from Task 3 Step 5)? If stage-3 regressed, note the likely cause (over-fragmentation of a giant, or a merge that buried an answer) — the resize is a measured experiment, kept only if it wins.

- [ ] **Step 3: Update the stage-3 memory**

Update `MEMORY.md` pointer / `rag-stage3-reranking.md` (or a new `rag-stage3-resize.md`) with the A/B outcome so the next session knows whether resize is kept. (This is memory upkeep, not a git commit.)

---

## Self-Review

**Spec coverage:**
- Import-extend extractor → Task 1. ✓
- Move 1 split + SUBHEAD_RE + all four guards → Task 2. ✓
- Move 2 merge stubs → Task 3. ✓
- Id scheme `3` / `3#A` → Task 3 (`_assign_ids`). ✓
- Page range inherited on split → Task 2 (`_split_giant`). ✓
- Stage-3 chunker via imported `build_children` → Task 4. ✓
- Eval `--stage3` flag + selftest honoring it → Task 5. ✓
- Gold-label caveat / remap → Task 5 Steps 3–4. ✓
- A/B + verdict → Task 6. ✓
- Non-goals (stage-2 untouched, window size unchanged) → Global Constraints + import-only reuse. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; the one `<answer keyword>` in Task 5 Step 4 is a genuine runtime input the engineer supplies per orphaned question, not a code placeholder.

**Type consistency:** `_is_subhead(m, last_letter, last_num)`, `_split_giant(sec)`, `_merge_stubs(secs)`, `_assign_ids(secs)`, `resize_sections(raw)`, `build_children(sections)`, `load_index(children, parents, cache_path=...)`, `rank_parent_child(index, query)`, `build_parent_child_ranker_stage3()`, `load_stage3_parents()` — names/signatures match across tasks and match the real stage-2 signatures verified in the codebase.
