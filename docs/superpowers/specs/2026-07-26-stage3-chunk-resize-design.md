# Stage 3 — Chunk-resize chunker (split giants + merge stubs)

_2026-07-26_

## Problem

Stage-2 parents (sections) span **69–22021 chars** — wildly uneven. Giant
sections split into many children that dilute each other's representation;
tiny stub sections (69/103/132 ch) add noise. Uneven parent size → uneven
child representation → the rank-8 retrieval misses we see in eval.

Goal: bring every parent into a **~600–5000 char band** so child
representation is even, and recall@k / hit@8 climb.

## Approach

**Import-and-extend** — stage 3 reuses stage-2's frozen primitives, adding
only the new resize logic. No code duplication; one source of truth for
section detection.

- `RAG/rag_stage_3/pdf_extractor.py` — imports the stage-2 extractor by
  explicit file path (see below), reuses `_pages_text`,
  `_sections_from_regex`, `DOCS_DIR`, `_is_heading`. Adds `SUBHEAD_RE`,
  `resize_sections()` (Move 1/2), a new `extract_doc` that runs
  `s2._sections_from_regex(pages)` → `resize_sections` → records, and its own
  `OUT_PATH` → **`stage_3/sections.json`**.
- `RAG/rag_stage_3/chunker.py` — imports the stage-2 chunker's
  `build_children`, runs it on `stage_3/sections.json`, writes
  **`stage_3/chunks.json`**. ~15 lines, no copied window logic.

**Import mechanism.** Both stage-2 files share their basenames with the
stage-3 files (`pdf_extractor.py`, `chunker.py`). Running from `rag_stage_3/`,
a plain `import pdf_extractor` resolves to the stage-3 file itself. So load
stage-2 modules by explicit path via `importlib.util.spec_from_file_location`
— e.g. load `../rag_stage_2/pdf_extractor.py` as module `s2_extractor`. No
rename, no `sys.path` ordering games, and **stage-2 stays untouched**.

**Stage-2 is untouched** (`pdf_extractor.py`, `chunker.py`, `sections.json`,
`chunks.json` all unchanged). Stage 3 is a parallel chunking strategy we A/B
against stage 2 with the existing eval harness. Because stage 2 is a frozen
baseline, the import coupling carries near-zero drift risk.

Resize lives in the extractor because the extractor is what *generates*
sections, and Move 1 (subhead split) is literally a second tier of heading
detection — it belongs next to `HEADING_RE` / `_is_heading`, reusing the same
detection primitives.

### Data flow

```
docs/*.pdf  (raw corpus, shared)
        │  rag_stage_3/pdf_extractor.py
        ▼
  section detection (HEADING_RE + _is_heading)   ── unchanged from stage 2
        ▼
  Move 1: split giants (char_len > 5000)         ── SUBHEAD_RE + guards
        ▼
  Move 2: merge stubs (char_len < 500)           ── fold into same-source sibling
        ▼
stage_3/sections.json  (reshaped parents)
        │  rag_stage_3/chunker.py (copy, repointed)
        ▼
  _child_windows (unchanged, 120-word / 20 overlap)
        ▼
stage_3/chunks.json    (children, parent_id -> reshaped parent)
```

### Move 1 — split giants (runs first)

Only fires when `char_len > 5000` — sub-threshold sections (DPR, Attention,
well-sized Arabic docs) are **never touched**.

Second-tier IEEE subhead regex:

```python
SUBHEAD_RE = re.compile(r"^\s*([A-Z]|\d+\))\.?\s+([A-Z][A-Za-z][^\n]{0,50})$")
```

Guards (mirror the spirit of stage-2 `_is_heading`), all must pass:
1. **No trailing sentence punctuation** — reject lines ending in `,;:`.
2. **Letter must climb** — for the `[A-Z]` form, the letter must increase over
   the last letter subhead seen in this giant (A→B→C); a repeat or backward
   letter is a body fragment, rejected. (Same guard spirit as stage-2's
   increasing-bare-integer check.)
3. **Number small + climbing** — for the `\d+)` form, the number must be **≤9**
   (kills citation years like `2019)` seen in real data) **and** climb over the
   last number seen since the most recent letter subhead (numbers reset under
   each new letter, e.g. C → 1) 2), then D → 1) 2) again).
4. **Short Title-Case line** — the `[^\n]{0,50}` cap + leading-cap already
   enforce this via the regex.

Verified against real `sections.json`: the IEEE RAG survey (Gao 2024) splits
cleanly on `A./B./C.` + valid `1)/2)`; the `≤9` guard rejects the `2019)`
citation-year false positive. Non-IEEE giants (ReAct, DPR, Self-RAG) have no
subheads → they are **not** split (resize is partial, by design — Move 2 and
the eval still apply).

**Id hierarchy:** the head keeps the original idx; splits append the subhead
letter. `source#3` → `source#3` (head) + `source#3#A` + `source#3#B`. Because
`chunker.build_children` composes `parent_id = f"{source}#{section_idx}"`, a
string `section_idx` like `"3#A"` yields `source#3#A` for free — no id-format
change needed downstream.

**Where it runs:** as a reshape pass inside stage-3 `extract_doc`, on the
section records returned by `s2._sections_from_regex(pages)`, *before*
`section_idx` assignment and the `<40 char` sliver drop.

**Page ranges:** splits work on the giant section's text lines, which no
longer carry per-line page numbers (the returned dict has only
`page_start`/`page_end`). Each split therefore **inherits the giant's full
page range**. This is a slight over-range but never points at the wrong
document; pages are cosmetic here (printed in `answer()`, not scored by the
eval, which ranks by parent_id). Keeping the DRY import of
`_sections_from_regex` is worth more than exact sub-page numbers.

### Move 2 — merge stubs (runs second)

Split-then-merge is **self-correcting**: a split can emit a small tail
(`#3#B`); merge then folds it into its nearest same-source sibling. Any
parent with `char_len < 500` is folded into the nearest same-source sibling
(prefer the preceding sibling; if none, the following one), concatenating
text and widening the page range (`min` start, `max` end). The absorbed
stub's idx disappears.

### Target band

Every surviving parent aims for **~600–5000 ch** (was 69–22021). Even size →
even child representation → rank-8 misses climb.

## Eval reuse

Existing harness, existing `qa.json`, existing metrics. Add a `--stage3`
boolean flag to `run_eval.py`:

- Without it: current behavior (stage-2 files via `parent_child_rag`).
- With it: `build_parent_child_ranker_stage3()` loads `stage_3/sections.json`
  + `stage_3/chunks.json`, builds a `ParentChildIndex` (reused from
  `parent_child_rag`), returns the same `(name, rank_fn)` shape. Metrics,
  aggregation, report, and save path are unchanged.
- `--selftest` also honors `--stage3` so labels are validated against the
  stage-3 parent store.

A/B run:

```
python eval/run_eval.py                # stage 2 baseline
python eval/run_eval.py --stage3       # stage 3 resized chunker
```

## Gold-label caveat (not a blocker)

Merge removes stub idxs; a `qa.json` `relevant_parent_id` pointing at an
absorbed stub will no longer resolve. This is **caught loudly** by
`--selftest --stage3` (label validation lists any unresolved id). Mitigation:
run selftest first; if any label breaks, remap those few by hand (point them
at the surviving merged sibling) before trusting the A/B. Split never removes
the head idx, so split alone does not orphan labels — only merge can.

## Non-goals

- No change to stage-2 anything.
- No change to child window size / overlap (`_child_windows` reused verbatim).
- No new qa questions (separate effort).
- No abstain/refusal threshold (separate effort).

## Success criteria

1. `python rag_stage_3/pdf_extractor.py` writes `stage_3/sections.json` with
   parent char_len collapsed toward 600–5000 (report min/max/histogram, prove
   the tails shrank vs stage 2's 69–22021). `python rag_stage_3/chunker.py`
   then writes `stage_3/chunks.json`.
2. `python eval/run_eval.py --selftest --stage3` passes (labels resolve; if
   not, the broken ids are remapped and it then passes).
3. `python eval/run_eval.py --stage3` runs and produces a results file.
4. A/B: stage-3 recall@k / hit@k / MRR ≥ stage-2 baseline on the eval's
   reported cutoffs (k=1,3,5 + ceiling 20,50; pass `--k 8` if the rank-8
   view is wanted explicitly). If it regresses, the resize is reverted or
   retuned — measured, not assumed.
