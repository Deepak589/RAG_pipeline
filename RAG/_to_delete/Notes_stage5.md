# Notes — Stage 5 (running log)

Reasoning log for Stage 5 (740-doc corpus, LangChain + pgvector + bge-m3).
Newest entry on top. This is the canonical Stage-5 update file.

---

## 2026-08-16 — Golden-set triage (qa stage5_v2)

**Validate re-run WITH retriever up** (`retriever up (pgvector), 497563 children`):
**68/178 flagged** — 61 multi-label, 7 negatives, 2 ungrounded. (First validate had
run blind — retriever was down — and flagged only 9; this is the real pass.)
Eval unchanged: R@5 0.704 (still on raw labels — number is provisional until v3).

**Triage of the 68 (source-match + question-shape buckets):**

| bucket | n | action |
|---|---|---|
| ADD_SIBLING | 36 | gold's own same-doc sibling outranked it → add sibling to label (verify answer is in it) |
| DROP_GENERIC | 5 | SEC boilerplate, not uniquely answerable |
| GRAY | 20 | hand-read — 16 SEC, 2 arxiv, 1 rtd, 1 wiki |
| DROP_NEG | 7 | negatives the retriever thinks are answerable |
| DROP_UNGROUNDED | 2 | answer not even in labeled parent (hallucinated) |
| KEEP_MISS | 0 | — |

**Key findings:**
- The flags are NOT all bad labels. 36 are a genuine **multi-label undercount** — the
  answer legitimately lives in a same-doc sibling parent that outranked gold. Caused by
  char-window + overlap making near-duplicate adjacent parents. So 0.704 is *understated*;
  real answers weren't being credited.
- The bad ones are **SEC generic questions** ("the Company's revenue increase", "unrecognized
  tax benefits") — answerable by any 10-K. Root cause = qa_gen prompt never forces
  entity-anchoring. **NOT a model-size issue** — labels were generated with qwen2.5:14b, so
  a bigger model won't fix it. Prompt is the only lever.
- `KEEP_MISS = 0`: almost every specific-but-missed question (e.g. `django save()` whose
  rank-1 was `xgboost`) still had a same-doc sibling retrieved → lands in ADD_SIBLING. Means
  true retrieval is healthier than 0.704 suggests; the label just didn't credit the sibling.

**Delivered:** `triage_apply.py` (run locally, needs sections.json for the groundedness check
my sandbox couldn't stage — 183MB timed out). It auto-applies ONLY verified edits: adds a
same-doc sibling **only if that sibling's text actually contains the answer** (≥0.5 token
overlap), and drops the 2 ungrounded. DROP_GENERIC / GRAY / negatives are routed to
`triage_todo.md` for hand-verify — nothing judgemental is auto-guessed.

**Next (in order):**
1. `python triage_apply.py --qa qa.validated.json --sections sections.json --out qa_stage5_v3.json --version stage5_v3`
2. Hand-resolve `triage_todo.md`: 20 gray + 7 negatives → drop generic, keep specific-miss.
3. Re-run eval on `qa_stage5_v3.json` → **real Stage-5 baseline** (expect R@1/R@5 up once
   verified siblings are credited).
4. Then decide: if gray SEC generics are many → hardened **entity-anchored** qa_gen prompt +
   regenerate SEC factuals (so dropping them doesn't gut SEC coverage).

**Still open:** blocker #2 — 39 scanned/archive docs dead (docling load-fail), lane unmeasured.
Char-window chunking is the standing regression behind BOTH the multi-label noise and the
generic-question ambiguity; structure-aware (Docling) for digital-PDF lanes is the deferred fix.

---
