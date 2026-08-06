# RAG From Scratch — Learning Log

A step-by-step record of how this RAG system was built and improved, with the
reasoning behind each decision and the measured result. Read top to bottom: each
stage builds on the last, and every improvement is tied to a number, not a hunch.

**Guiding philosophy:** no frameworks. Everything is hand-rolled with `numpy` +
`sentence-transformers` + stdlib + a local Ollama LLM. The point is to understand
*how* retrieval works, not to configure a library that hides it.

---

## Stage 1 — Naive RAG (the baseline)

**What:** the simplest thing that works. Split a Markdown guide into word-chunks,
embed them, retrieve the top-k by similarity, stuff them into a prompt, generate.

**Two retrievers, both from scratch:**
- **TF-IDF** (lexical): `TF = term count / chunk length`,
  `IDF = log((1+N)/(1+df)) + 1`, rows L2-normalized so cosine = dot product.
  Matches on *exact words*.
- **Dense** (semantic): a bi-encoder (`all-MiniLM-L6-v2`) encodes each chunk to a
  vector; retrieval is cosine similarity. Matches on *meaning*.

**Lesson — lexical vs semantic are different tools.** TF-IDF finds exact terms but
is blind to synonyms ("car" ≠ "automobile"). Dense finds meaning but can miss a
rare exact keyword. This tension is the whole reason hybrid retrieval exists later.

> Note on names: we used the bi-encoder *architecture* with a general pretrained
> model, **not** DPR. DPR (Karpukhin 2020) is one specific *trained instance* of
> that architecture. Same family, different weights.

---

## Stage 2 — Parent-Child Dense RAG over real PDFs

Baseline was too easy (one clean Markdown doc). Stage 2 moves to a real corpus:
**7 academic PDFs** (Attention, DPR, RAG, REALM, Self-RAG, ReAct, the RAG survey).

### The core idea: parent-child chunking

A small chunk embeds to a **sharp, specific** vector, so retrieval lands on the
right spot. But a tiny window loses context the LLM needs to answer. Solution:

- **Children** = 120-word windows (20-word overlap) → what we *search over*.
- **Parents** = whole sections → what we *feed the LLM*.

Retrieve precise children, then hand their **parent section** to the generator.
Precision of small chunks, context of large ones.

**Pipeline (each step writes a file the next one reads):**
```
pdf_extractor.py → sections.json  (parents: 111 sections)
chunker.py       → chunks.json    (children: 500 windows)
parent_child_rag.py → embed children, retrieve, collapse to parents, generate
```

### Extraction decision 1 — regex headings beat the PDF's table of contents

PDF bookmarks (TOC) only give **page-level** granularity, so multiple headings on
one page each grabbed the whole page → **82 duplicate sections**. Running a regex
on numbered headings *in the text* (`1 Introduction`, `2.1 Method`, and Roman
`II. Overview` for the IEEE-style survey) is character-precise → **0 duplicates**.

**Lesson:** the obvious metadata (bookmarks) isn't always the right signal. Look
at what granularity you actually need.

### Extraction decision 2 — cut everything after "References"

Reference lists have no answerable content, and their `year`/URL lines
false-match the heading regex, dumping an entire bibliography into one giant noisy
section. Truncating each doc at its standalone `References`/`Bibliography` line
removed that noise. All 7 PDFs have a clean references heading.

**Lesson:** garbage in the index doesn't just waste space — it actively pollutes
retrieval and inflates your denominators. Clean the corpus before measuring.

### Deliverable 0 — clean section detection (do this *before* evaluating)

One junk parent survived: a **footnote** (`3 Note that we still fine-tune...`)
whose leading `3` + sentence text satisfied the heading regex. Two guards fixed it:
1. **Reject titles ending in `,` `;` `:`** — real headings don't; footnote
   fragments do.
2. **Require increasing bare-integer heading numbers** — a second `3` after
   `3 Approach` is a false match, not section 3 again.

**Lesson — a poisoned index makes your eval measure the *extractor* bug, not the
retriever.** Fix data quality first, or every number downstream is a lie.

---

## The Eval Harness — judge by numbers, not vibes

Before improving retrieval you need to *measure* it, or "better" is just opinion.
Built a **retriever-agnostic** harness (`rag_stage_2/eval/run_eval.py`): any
retriever that turns a query into a ranked list of parent IDs can be scored
against the same golden question set.

**The golden set (`qa.json`)** — hand-written questions, each labeled with the
section(s) that answer it, in three types:
- **factual** — uses the document's own vocabulary.
- **paraphrase** — same target, reworded with synonyms → tests whether dense
  retrieval really beats lexical.
- **negative** — the answer is **not** in the corpus → a good retriever should
  score it low and a good system should refuse.

**Metrics (deterministic, no LLM):**
| metric | meaning |
|--------|---------|
| **Recall@k** | fraction of relevant parents found in the top-k |
| **Hit@k** | 1 if *any* relevant parent is in the top-k |
| **MRR** | 1 / rank of the first relevant parent (rewards ranking it high) |
| **Negative separation** | mean top-1 score on answerable − on negatives (bigger gap = easier to set a refusal threshold) |

**Lesson — build the ruler before you try to grow.** The harness is the single
most important tool here: it turns "I think this helps" into "+10 points on
hit@3." Every stage-3 change is an A/B against it.

---

## Diagnosing the baseline — *where* is retrieval failing?

Stage-2 dense baseline (21 answerable questions), measured at increasing k:

| k | hit@k |
|---|-------|
| 1 | 0.29 |
| 3 | 0.48 |
| 5 | 0.57 |
| 10 | 0.76 |
| **20** | **0.86** |

**The key insight:** hit@20 = 0.86 but hit@1 = 0.29. The right section is almost
always *in the candidate pool* — it's just **ranked too low**. This is a **ranking
problem, not a recall problem**.

*Why:* all 7 papers are about RAG/retrieval/attention, so their sections are
near-identical in embedding space. Cosine similarity can't tell the *right*
RAG-paper section from the four *other* RAG-paper sections that look just like it.

**Lesson — don't guess which fix to build. Let the diagnostic pick.** A recall
problem needs a wider net (hybrid/BM25). A ranking problem needs a better judge
(reranking). We had a ranking problem → reranking first.

---

## Stage 3, Step 1 — Cross-Encoder Reranking

**The fix for a ranking problem.** A bi-encoder embeds query and passage
*separately* (fast, but coarse). A **cross-encoder** feeds `(query, passage)` in
*together* and outputs a relevance score — much sharper at discriminating
near-duplicates, but too slow to run over all 500 chunks.

**So: retrieve wide with the cheap model, re-rank narrow with the expensive one.**
```
query → dense top-N children → cross-encoder rescores each (query, child)
      → collapse to parents by best score → ranked parents
```

**Design decisions (each one a real choice):**
- **Rerank children, not parents** — chunks fit the cross-encoder's ~512-token
  window; full sections would be truncated.
- **Model:** `cross-encoder/ms-marco-MiniLM-L-12-v2` (via sentence-transformers).
- **Depth N = 20** — see the lesson below.
- **Eval-first** — built as a new retriever in the harness and measured *before*
  wiring it into the live pipeline. Prove the win, then integrate.

### Lesson — depth is a knob, and more is not better

Swept N from 1 to 50:
- **N=1** → identical to dense (nothing to reorder) — a good sanity check.
- **N≈20** → best.
- **N=50** → hit@1 got *worse* (0.24). Too deep a pool feeds the cross-encoder
  more near-duplicate wrong-paper chunks, and it occasionally promotes one.

**Takeaway:** a wider candidate pool raises the *ceiling* but also the *noise*.
Tune it; don't max it.

### Lesson — small eval sets lie (the most important one)

First measured on **21** answerable questions → looked like a clean win
everywhere, including paraphrase hit@1 jumping 0.40 → 0.60.

Then grew the set to **30 answerable + 5 negatives** and re-ran. The story
changed:

| metric | dense | reranked@20 | Δ | reading |
|--------|-------|-------------|-----|---------|
| hit@1 | 0.400 | 0.367 | −0.03 | no top-1 gain |
| **hit@3** | **0.533** | **0.633** | **+0.10** | real win |
| hit@5 | 0.633 | 0.700 | +0.07 | real win |
| hit@10 | 0.833 | 0.800 | −0.03 | slight ceiling drop |
| MRR | 0.522 | 0.528 | +0.01 | flat |
| neg separation | +0.22 | **+7.8** | — | huge abstain signal |

The paraphrase spike **vanished** — it was noise from 5 samples. The `qa.json`
readme literally warns "grow to 50+ before trusting small deltas," and this was
that warning coming true in real time.

**Takeaway:** with tiny test sets, a few lucky questions swing every metric. Grow
the ruler before you trust it.

### Lesson — measure the metric your system actually uses

By the original hit@1 gate, reranking looked like a wash. But the **live pipeline
feeds the top-3 parents to the LLM** (`TOP_PARENTS = 3`), so **hit@3 is the metric
that matters** — and it went 0.53 → 0.63. Reranking gets the right section into
the LLM's context 10 points more often. That's a genuine, shippable win.

**Takeaway:** pick your success metric from how the product *consumes* retrieval,
not from a generic leaderboard number.

### Lesson — audit *which* queries moved, not just the average

A net −1 on recall@1 could be noise, or it could be the reranker actively
demoting correct answers. The only way to know is a **flip list**: per query,
the rank of the correct parent under dense vs reranked.

| class | n | what it means |
|-------|---|---------------|
| DEMOTION | 3 | dense had correct @1, reranker pushed it to @2 |
| PROMOTION | 2 | reranker lifted correct to @1 |
| KEPT@1 | 9 | both @1 |
| OTHER | 16 | moved within lower ranks |

The −1 was **real**: 3 genuine demotions vs 2 promotions, not churn. But the
demotions were *contained* — all landed at rank 2, still inside top-3, which is
why hit@3 rose while hit@1 fell.

The flip list also exposed a **structural cost the averages hid**: four queries
went to rank `None` under reranking. Their correct parent sat at dense rank 15–31
— *beyond the depth-20 child cutoff* — so the reranker never saw it and **dropped
it from the output entirely**. Reranking a truncated pool can only reorder what's
in the pool; anything below the cutoff is silently evicted, which lowers the
ceiling (this is why hit@10 fell 0.83→0.80).

**Takeaways:**
- **Averages hide flips.** Always inspect the per-query movement before trusting
  a delta. `scratchpad/flip_diag.py` (dense vs reranked rank per query) is the tool.
- **Never let a stage drop the recall ceiling silently.** The harness now *always*
  reports recall@20 and recall@50 (`CEILING_KS`), so a good hit@3 can't mask a
  capped ceiling. Dense recall@50 = 1.00 (every answer is reachable); any reranker
  that reports < 1.00 there has thrown away reachable answers.
- **The fix (implemented):** rerank the top-N head but *preserve the dense tail*
  below it, so reranking can reorder but never evict. Because CE scores and cosine
  aren't on the same scale, the two halves are joined by **rank** (concatenate the
  reranked head, then the dense-ordered remainder), not by score. Result: recall@20/@50
  restored to the dense ceiling (0.90 / 1.00), hit@1/@3/@5 unchanged (head is
  untouched), MRR nudged up (0.528→0.534) as the evicted answers regained a finite
  rank. A pure bookkeeping win — the ceiling metric is honest again at zero quality
  cost.

### Bonus find — a strong refusal signal

The cross-encoder's scores separate in-corpus from out-of-corpus queries far more
sharply than cosine (+7.8 vs +0.22). That's the basis for a future **abstain
threshold**: below some score, the system says "I don't know" instead of making
something up.

---

## Stage 3, Step 2 — Chunk Resize (split giants, merge stubs)

**The next weakest link, found by inspection, not eval.** Stage-2 parents ranged
**69–22,021 chars** — a 300× spread. Giant sections (Gao `III RETRIEVAL` ≈ 22k)
get diluted across many children that don't represent the whole section; tiny
stubs (69/103/132 ch) are too thin to win on their own merits. Goal: squeeze
every parent into a **~600–5000 char band** so child representation is even.

**Architecture — import-and-extend, not copy-and-modify.** `rag_stage_3/` loads
the frozen stage-2 `pdf_extractor.py`/`chunker.py` **by explicit file path**
(`importlib.util.spec_from_file_location`, since both stages use the same
filenames) and reuses their heading-detection primitives verbatim, adding only
the resize pass. Stage 2 stays untouched as the A/B baseline — this is the same
principle as Deliverable 0's "clean the data first," applied to architecture:
freeze what works, don't risk it while iterating on the next idea.

**Resize = split then merge, in that order:**
1. **Split giants** (`char_len > 5000`) on a second-tier IEEE subhead regex
   (`A.`/`B.`/`C.` — letters only). Guards mirror stage-2's heading checks:
   reject lines ending in `,;:`, require the letter to climb (A→B→C, not a
   repeat/backward match = body-text false positive).
2. **Merge stubs** (`char_len < 500`) into the nearest same-source sibling.
   Split-then-merge is self-correcting: a split's small tail gets folded right
   back in.

Result: **111 → 118 sections**, char_len band **69–22,021 → 536–9,972**, zero
stubs left. The structural goal was hit cleanly.

### Lesson — re-chunking invalidates every gold label (re-learned the hard way)

First dense-only eval on stage-3 *looked like a regression* (recall@1 0.38→0.28,
MRR 0.52→0.42) — alarming, since nothing about retrieval changed, only chunk
boundaries. Root-caused to **three stacked problems**, most of it not a real
regression:

1. **Stale `qa.json` labels (the dominant cause).** Splitting/merging renumbers
   `section_idx`, so old `relevant_parent_id`s point at sections that moved,
   merged away, or now mean something different. One question scored
   recall@50 = 0.0 — mathematically impossible for a real retrieval miss, and
   the tell that the label itself was broken, not the retriever.
2. **Two real code bugs in the resizer**, caught by the label breakage:
   - `_merge_within_group`'s "tiny head folds forward" case adopted the
     *next* fragment's title instead of keeping the section's own — silently
     mislabeling merged sections.
   - `SUBHEAD_RE` originally matched `\d+)` as well as `A.`/`B.` letters, so
     numbered list items inside a subsection (`1) New Modules`) got promoted
     to peer parents — an over-split.
3. **Compared dense-vs-dense, forgot the reranker.** The stage-3 run being
   judged was un-reranked; recall@8 lined up almost exactly with the
   reranker's candidate window, i.e. the "regression" was partly just judging
   the wrong half of the pipeline.

**Fix, in order, one knob at a time:** restrict `SUBHEAD_RE` to letters only;
fix the forward-merge to always keep its own title; re-point the broken
`qa.json` labels at the new section ids; re-baseline dense-s3 on the corrected
labels; only then compare rerank-s3. Both code bugs are fixed and verified by
`--selftest` (`RAG/rag_stage_3/pdf_extractor.py`).

**Takeaway:** changing chunking changes every downstream id. A chunking change
is not just a retrieval-quality experiment — it's a **schema migration on the
gold set**, and skipping the relabel step makes the eval measure its own
broken labels, not the retriever. Same root lesson as Deliverable 0, one layer
up the stack.

### Result — after the fix, a real (small) A/B

| retriever | hit@1 | hit@3 | hit@5 | MRR | recall@20 |
|-----------|-------|-------|-------|-----|-----------|
| stage-2 dense | 0.40 | 0.53 | 0.63 | 0.52 | 0.90 |
| stage-3 dense (resized) | 0.37 | 0.53 | 0.63 | 0.49 | 0.87 |
| stage-2 + rerank | 0.37 | 0.63 | 0.70 | 0.53 | 0.90 |
| **stage-3 + rerank (resized)** | 0.37 | **0.67** | 0.70 | 0.52 | 0.87 |

Resizing plus reranking nudges hit@3 up again (0.63 → 0.67) over the already-
reranked stage-2 baseline — but recall@20 (the honest ceiling metric from the
reranking lesson) **dropped 0.90 → 0.87**, and MRR is flat-to-slightly-down.
**Verdict: inconclusive, not a clear win.** The resize fixed a real structural
problem (size variance) but the eval set (30 Q) is small enough, and the effect
size close enough to noise, that it doesn't clear its own bar yet.

**Follow-up:** ran `rag_stage_3/diagnose_demotions.py` on the 3 stage-3 rerank
demotions (dense rank vs CE score vs final rank, per child). Finding: the CE
isn't confused *across* papers, it's confused **within the same paper** —
twice it promoted a sibling section of the *correct* document (REALM's intro
over REALM's training section; a different Lewis section over the gold one)
ahead of the actual answer. Paraphrased queries make it worse (lose the
lexical anchor that would've kept CE pointed at the right section). Two of
three demotions stayed contained at rank 2 (inside top-3, low cost); one fell
from rank 4 to rank 7 — a real loss. **Read:** not a resize bug, a pre-existing
CE weakness the resize exposed. Didn't change the resize decision; motivated
the hybrid work below (see "why hybrid" in `hybrid.py`'s docstring).

---

## The gold-set versioning fix — `relevant_parent_ids_s3` override

The "re-chunking invalidates every gold label" lesson above needed an actual
schema, not just a one-time relabel. `qa.json` now carries a `"version"` field
(`v2-55q-2026-08-02`) and, per-question, an **optional**
`relevant_parent_ids_s3` override: when a resized stage-3 chunk boundary moved
or split the answer to a different `parent_id` than stage-2's, the question
carries both — `relevant_parent_ids` (stage-2 truth) and
`relevant_parent_ids_s3` (stage-3 truth) — instead of two separate qa files.
`run_eval.py`'s `gold_ids(q, stage3)` picks whichever applies. One golden set,
valid against both chunking strategies at once; growing the set no longer
means forking it per stage.

The set also grew **30 → 55 questions** (36 factual / 12 paraphrase / 7
negative), hand-written and grounded against the actual section text (not
generated) — the "small test sets lie" lesson from step 1, actually acted on
this time.

---

## Stage 3, Step 3 — Hybrid Retrieval (BM25 + dense, fused by RRF)

**Motivation.** The worst dense misses are exact-term queries — model names,
acronyms, numbers (`FAISS`, `995 questions/sec`, `21,015,324 passages`) — that
a bi-encoder blurs into "vaguely about retrieval speed" but lexical matching
nails outright. BM25 is dense's mirror image: exact on vocabulary, blind to
paraphrase. Fusing the two should raise the ceiling above either alone.

**Design (`RAG/rag_stage_3/hybrid.py`):**
- **BM25 from scratch** — Okapi BM25 (`k1=1.5`, `b=0.75`) over children, pure
  numpy/stdlib, no model, no network. Same "hand-roll it to understand it"
  philosophy as the stage-1 TF-IDF.
- **Fusion is rank-based (Reciprocal Rank Fusion)**, not score-based:
  `RRF(parent) = 1/(k + rank_dense) + 1/(k + rank_bm25)`, k=60. BM25 scores and
  cosine similarities live on incomparable scales — combining the *rankings*
  sidesteps that the same way stage-1's tail-preservation fix did for CE vs
  cosine.
- **Composes with reranking, doesn't replace it.** `reranker.py` gained
  `rank_hybrid_reranked`: the cross-encoder's candidate pool is now drawn from
  the *hybrid* RRF ranking instead of pure dense, so exact-term children BM25
  surfaces are actually available for the CE to promote. Retrieval (hybrid)
  and refinement (rerank) stay separate stages that compose in sequence — the
  architecture called for back in `NOTES.md` before either was built.

### Result — hybrid is the best retriever so far, and stacking rerank on top hurts

All five retrievers, same 48-answerable-question set (`v2-55q`), stage-3 chunks:

| retriever | hit@1 | hit@3 | hit@5 | MRR | recall@20 |
|-----------|-------|-------|-------|-----|-----------|
| dense only | 0.35 | 0.56 | 0.65 | 0.50 | 0.92 |
| dense + rerank | 0.44 | 0.65 | 0.71 | 0.56 | 0.94 |
| BM25 only | 0.38 | 0.71 | 0.77 | 0.56 | 0.98 |
| **hybrid (dense+BM25 RRF)** | **0.44** | 0.69 | **0.79** | **0.61** | **0.98** |
| hybrid + rerank | 0.42 | 0.69 | 0.75 | 0.57 | 0.98 |

**Hybrid alone is the best all-round retriever measured yet** — best MRR
(0.61), best hit@5 (0.79), ties best hit@1 and recall@20. BM25 alone already
beats pure dense on nearly every cut, which says something about *this*
question set (many exact-term factual questions) as much as about the corpus.

**The counter-intuitive part: reranking the hybrid pool made it slightly
worse**, not better (MRR 0.61→0.57, hit@1 0.44→0.42, hit@5 0.79→0.75). This is
the same CE weakness `diagnose_demotions.py` found above — the cross-encoder
occasionally promotes a same-document sibling over the true answer — now
applied to a pool that was already *better curated* by RRF than dense-alone
was. Reranking helped when it was fixing a worse candidate pool (dense-only);
once the pool itself got good (hybrid), the CE's own noise became the binding
constraint instead of the thing it was fixing.

**Takeaway:** a refinement stage is only a net win relative to *how bad* the
stage before it is. Don't assume reranking (or any fixed pipeline stage)
composes for free — measure it fresh every time the upstream stage changes,
same as the resize A/B taught for chunking.

---

## Stage 4 — Contextual Blurb Chunking (index-time context injection)

**The idea (Anthropic's Contextual Retrieval).** A 120-word child is often
un-interpretable on its own — the 43-word Vaswani formula window embeds to
"some math," and a bare "3.2 Retriever" fragment looks identical across five
RAG papers. Fix: before embedding, prepend a **one-sentence, LLM-written blurb**
that says where the passage lives and what it's about ("This passage is from
Vaswani sec 5.3 Optimizer, on the Adam warmup+decay schedule"). Then embed and
BM25-index `blurb + "\n" + chunk`. The parent handed to the LLM is unchanged —
this is an **index-time** trick, not a content change.

**Critically: the retriever did not change.** Stage 4 reuses the Stage-3 hybrid
(BM25 + dense + RRF) *verbatim* — `rag_stage_4/hybrid.py` imports Stage-3's BM25
and fusion by explicit path. Only the chunk content moved. That's what makes the
delta attributable: **one knob, frozen baseline.**

**Design decisions (each a real choice):**
- **Blurb context = the parent section**, not the whole document. Feeds the LLM
  paper title + section title + parent text + the child. Cheap, deterministic,
  and strong on this clean-sectioned corpus; whole-doc would blow qwen's context
  on the Gao survey for marginal gain.
- **Determinism is enforced, not assumed.** Ollama samples by default, so the
  blurb call pins `temperature=0, seed=7`. Without this, "cached and reproducible"
  is a lie — every rebuild would produce different blurbs and a different eval.
- **The cache is cost-only.** Each blurb is fingerprinted over
  `model | prompt-version | parent-text | child-text`; a change in any of them
  regenerates. The recall win comes from the blurb *text*, never from caching.
  (A subtle bug caught here: the fingerprint must include the **model** — switch
  the LLM and stale blurbs would otherwise be silently reused.)

### Lesson — a mock is a plumbing test, not a preview of the result

Before the real LLM run, a `--mock` mode (templated blurbs, no model) validated
the cache, fingerprints, and eval wiring offline. It also *predicted the wrong
outcome*: templated blurbs repeat the same author/section boilerplate on every
child of a section, which inflated BM25 doc-length and **hurt** paraphrase
(0.58 → 0.33). That scared us into building a `--raw-bm25` / `--blurb-bm25`
toggle to measure whether the blurb belongs in the lexical index at all. Good
that we did — but the mock's *number* was an artifact of templating, not a real
prediction. **Takeaway:** use mocks to test that the pipe runs; never read their
metrics as a forecast.

### Result — blurbs win, and the toggle disproved our own worry

55-question set (`v2-55q`), 48 answerable, hybrid retriever frozen:

| retriever | hit@1 | hit@3 | hit@5 | MRR | recall@20 |
|-----------|-------|-------|-------|-----|-----------|
| Stage-3 hybrid (baseline) | 0.44 | 0.69 | 0.79 | 0.61 | 0.96 |
| S4 hybrid, blurb→dense only (`--raw-bm25`) | 0.52 | 0.75 | 0.79 | 0.65 | 0.98 |
| **S4 hybrid, blurb→both (`--blurb-bm25`)** | **0.54** | **0.75** | **0.83** | **0.67** | **0.99** |

In recall terms the headline is **recall@5 0.792 → 0.833** and **recall@1
0.427 → 0.531**. **Verdict: keep Stage 4, blurb in both.** The idf-skew risk the
mock warned about did *not* materialise with real blurbs — they carry topic
keywords, not boilerplate, so putting them in BM25 helps.

### Lesson — the two runs isolate *where* the gain comes from, and it's the opposite of the hypothesis

Because `--raw-bm25` blurbs only the dense side (BM25 stays identical to Stage
3), the two runs cleanly decompose the effect:

- **Blurb on dense → moved precision, not recall.** recall@1 +0.08, MRR +0.04,
  but **recall@5 stayed at 0.792.** The blurb sharpened *where* the answer ranks.
- **Blurb on BM25 → drove the recall gain.** recall@5 0.792 → 0.833, factual@5
  0.806 → 0.861. Real keywords let lexical matching find thin chunks it missed.

NOTES called blurbs "the biggest recall lever, mainly for dense." On *this*
corpus that was wrong: the recall lever was **lexical**, and dense only bought
top-1 precision. **Takeaway:** don't trust the mechanism you assumed — design the
experiment so it tells you which half actually moved.

### Lesson — report the caveats that shrink your win

- **Paraphrase didn't budge (0.750, flat).** The bucket blurbs were *supposed*
  to help most is exactly where nothing moved; the gain came from factual + top-1.
- **Small n again.** recall@5 0.792→0.833 on 48 positives ≈ ~2 more questions.
  Modest. The recall@1 jump (+0.10 ≈ 5 questions) is the trustworthy signal.
- **The 0.85 gate was NOT hit** (0.833, ~1 question short). Cheapest closers, in
  order: sweep `--rrf-k` (free, no re-embed) → better blurb prompt/model →
  the deferred token-sizing / splitter-title fixes. Or accept 0.833 — for a
  top-3 served system, recall@1/MRR moving this much is arguably the better prize.

### Lesson — latency and storage costs are index-time here, not query-time

The ~500 LLM calls *feel* expensive (minutes on a local model), but they're a
**one-time offline build**, cached and amortised to zero per query. At serve
time the blurb is already baked into the stored vector and BM25 index — **no
extra query-time latency, and the vector count is unchanged, so no index-storage
growth.** This is the whole reason contextual retrieval prepends at index time
instead of doing anything per query. The cost that *does* scale is regeneration:
1 call per chunk means a 1000-doc corpus is a multi-hour batch — which is why the
generator was made parallel (`--workers`) and resumable (cache flushes every 20,
so a Ctrl-C never loses more than 20 blurbs).

---

## Running scoreboard

**30-question set (`v1`, stage-2 chunk ids):**

| stage | retriever | hit@1 | hit@3 | hit@5 | MRR |
|-------|-----------|-------|-------|-------|-----|
| 2 | dense (parent-child) | 0.40 | 0.53 | 0.63 | 0.52 |
| 3.1 | + cross-encoder rerank (N=20) | 0.37 | 0.63 | 0.70 | 0.53 |
| 3.2 | resized chunks, dense only | 0.37 | 0.53 | 0.63 | 0.49 |
| 3.2 | resized chunks + rerank | 0.37 | 0.67 | 0.70 | 0.52 |

**55-question set (`v2-55q`, stage-3 resized chunks, 48 answerable):**

| retriever | hit@1 | hit@3 | hit@5 | MRR | recall@20 |
|-----------|-------|-------|-------|-----|-----------|
| dense only | 0.35 | 0.56 | 0.65 | 0.50 | 0.92 |
| dense + rerank | 0.44 | 0.65 | 0.71 | 0.56 | 0.94 |
| BM25 only | 0.38 | 0.71 | 0.77 | 0.56 | 0.98 |
| hybrid (dense+BM25 RRF) — Stage 3 | 0.44 | 0.69 | 0.79 | 0.61 | 0.96 |
| hybrid + rerank | 0.42 | 0.69 | 0.75 | 0.57 | 0.98 |
| S4 hybrid, blurb→dense only | 0.52 | 0.75 | 0.79 | 0.65 | 0.98 |
| **S4 hybrid, blurb→both (current best)** | **0.54** | **0.75** | **0.83** | **0.67** | **0.99** |

The two tables aren't directly comparable (different question sets) — the v2
table is the one to trust going forward; v1 numbers are kept for the historical
trail of how each idea was diagnosed. **Current serving pipeline = Stage-4
hybrid over blurbed chunks, blurb in both dense + BM25.** New baseline every
later stage must beat: recall@5 0.833, recall@1 0.531, MRR 0.672.

---

## Principles distilled (the reusable takeaways)

1. **Clean the data before you measure** — a bad index measures the extractor, not
   the retriever.
2. **Build the ruler first** — a retriever-agnostic eval harness turns opinion into
   evidence.
3. **Diagnose before you fix** — recall problem vs ranking problem need opposite
   fixes; the hit@k-vs-k curve tells you which.
4. **Small test sets lie** — grow the gold set until deltas stop swinging.
5. **Measure what the product consumes** — hit@3 mattered because the LLM gets 3
   parents; hit@1 was the wrong gate.
6. **More candidates ≠ better** — depth raises both ceiling and noise; tune it.
7. **Report honestly** — the reranking win shrank when the data grew, and that got
   written down, not buried.
8. **Re-chunking is a schema migration on the gold set** — a changed `section_idx`
   silently invalidates old labels; a recall@50 = 0.0 is a red flag for a broken
   label, not a broken retriever. Solved structurally via a per-question
   `relevant_parent_ids_s3` override rather than forking the qa file.
9. **A refinement stage's win is relative to the stage before it** — reranking
   helped a mediocre dense-only pool a lot; it slightly *hurt* the
   already-better hybrid pool. Re-measure every fixed pipeline stage whenever
   what feeds it changes.
10. **Freeze the retriever to attribute a chunking win** — Stage 4 changed only
    the chunk content and reused the hybrid retriever verbatim, so the +0.041
    recall@5 is provably the blurb's doing and nothing else. One knob, frozen
    baseline, every time.
11. **Design experiments to reveal the mechanism, not just the score** — the
    `--raw-bm25` toggle split the blurb's effect into a dense half (precision)
    and a BM25 half (recall) and showed the recall lever was *lexical*, the
    opposite of the going-in assumption. A single blurbed-both number would
    have hidden that.
12. **Mocks test the pipe, not the outcome** — templated mock blurbs ran the
    plumbing but predicted a paraphrase drop that was a boilerplate artifact,
    not a real forecast. Read a mock's plumbing, never its metrics.
13. **Push cost to index time when you can** — contextual blurbs add real
    build-time cost (1 LLM call/chunk) but *zero* query-time latency and *zero*
    index-storage growth, because the work is baked into vectors offline. Know
    which axis a cost lands on before calling a technique "expensive."

---

## What's next (planned, not built)

- **Close the last ~1 question to the 0.85 gate, or call it.** Stage 4 lands at
  recall@5 0.833. Cheapest first: sweep `--rrf-k` (free, no re-embed). Then a
  better blurb prompt/model, then the deferred token-sizing / splitter-title
  fixes. Or accept 0.833 and move on — recall@1/MRR are the better prize here.
- **Audit which questions the blurb flipped** before treating +0.041 as solid —
  the recall@5 gain is ~2 questions on n=48. A per-query flip list (same tool
  discipline as the reranker audit) separates a real win from two lucky hits.
- **Guard blurb quality at scale** — a wrong blurb mis-places a vector. A
  cheap `--audit` pass (flag blurbs that are empty, too long, or don't name the
  paper/section) is worth adding before the corpus grows.
- **Decide hybrid's place in the live pipeline.** Eval says hybrid-alone beats
  hybrid+rerank and dense+rerank on MRR/hit@5 — `parent_child_rag.py`'s
  `--rerank` CLI flag doesn't have a `--hybrid` counterpart yet, and nothing in
  the interactive pipeline is repointed at stage-3/4 chunks at all.
- **Explain the CE-demotes-same-document-sibling pattern more thoroughly** —
  `diagnose_demotions.py` found it on 3 examples; still don't know if it's
  systemic across the corpus or specific to homogeneous RAG-paper sections.
  Would inform whether a stronger/different cross-encoder is worth trying.
- **Resize verdict still open** — chunk resize alone (no hybrid) was
  inconclusive (hit@3 up, recall@20 ceiling down) on the old 30Q set; rerun
  that specific comparison on the new 55Q set before deciding keep/retune/revert.
- **Abstain threshold** using the negative-separation signal (BM25/hybrid
  negative-gap not yet measured — only cross-encoder's was).
- **Housekeeping:** gitignore committed junk (`*.npz` caches, `__pycache__/`);
  decide whether to LFS or drop the raw PDFs.
- Later stages: query optimization (HyDE / multi-query), context engineering.
