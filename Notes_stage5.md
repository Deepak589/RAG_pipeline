# RAG_dev — Stage 5 notes

Split out from NOTES.md — stage 5 (scale & real-time roadmap) content only.

---

## Stage 5 — scale & real-time roadmap (planned, not started)

Stage 4 stays at the 7-PDF corpus on the current numpy pipeline. Stage 5 is the
jump to a large, growing corpus (~1000 docs ≈ 70k children at today's ratio) and
a rebuild toward a real serving system. Captured now so it isn't re-derived later.

**Reframe first — three axes, not one. Two of them fight each other.**
- *Scale* (many docs) = an INFRA problem → vector DB, ANN index, hybrid engine.
- *Real-time* (low query latency) = a SYSTEMS problem → approximate search,
  caching, batching.
- *Agentic* = a REASONING-QUALITY problem → and it ADDS latency/cost/nondeterminism
  (every agent step is another LLM call). "Real-time agentic" is near-contradiction.
  Decide which axis the query workload actually needs before building; don't bolt
  on agents because they're fashionable.

**What breaks at ~70k children (why the current code can't just scale):**
1. Dense is brute-force `matrix @ qvec` — O(N) scan, no index. ~107 MB in RAM,
   survivable but it's the ceiling; past ~100k–1M needs a real ANN index.
2. BM25 is recomputed per query in python dicts — painfully slow at 70k; needs a
   real inverted index (bm25s / Tantivy / OpenSearch).
3. Everything is JSON — a 70k-child chunks.json (raw_text+blurb+text) is hundreds
   of MB to parse per load; move to parquet/sqlite or the vector DB itself.
   → The stage-4 code validates the IDEA cheaply; it is NOT the scaled system.
   Don't fight to make numpy scale to 70k — prove recall small, rebuild on infra.

**Foundation (do first, unavoidable):**
- Vector store: Qdrant or pgvector (self-host / learning), or Weaviate/Milvus;
  managed = Pinecone. For native hybrid (BM25+dense) AND server-side ranking at
  scale, look hard at Vespa or OpenSearch — it's the current hybrid design, built
  to scale.
- ANN tradeoff: HNSW (fast, memory-hungry) vs IVF-PQ (compressed, cheaper RAM,
  slightly lower recall) — pick against a memory budget.
- Metadata filtering (source/section/date) — trivial at 7 docs, essential at 1000.

**Ingestion pipeline is what changes most** — from "process 7 PDFs by hand" to
"a corpus that grows." Pipeline becomes the product: parse → chunk → (blurb) →
embed → upsert, and it MUST be incremental, idempotent, resumable, and handle
updates/deletes/dedup (the resumable-cache instinct from stage 4 generalizes).
Don't reach for Airflow/Prefect/Dagster until doc churn justifies it — a plain job
runner first.

**Agentic RAG — where it earns cost, where it's a trap:**
- Patterns: query rewriting/decomposition (compound → sub-queries); CRAG /
  Self-RAG (retrieve → LLM-GRADE results → re-retrieve or fall back) = highest-ROI
  quality pattern, and we already have the ingredients (refusal-signal idea +
  hybrid; Self-RAG is one of our own source papers); multi-hop/iterative retrieval;
  router agent (pick index/source/tool — matters with heterogeneous sources).
- TRAP: making EVERY query agentic. Each pattern = 1–N extra LLM calls + latency +
  nondeterminism. Discipline: keep single-shot hybrid+rerank as the default FAST
  path; route only the queries that need it (a cheap "is this multi-hop?" gate)
  into the agentic path. A gate in front beats an agent loop around everything.

**Reranking comes back at scale.** Dropped ms-marco correctly at stage 3; at 70k
docs top-k precision matters more (bigger haystack), so a real reranker earns its
place — paraphrase-tolerant (BGE-reranker-v2 / mxbai) or late-interaction
(ColBERTv2), NEVER ms-marco. Retrieve wide → rerank → serve narrow (same shape).

**Latency (when you get there):** approximate search (HNSW); a SEMANTIC CACHE for
repeated/similar queries (this is the query-time caching deferred at stage 4 — it
becomes real at serving scale); async + batched embedding; a latency SLA (e.g. p95
retrieval < 300 ms) designed backward from.

**Recommended Stage-5 sequence (don't change ten things at once):**
1. Swap infra: vector DB + native hybrid + ANN (makes 1000 docs possible).
2. Build incremental ingestion (updates/deletes).
3. Add reranking for top-k precision.
4. Grow + re-label the golden set — synthetic QA generation to scale labeling
   (this is the biggest HUMAN cost, not a code cost).
5. THEN layer agentic behaviour — start with CRAG-style retrieval grading, measured
   against the non-agentic baseline.
6. Optimise latency/caching LAST, against an SLA.

**Carry-over rule:** every one of these is a knob, measured against a versioned
baseline. Scale doesn't excuse dropping the one-knob eval discipline — at 70k
chunks you can't eyeball what broke, so it matters more.

---

## Stage 5 — EXECUTION: framework adopted (LangChain) + pgvector + mixed corpus

The roadmap above was the plan; this is what actually shipped. Two big pivots:
drop the no-framework rule, and move off numpy/JSON onto real infra.

### Framework decision — LangChain / LangGraph
The "no frameworks" rule was a *learning* rule for stages 1–4. It did its job —
we now understand chunking, BM25, RRF, reranking, blurbs from the metal up — so
re-hand-rolling them at scale is just toil. Chose LangChain (most-used, biggest
ecosystem, best agentic story for the planned CRAG/routing) over LlamaIndex
(more RAG-native, would cut the most code) and Haystack (more explicit, closer
to our ethos). **Hard line: the eval harness stays hand-written.** It is the
judge — framework default drift silently breaks baseline comparability, which
kills the one-knob discipline. Same for the refusal/negative-gap metric.

### Corpus pulled — ~740 docs, ~3.5 GB (was 7 papers)
Resumable multi-source downloader (`ingest/corpus_puller.py`) run on the Mac
(the sandbox proxy blocks these hosts). Landed: arxiv 296 (pdf), sec 200 (htm,
NOT pdf — modern filings are HTML), wikipedia 151 (pdf), archive 61 (SCANNED
pdf, ~21 MB each), rtd 32 (pdf), plus rag_papers (the original 7). datagov came
back empty (dead links, as predicted). SEC `srqsb` endpoint is dead — used the
`data.sec.gov` submissions API; GitHub-docs→PDF replaced with ReadTheDocs PDF
builds (no site-building).

### Reproduce-first — the discipline paid off
Rebuilt the stage-4 hybrid on LangChain over the SAME corpus + labels
(`v2-55q-2026-08-02`), framework as the only knob. `EnsembleRetriever(c=60)` ==
our RRF `1/(60+rank)`.

| run | R@5 | note |
|---|---|---|
| hand-rolled stage-4 hybrid | 0.833 | the target |
| LangChain, first run | **0.771** | regression |
| LangChain, after tokenizer fix | **0.854** | within noise of 0.833 |

Root cause of the 0.771: LangChain's `BM25Retriever` default `preprocess_func`
is `text.split()` — no lowercasing, keeps punctuation — so `"BART."` ≠ `"bart"`
and the corpus's dominant lexical lever silently degraded. Fix: pass
`preprocess_func = lambda t: re.findall(r"[a-z0-9]+", t.lower())` to match
`hybrid.py`'s tokenizer. 0.854 vs 0.833 is +0.021 on 48 positives ≈ ~1 question
= **faithful reproduction, NOT an improvement.** Framework validated; safe to
scale. **New baseline = 0.854** — re-baseline future changes against THIS, not
0.833 (LangChain fuses child rankings then collapses to parents; hand-rolled
collapsed to parents then fused — a real, small difference). Decided NOT to
reimplement parent-level RRF for exact parity; the simplicity wins.

**Lesson (the concrete "frameworks hide knobs" instance): a framework default
cost 0.06 recall with no error.** When adopting any framework component, diff
its defaults against the hand-tuned version and re-baseline before trusting it.

LangChain v1 API reality (installed 1.3.14): `langchain.retrievers` is GONE;
`langchain-community` is being SUNSET; `EnsembleRetriever`/`ParentDocumentRetriever`
now in `langchain_classic`; `BM25Retriever` still in community (the one remaining
community dep — migrate off it at scale). `1.3.14` (langchain) and `1.5.3`
(langchain-core) are DIFFERENT packages, not two versions of one — core is the
interfaces-only foundation you can't build on alone.

### Vector DB — pgvector; parser-routed ingestion
Chose pgvector (Postgres) for the dense store. Built `rag_stage_5/ingest.py`:
FILE-TYPE routing (this is the "routing" we meant — NOT agentic query routing)
into lanes matched to the corpus —
- digital PDF (arxiv/wikipedia/rtd/rag_papers) → PyMuPDFLoader (fast)
- HTML (sec .htm) → BSHTMLLoader
- scanned PDF (archive) → DoclingLoader + OCR (slow, hours)
- markdown → TextLoader

It parent/child-chunks and writes the project's canonical `sections.json` +
`chunks.json` (so run_eval/qa_gen/lc_pipeline keep working) AND upserts child
embeddings into a pgvector collection. Resumable (per-doc checkpoint, skip
already-parsed). `lc_pipeline.py` gained a `--vector-store pgvector` dense path
alongside the in-memory reproduce path.

Architecture clarified while building: **embeddings come from the DENSE model
only**; BM25 is a vectorless keyword ranker (term stats, no vectors); "dense +
BM25" is hybrid *retrieval*, not embedding. The two never meet in vector space —
they each rank the same chunk IDs and RRF fuses the two *rankings* by ID (ranks,
not scores, because cosine and BM25 scales are incomparable).

### Cons carried in (eyes open)
Framework churn/lock-in + hidden knobs + lost mechanics visibility; baseline
discontinuity (0.854 ≠ 0.833 measurement); char-window chunking in ingest.py is
a regression from the section-aware extractor (re-tune likely needed); parser
quality is a NEW silent-failure surface (bad OCR/tables read as retrieval
misses); infra/ops burden (Docker+Postgres+torch+docling) departs from
numpy+stdlib; reproducibility harder; **hybrid only half-migrated — BM25 still
in-process `rank_bm25`, the per-query-Python bottleneck flagged for scale.**

---

## Stage 5 — PLANNED / open threads (in order)

1. **Golden set for the new corpus — THE blocker.** New chunks void every old
   label; nothing on the 740-doc corpus is measurable until we generate QA
   (`qa_gen.py`) and HAND-VERIFY a sample. `--eval` is meaningless until then;
   only `--ask` works. Synthetic labels are a draft — spot-check before trusting.
2. **Verify parser quality per source** (esp. scanned + SEC tables) before
   trusting any recall number — check the ingest WARN lines for empty extracts.
3. **Move BM25 into Postgres FTS** so both sides live in pgvector — kills the
   in-process rank_bm25 bottleneck; makes the hybrid fully scaled.
4. **Re-tune chunking** — replace blind char windows with structure-aware
   splitting (Docling structure), re-baseline against 0.854.
5. **Reranking returns at scale** — bigger haystack, top-k precision matters
   again. BGE-reranker-v2 / ColBERTv2, NEVER ms-marco. Retrieve wide → rerank →
   serve narrow.
6. **Agentic LAST** — CRAG-style retrieve→GRADE→re-retrieve/refuse first (also
   finally gives the refusal signal open since stage 3), THEN query routing.
   Keep single-shot hybrid as the default fast path; gate only the queries that
   need the agentic path. OSS-model gotcha: small local models are unreliable at
   tool-calling — use Qwen2.5/Llama-3.x tier, enforce structured output
   (Ollama format=json / grammars), move to vLLM for serving. DSPy optional to
   harden weak models.
7. **Latency/semantic-cache LAST**, against an SLA.

Carry-over rule unchanged: every one of these is a knob, measured against a
versioned baseline. At 740 docs you can't eyeball what broke.

---

## Stage 5 — retrieval method shortlist (beyond dense+BM25 hybrid)

Already shipped: contextual blurbs (index-time), hybrid BM25+dense (RRF),
cross-encoder rerank (dropped at stage 3, due back at scale per item 5 above).
Ranked by leverage for THIS corpus if/when the current hybrid plateaus:

1. **Self-query retriever** — LLM parses the NL query into a structured
   metadata filter (`source_type=sec`) + semantic remainder, executed as a SQL
   `WHERE` against pgvector's jsonb metadata. This IS the routing answer —
   replaces a hand-built router with a stock LangChain component. Pairs
   directly with item 6 above (query routing); do this instead of hand-rolling
   a router.
2. **SPLADE (learned sparse)** — neural sparse retrieval, learns term
   expansion (query "car" also weights "automobile") while staying
   interpretable/fast like BM25. Candidate BM25 replacement if paraphrase
   recall stays the weak bucket at scale.
3. **Multi-query retriever / HyDE** — multi-query: LLM generates 3-5
   reformulations, union-retrieves, dedupes. HyDE: embed an LLM-written
   hypothetical answer instead of the raw query. Cheap fan-out in front of the
   existing `EnsembleRetriever`; target paraphrase bucket specifically.
4. **RAPTOR (hierarchical summarization tree)** — recursively cluster +
   summarize chunks, retrieve at multiple abstraction levels. High value for
   long docs (arxiv/SEC filings) but expensive to build/index — don't reach
   for this until flat retrieval demonstrably fails on long-doc questions.
5. **ColBERT / late-interaction** — token-level embeddings, near
   cross-encoder quality at bi-encoder speed. Real infra lift (separate index
   format) — only if reranker latency becomes the bottleneck at scale.

Diagnostic-first, same discipline as every stage above: don't build all five
speculatively — rerun eval split by query type on the real golden set once it
exists (blocker #1), find the weak bucket, pick the method that targets it.

---

## Stage 5 — EXECUTION LOG (2026-08-10): embedder + chunking + labeling toolchain

Got the ingestion + labeling pipeline ready to produce a MEASURABLE baseline. Nothing measured yet at this point — blocker #1 (golden set) still the gate.

**Embedder: all-MiniLM-L6-v2 → BGE-M3.** Structural reasons, not a bakeoff: 8192-token ctx (all-MiniLM capped at 256, was silently truncating long arxiv/SEC chunks), multi-function (dense + learned-sparse + ColBERT in ONE model → shortlist #2/#5 come free later), domain-robust. all-MiniLM was the weakest link (dense-only R@5 0.646, BM25 beat it). RAM not the constraint (70k×1024×4B ≈ 290 MB); real costs = ColBERT per-token storage + M3 CPU-bound embed time. `normalize_embeddings=True`. pgvector collection auto-tagged `rag_stage5__bge_m3` (1024-dim can't collide with old 384-dim) → first embed needs `--reset`. **Discipline: BUILD all modes, ACTIVATE one knob at a time** (M3-dense vs MiniLM → M3-sparse vs BM25 replace-not-stack → ColBERT as rerank).

**Chunking: section-aware is PER-LANE, decided by a structure probe.** Probed real docs before writing a splitter. SEC .htm (WFC 10-K): zero heading tags (inline XBRL) but `ITEM 1A.` regex rock-solid (22 headings) — item number + title land on SEPARATE lines so the regex allows `\s+`. arxiv PDF: no stable heading signal across LaTeX templates, font/bold detection inconsistent (42/0/2/5 across 4 papers) → hand-rolling REJECTED. Shipped `ITEM_RE` + `section_units(text, lane)`: sec/html   splits on ITEM headings, every other lane = single blob → char-window (unchanged). Giant ITEM (46k) → many parents all keeping the ITEM title. Corpus gotcha: some SEC 10-Ks are SHELL filings (real Risk Factors incorporated by reference to an exhibit the puller didn't grab) → thin content, a corpus_puller gap not a splitter bug.

**Labeling toolchain: qa_gen fixes + a `--validate` GATE.** Fixed the crash (`qwen3.5` → `qwen2.5:7b`, also in lc_pipeline `--llm-model`), added `"format":"json"`, spread paraphrase across all factuals. New `--validate` mode = quality gate (not generator): (1) groundedness — is `_answer` in the labeled parent?; (2) multi-label — retrieve the query, if a non-gold parent outranks gold suggest adding it (needs retriever); (3) negative — content overlap vs any parent. Writes `<out>.validated.json`, FLAGS for human review, does not silently rewrite. Only catches labels the retriever DISAGREES with — a net, not a wall.

---

## Stage 5 — FIRST EVAL (2026-08-15): SMOKE number 0.704, gate initially skipped

Ran the first eval on the 740-corpus. **Number is PROVISIONAL** — golden set `stage5_v2` is a synthetic DRAFT.

Setup: 497,563 children in pgvector (bge-m3, 1024-dim). qa_gen `stage5_v2` = 178 Q (120 factual / 39 paraphrase / 19 negative) from 78,676 candidate parents. LangChain hybrid (EnsembleRetriever c=60).

| group | R@1 | R@5 | R@20 | R@50 | MRR |
|---|---|---|---|---|---|
| ALL (159) | 0.453 | **0.704** | 0.868 | **0.912** | 0.565 |
| factual (120) | 0.492 | 0.717 | 0.875 | 0.917 | 0.593 |
| paraphrase (39) | 0.333 | 0.667 | 0.846 | 0.897 | 0.477 |

Caveats: (1) NOT comparable to 0.833/0.854 — new corpus + labels + embed model = different ruler; the log header "compare vs 0.833" is a trap. (2) R@50 0.912 is a FETCH-LIMITED ceiling — k capped at 200/retriever (perf fix), old 0.99 was k=full. (3) First validate ran BLIND (retriever down) → flagged only 9, multi-label check never ran. (4) No hand-verify — read straight off the raw draft.

Parser damage (blocker #2 confirmed): `parse_failures.json` = 39 docs, ALL archive/scanned lane, docling load-fail. Scanned lane mostly DEAD, unmeasured.

Perf lesson (learning_stage5.md): `k=n` (full 497k) on both retrievers = 70+ min. `k` is a RETURN-size knob, not a SCAN-size knob — both BM25 and exact pgvector are O(corpus)/query regardless; the tax was materializing 497k Document objects/query. Fix `k=max(200, max(CEILING_KS))`. True fix at scale = ANN index (not yet configured).

---

## Stage 5 — GOLDEN-SET TRIAGE (2026-08-16)

Re-ran validate WITH retriever up (`retriever up (pgvector), 497563 children`): **68/178 flagged** (61 multi-label, 7 negatives, 2 ungrounded) — the real pass (first was blind, flagged 9).

Triage buckets (source-match + question-shape):

| bucket | n | action |
|---|---|---|
| ADD_SIBLING | 36 | gold's own same-doc sibling outranked it → add sibling to label (verify answer in it) |
| DROP_GENERIC | 5 | SEC boilerplate, not uniquely answerable |
| GRAY | 20 | hand-read — 16 SEC, 2 arxiv, 1 rtd, 1 wiki |
| DROP_NEG | 7 | negatives the retriever thinks are answerable |
| DROP_UNGROUNDED | 2 | answer not even in labeled parent |
| KEEP_MISS | 0 | — |

Findings:
- Flags are NOT all bad labels. 36 = genuine **multi-label undercount** — answer legitimately lives in a same-doc sibling parent that outranked gold, caused by char-window + overlap near-dup parents. So **0.704 is UNDERSTATED**; real answers weren't credited.
- Bad ones = **SEC generic questions** ("the Company's revenue increase") answerable by any 10-K. Root cause = qa_gen prompt not entity-anchored, **NOT model size** — labels generated with qwen2.5:14b, so a bigger model won't fix it. Prompt is the lever.
- `KEEP_MISS = 0`: almost every specific-but-missed question (e.g. `django save()` whose rank-1 was `xgboost`) still had a same-doc sibling retrieved → lands in ADD_SIBLING. True retrieval healthier than 0.704 suggests.

Delivered `RAG/rag_stage_5/triage_apply.py` (run locally, needs sections.json — 183MB timed out staging to sandbox). Auto-applies ONLY groundedness-verified sibling adds (sibling text must contain the answer, ≥0.5 overlap) + drops the 2 ungrounded; routes DROP_GENERIC/GRAY/negatives to `triage_todo.md` for hand-verify.

**Next (in order):**
1. `python triage_apply.py --qa qa.validated.json --sections sections.json --out qa_stage5_v3.json --version stage5_v3`
2. Hand-resolve `triage_todo.md` (20 gray + 7 neg): drop generic, keep specific-miss.
3. Re-run eval on `qa_stage5_v3.json` → **real Stage-5 baseline** (expect R@1/R@5 up once verified siblings credited).
4. If many gray SEC generics → hardened entity-anchored qa_gen prompt + regenerate SEC factuals so dropping them doesn't gut SEC coverage.
5. THEN one knob vs the weak bucket (paraphrase): blurb re-embed vs query-translation. Char-window chunking is the standing regression behind both the multi-label noise and generic ambiguity — structure-aware (Docling) for digital-PDF lanes is the deferred fix.

---

## Stage 5 — GOLDEN-SET HAND-RESOLVE COMPLETE (2026-08-17): qa_stage5_v3 finalized

Finished the 43-item hand pass on top of the auto-triage. `qa_stage5_v3.json` is now the cleaned set: **176 → 160 Q** (102 factual / 39 paraphrase / 19 negative). Backup: `qa_stage5_v3.autotriage.bak.json`.

**Dropped 16** (generic / not gold-unique — the entity-anchoring root cause, NOT bad retrieval): 15 SEC + 1 archive. SEC drops were template questions answerable by many 10-Ks ("the Company's revenue % 2024→2025", ICFR "maintain records" SOX boilerplate, "failure to meet analysts' expectations", social-media-channel line, GDPR €20M penalty, CMS HCC v28 shared across MA insurers, generic M&A/compliance risk factors). Archive drop = OCR-garbled scanned doc, answer ungroundable.

**Kept 18 specific misses** (entity-named or doc-unique, genuine retriever miss, gold correct): CSX $61B, Citigroup 170,667, SLB 1,894, AROs 2,982, MPS equity, Apollo $108, MetLife 1,412, $14M pension, HPE divestiture, django CVE index, twisted "Anonymous", requests prepare_cookies, 2 arxiv, 2 wikipedia, etc.

**Fixed 1 gold** (mislabeled parent): `arxiv-2608-06358v1-94-f` #94 (bibliography) → **#27** (constants/Lemma 3.4 section where p(n) is defined). Verified 2/2 answer terms.

**Negatives: kept all 7.** The validator's "possibly ANSWERABLE" flags were FALSE POSITIVES — topical similarity, not answerability (exact stock prices / Beijing weather / vague EU regs don't exist in the corpus). Refusal-test set is solid.

**Lesson — verify against FULL parent text, not a clip.** 2 of 3 suspected-mislabels (Moody's 39% FX, sec-077476 commodity list) were FALSE alarms from a 600-char triage-dump truncation; the answer sat later in the long parent. A same-doc term-search (not eyeballing a snippet) is the correct check. The apply script term-searched and correctly left both on their original gold.

**RISK flagged:** `sections.json` (175M) + `chunks.json` (262M) are gitignored (`*.json`) — they were briefly lost this session then recovered by the user. They are NOT backed up and NOT rebuildable from git. Rebuild path if lost again: reconstruct from pgvector (`rag_stage5__bge_m3`, 497,563 children carry `{parent_id, child_id, source}` + child text) — chunks.json exact, sections.json parent text stitched from children; OR `python ingest.py --parse` (re-OCRs scanned lane, hours).

**Next (in order):**
1. Interim baseline — run eval on `qa_stage5_v3.json`:
   `python lc_pipeline.py --eval --qa qa_stage5_v3.json --vector-store pgvector --collection rag_stage5__bge_m3 --k 1 5`
   (expect R@1/R@5 UP vs 0.704: 16 ambiguous questions gone + 23 auto-siblings + 1 gold fix credited.)
2. Refill dropped SEC coverage — entity-anchor the qa_gen prompt (force "{company} … {period}"), regenerate SEC factuals, re-validate. THIS is the real Stage-5 baseline.
3. Then one knob vs the weak bucket (paraphrase 0.667): blurb re-embed vs query-translation; structure-aware (Docling) chunking is the standing root fix.

---

## Stage 5 — REAL BASELINE (2026-08-17): qa_stage5_v3, cleanup + weight knob

First trustworthy Stage-5 numbers (bge-m3 1024-dim, pgvector `rag_stage5__bge_m3`, 160-Q cleaned set). Ran as TWO isolated knobs.

| group | R@5 v2 (pre-clean) | R@5 v3 @50/50 | R@5 v3 @70/30 |
|---|---|---|---|
| ALL_POSITIVE | 0.704 | 0.803 | **0.830** |
| factual | 0.717 | 0.855 | **0.882** |
| paraphrase | 0.667 | 0.667 | **0.692** |

**Two effects, separated (one-knob discipline):**
1. **Cleanup (v2→v3 @50/50): +0.099 ALL** (0.704→0.803). The dominant move — dropping 16 ambiguous Qs + crediting 23 auto-siblings + 1 gold fix. Confirms 0.704 was understated by bad labels, not bad retrieval.
2. **Weights (50/50→70/30 dense-heavy): +0.027 ALL, uniform across buckets.** ~4 questions flipped. Small but directionally consistent (both factual AND paraphrase up under more semantic weight) → a modest REAL gain, not noise. Paraphrase +0.025 alone ≈ 1 Q = would be noise; the cross-bucket consistency is what makes it credible.

**CURRENT BASELINE = v3 @ 70/30 → ALL R@5 0.830.** Run flag: `--weights 0.7 0.3` (first number = DENSE; main() flips the pair). Default in code still 50/50 — 70/30 is passed at runtime, NOT hardcoded.

**Caveats (do not misread the number):**
- NOT comparable to old 0.833/0.854 — different corpus + labels + embed model. factual 0.882 crossing 0.854 is a DIFFERENT ruler, not a record.
- **Weight 70/30 is PROVISIONAL.** SEC refill (regenerate the 14 dropped SEC factuals, entity-anchored) will add lexical/number Qs that may pull optimal weight back toward BM25. Re-confirm the weight after any set change.
- Paraphrase 0.692 is still the weak bucket — weights nudged it; the real lever (blurb re-embed / query-translation / structure-aware chunking) is untouched.

**Decision point:** SEC refill (eval-fidelity, not a system gain — 118 factual Qs remain so SEC isn't gutted) vs move to agentic (CRAG retrieve→grade, also unlocks the refusal metric via the 19 negatives) measured against THIS 0.830 baseline.
