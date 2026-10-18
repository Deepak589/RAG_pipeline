# GraphRAG — idea note

**Status:** exploring, not scheduled. Not a stage.
**Written:** 2026-10-01, after Stage 6 (CRAG, 69.5% e2e / 0.830 R@5).
**Question that started it:** can we build GraphRAG on top of this?

Two separate answers, because there are two separate goals.

---

## 0. Verdict up front

| goal | answer |
|---|---|
| ship it on the main line, as Stage 7 | **No.** Wrong next knob — see §1. |
| learn how GraphRAG actually works | **Yes** — but on a 100-doc sandbox, not the 497k corpus. See §3. |
| cheap graph signal on the real corpus | **Maybe** — free metadata edges, no LLM. See §2. Independent of the other two. |

These do not have to happen in that order, but they must not be confused for each other.
The learning track is a sandbox and never touches `rag_stage_5/` or `rag_stage_6/`.

---

## 1. Why not on the main line (yet)

Four objections, in descending strength.

### 1.1 The eval can't see the gain — fatal on its own

`qa_stage5_v3` is 160 Q: 102 factual / 39 paraphrase / 19 negative.
Essentially **zero multi-hop or global/summarization questions**.

GraphRAG's whole claim is on exactly the question shapes that golden set does not contain.
Build it and the number doesn't move — which tells us nothing about whether it worked.
That breaks the rule the repo exists to enforce: no change ships without a number behind it.

**Precondition for any graph work on the main line: a question slice that can see it.** See §5.

### 1.2 Cost is not incremental

Entity + relation extraction over 497,563 children is on the order of **400M input tokens**
before any community summarization passes. (Rough: ~800 tok/chunk in, ~200 out. Verify before
committing to anything.) That is a real bill for an unmeasured gain, and a far bigger
one-shot bet than any knob moved so far.

### 1.3 The corpus fights it

The corpus is arxiv + SEC + Wikipedia + scanned archive PDFs + ReadTheDocs.

- Entity collisions across lanes: "Transformer" in a paper vs. in a filing are different nodes
  that naive extraction will merge.
- Genuine cross-lane edges are rare — the lanes are not about the same things.
- A graph whose edges are mostly within-lane and mostly weak is an expensive index, not a graph.

GraphRAG assumes a reasonably coherent domain. This corpus is deliberately heterogeneous
(that was the point of Stage 5).

### 1.4 It conflicts with the production direction

`solution.md` commits to incremental ingest: content-hash + embedder_version upserts, new docs
flow in continuously. GraphRAG community summaries are **global state** — one new document can
invalidate a community, and strictly you re-cluster and re-summarize. Designing the graph now
means designing against the next milestone.

Not unsolvable (incremental community update is a real research area), but it is a second hard
problem stacked on an unsolved first one.

### 1.5 What to do instead, on the main line

Current attribution on 160 Q: `retrieval_miss` 23, `refused_on_hit` 14, `answered_wrong` 6.
Retrieval is the top bucket again — and **why** those 23 miss is still undiagnosed.

If the cause is lane confusion (and production traffic is expected to jump between lanes),
**self-query metadata routing** — roadmap #2, `source_type=sec` etc. via pgvector's jsonb filter —
targets it directly for a fraction of the cost. Do that, re-read the attribution table, then
revisit this note.

---

## 2. The free-metadata graph (worth considering on its own)

You do not need LLM extraction to get a graph. These edges already exist in the corpus and cost
approximately nothing:

| edge | source | extraction |
|---|---|---|
| parent → child | current chunking | already stored |
| paper → cited paper | arxiv references section | regex / GROBID, no LLM |
| filing → company / ticker / period | SEC CIK + filing metadata | parse, no LLM |
| page → linked page | ReadTheDocs internal hyperlinks | parse the HTML already loaded |
| article → linked article | Wikipedia wikilinks | parse, no LLM |

**Storage:** an edge table in Postgres. `(src_parent_id, dst_parent_id, edge_type, weight)`.
**Do not add Neo4j.** A recursive CTE handles 1–2 hop expansion at this scale; a second datastore
buys a second consistency problem and breaks the incremental-ingest design in `solution.md`.

**Retrieval shape:** hybrid top-K → 1-hop expansion over the edge table → re-score → CRAG as today.
Drops in as a retriever variant behind the existing `query -> ranked [(parent_id, score)]` contract,
so `run_eval.py` and `crag_eval.py` work unchanged.

This is testable in days, not weeks, and it is a genuine one-knob change. It still needs the
question slice from §5 to be worth measuring.

---

## 3. Learning track — the sandbox

Goal here is understanding the mechanics, not moving a production number. Optimise for being able
to **look at the graph and tell whether it is right**.

### 3.1 Setup

- **Slice:** ~100 arxiv papers from *one* subfield. Coherent entity space, so extraction errors are
  visible by eye. A cross-lane slice teaches nothing except that it is broken.
- **Graph:** in-memory `networkx`. No pgvector, no Neo4j, nothing from Stage 5/6 infra.
- **Location:** `RAG/graph_lab/` — a sandbox, **not** a stage folder. Keep the staged line clean.
- **Models:** Haiku 4.5 for extraction (volume), Sonnet 5 for community summarization and judging.
  `temperature=0`. Remember the -5 family rejects the `temperature` param — reuse
  `make_chat_anthropic()` from `crag_pipeline`.

### 3.2 The six mechanics, in build order

1. **Entity + relation extraction**
   LLM with structured output → `(source, relation, target, description)` triples per chunk.
   Prompt carries an entity-type list; this is itself a tunable knob.

2. **Entity resolution**
   Collapse `BERT` / `bert` / `BERT-base` / `Devlin et al.` into one node.
   The part tutorials skip and the part that actually decides quality. Start with
   normalize + exact match, then embedding similarity + threshold, and *look at what merges*.

3. **Graph construction**
   Nodes carry an aggregated description (LLM-summarized from all mentions). Edges weighted by
   co-occurrence count. Keep a node → source-chunk-id index — that is what retrieval returns.

4. **Community detection**
   Leiden (`leidenalg`), or `networkx.greedy_modularity_communities` to start.
   Hierarchical: communities of communities. Levels matter for global search.

5. **Community summarization**
   LLM-summarize each community bottom-up: leaf communities from their nodes/edges, parent
   communities from their children's summaries. This is what makes global search possible at all.

6. **Two query modes**
   - **Local search** — anchor on entities mentioned in the query, walk the neighborhood, pull
     their source chunks, answer. Practical, cheap, comparable to current retrieval.
   - **Global search** — map-reduce over *every* community summary at a chosen level, then reduce
     to a final answer. Answers "what are the main themes", costs one LLM call per community.

### 3.3 Method: hand-roll first, then diff against the library

Same approach as stages 1–4 (framework-free numpy/stdlib before LangChain). Build all six steps by
hand, then compare against:

- `microsoft/graphrag` — the reference implementation; read its prompts and its gleanings loop
- LangChain `LLMGraphTransformer` — the convenience path

Expect to find the same class of silent default that the `BM25Retriever` tokenizer turned out to be
in Stage 5. Diffing framework defaults against a hand-tuned version before trusting them is already
a hard rule here.

---

## 4. Traps worth hitting deliberately

- **Extraction is non-deterministic.** Run it twice on the same doc, diff the triples. MS GraphRAG's
  "gleanings" (repeat extraction passes until no new entities) exists because of this. Measure the
  variance before tuning anything downstream of it.
- **Bad entity resolution silently poisons everything.** One concept split across five nodes means
  every edge weight is wrong. The graph will *look* fine. Inspect the merge decisions directly.
- **Global search cost scales with community count, per query.** Measure the latency once and it
  becomes obvious why nobody serves it directly on the hot path.
- **Single-fact lookup gets worse than the hybrid baseline.** Slower and less accurate. Seeing that
  failure clearly is the most useful thing the sandbox produces.
- **Community summaries are lossy by construction.** A fact present in a chunk can be absent from
  every summary above it. Global search cannot find it; local search can.

---

## 5. Eval design — the actual learning artifact

25 questions on the sandbox slice:

- 10 **single-fact** — one chunk contains the answer
- 10 **multi-hop** — requires joining two or more documents
- 5 **global** — "what are the main themes across these papers", no single source document

Run all 25 against three methods: **baseline hybrid** / **local search** / **global search**.

The 3×3 result table is the output of this whole exercise. It says exactly which question shape
each method owns, and it is reusable as the precondition in §1.1 — if a multi-hop slice is ever
added to `qa_stage5_v3`, these questions are the template.

Expected shape, to be falsified:

| | single-fact | multi-hop | global |
|---|---|---|---|
| hybrid (current) | best | weak | fails |
| local search | worse, slower | best | weak |
| global search | fails | weak | best |

If the real table does not look like this, that is the finding.

---

## 6. Exit criteria — what would promote this off the idea list

Graph work earns a stage folder only when **all** of these hold:

1. A multi-hop / relational question slice exists in the golden set and the current pipeline
   measurably fails it (not 75%+ already).
2. `retrieval_miss` has been diagnosed and metadata routing (roadmap #2) has been tried first.
3. The free-metadata edges of §2 have been measured, because they are the cheap version of the
   same bet.
4. There is an answer for how communities update under incremental ingest, or a decision to
   accept periodic full re-clustering with a stated cost.

Until then this stays an idea note.

---

## 7. Open questions

- What fraction of the 23 `retrieval_miss` cases are actually relational vs. lane-routing vs.
  chunking artifacts? Nothing here is justified until that is answered.
- Does the free-metadata graph (§2) capture most of the achievable gain at ~1% of the cost of
  LLM extraction? Plausible, untested.
- Is there a hybrid: LLM extraction on **one** lane only (arxiv, where entity space is coherent),
  metadata edges everywhere else?
- Does local search beat a well-tuned reranker on multi-hop, or is it solving the same problem
  more expensively? Reranking is already slated to return at scale (roadmap #4).

## 8. References to read before building

- Microsoft Research, "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
- `microsoft/graphrag` source — especially the extraction prompts and the gleanings loop
- Traag et al., "From Louvain to Leiden: guaranteeing well-connected communities"
- LangChain `LLMGraphTransformer` docs — the convenience path, for the diff
