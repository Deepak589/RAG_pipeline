# Retrieval-Augmented Generation (RAG): A Complete Guide

A learning document covering RAG from first principles to the research frontier.
Each section cites the papers that introduced the ideas, and the final section maps
everything back to your own implementation in `Naive_rag.py`.

---

## Table of Contents

1. [Why RAG Exists](#1-why-rag-exists)
2. [Origins: The Research That Led to RAG](#2-origins-the-research-that-led-to-rag)
3. [Anatomy of a RAG Pipeline](#3-anatomy-of-a-rag-pipeline)
4. [Chunking: Splitting Documents](#4-chunking-splitting-documents)
5. [Representing Text: Sparse vs. Dense Retrieval](#5-representing-text-sparse-vs-dense-retrieval)
6. [Vector Search at Scale](#6-vector-search-at-scale)
7. [Reranking](#7-reranking)
8. [Generation: Using Retrieved Context](#8-generation-using-retrieved-context)
9. [Advanced RAG Techniques](#9-advanced-rag-techniques)
10. [Evaluating RAG Systems](#10-evaluating-rag-systems)
11. [Known Failure Modes](#11-known-failure-modes)
12. [RAG vs. Long Context vs. Fine-Tuning](#12-rag-vs-long-context-vs-fine-tuning)
13. [Your Naive_rag.py, Mapped to This Guide](#13-your-naive_ragpy-mapped-to-this-guide)
14. [Reading List](#14-reading-list)

---

## 1. Why RAG Exists

Large language models have three structural problems that no amount of scaling fixes:

1. **Frozen knowledge.** A model's knowledge stops at its training cutoff. It cannot
   know yesterday's news, your company's internal wiki, or a document you wrote this
   morning.
2. **Hallucination.** When a model doesn't know something, it often produces a fluent,
   confident, wrong answer. Parametric memory (knowledge stored in the model's weights)
   has no mechanism to say "I never saw this."
3. **No attribution.** You cannot ask a model's weights *where* a fact came from. There
   is no citation, no source, no way to audit.

RAG's core idea: **separate knowledge from reasoning.** Keep the model for language
understanding and generation, but store knowledge in an external corpus that is
searched at question time. The model answers *grounded in retrieved evidence* instead
of from memory alone.

The canonical formulation comes from **Lewis et al., 2020 — "Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks"** (arXiv:2005.11401), which coined the
term "RAG." Their framing: a *retriever* p(z|x) selects passages z given input x, and
a *generator* p(y|x,z) produces the answer conditioned on both. The answer is
marginalized over retrieved passages. Modern RAG systems simplify this (retrieve once,
paste into the prompt), but the two-component structure — retriever + generator — is
unchanged.

Benefits, as demonstrated in that paper and thousands since:

- **Updatable knowledge**: swap documents in the index, no retraining.
- **Reduced hallucination**: the model is instructed to answer only from evidence.
- **Provenance**: you can show the user exactly which passage supported the answer.
- **Access control & privacy**: retrieval can respect per-user permissions; the model
  never memorizes your private data.
- **Cost**: a small model + good retrieval often beats a much larger model answering
  from memory on knowledge-intensive tasks.

---

## 2. Origins: The Research That Led to RAG

RAG did not appear from nowhere. It is the merger of two research lines: open-domain
question answering (IR + reading comprehension) and neural retrieval.

**DrQA — Chen et al., 2017, "Reading Wikipedia to Answer Open-Domain Questions"**
(arXiv:1704.00051). The template for everything after: a TF-IDF retriever selects
Wikipedia articles, a neural reader extracts the answer span. Retrieval was still
classical/sparse; only reading was neural. Your `Naive_rag.py` is architecturally a
descendant of DrQA — sparse retrieval feeding a neural reader/generator.

**ORQA — Lee et al., 2019, "Latent Retrieval for Weakly Supervised Open Domain
Question Answering"** (arXiv:1906.00300). First to show the retriever itself could be
a trained neural network, learned end-to-end from QA pairs, beating BM25 on some
benchmarks. Introduced the "inverse cloze task" for pretraining retrievers.

**REALM — Guu et al., 2020, "REALM: Retrieval-Augmented Language Model
Pre-Training"** (arXiv:2002.08909). Integrated retrieval into language-model
*pretraining* itself: the model learns to retrieve documents that help it predict
masked tokens. Proved retrieval could be a first-class part of the LM, not a bolt-on.

**DPR — Karpukhin et al., 2020, "Dense Passage Retrieval for Open-Domain Question
Answering"** (arXiv:2004.04906). The workhorse paper. Two BERT encoders — one for
questions, one for passages — trained with contrastive loss so that a question and
its answer-bearing passage land close together in vector space. Showed dense
retrieval decisively beating BM25 on open-domain QA. Almost every "embedding model"
used in RAG today is a descendant of DPR's bi-encoder design.

**RAG — Lewis et al., 2020** (arXiv:2005.11401). Combined DPR retrieval with a BART
generator, trained jointly, and named the pattern. Two variants: RAG-Sequence (one
set of retrieved docs for the whole answer) and RAG-Token (can draw on different docs
per generated token).

**FiD — Izacard & Grave, 2021, "Leveraging Passage Retrieval with Generative Models
for Open Domain QA"** (arXiv:2007.01282). "Fusion-in-Decoder": encode each retrieved
passage independently, fuse them in the decoder. Scaled much better with the number
of retrieved passages and set QA state-of-the-art. Precursor to how modern systems
think about combining many pieces of evidence.

The final shift: once instruction-tuned LLMs (GPT-3.5+, Llama, etc.) arrived, joint
training became unnecessary for most applications. You could simply **paste retrieved
text into the prompt** of a frozen LLM. This "in-context RAG" (studied in **Ram et
al., 2023, "In-Context Retrieval-Augmented Language Models"**, arXiv:2302.00083) is
what nearly everyone, including your script, does today.

---

## 3. Anatomy of a RAG Pipeline

Every RAG system, from your 160-line script to enterprise deployments, has two phases:

### Indexing phase (offline, done once per corpus)

```
documents → load → chunk → embed → store in index
```

1. **Load**: read raw files (txt, PDF, HTML, ...). Real systems spend enormous effort
   here — PDF parsing, table extraction, OCR, deduplication.
2. **Chunk**: split documents into retrieval units (Section 4).
3. **Embed**: convert each chunk into a vector (Section 5).
4. **Store**: put vectors in a searchable index (Section 6).

### Query phase (online, per question)

```
question → embed → search index → (rerank) → build prompt → LLM → answer
```

1. **Embed the question** into the same vector space as the chunks.
2. **Search**: find the k most similar chunks (top-k retrieval).
3. **Rerank** (optional): re-score candidates with a stronger model (Section 7).
4. **Augment**: paste chunks into a prompt template with the question.
5. **Generate**: the LLM answers, grounded in the provided context.

The single most important insight about RAG quality: **the generator cannot fix bad
retrieval.** If the answer isn't in the retrieved chunks, the best possible outcome is
the model saying "I don't know" — and the worst is a hallucination. Retrieval quality
is the ceiling on system quality, which is why most of the research effort
(Sections 4–7, 9) targets the retrieval side.

---

## 4. Chunking: Splitting Documents

Why chunk at all?

- Embedding models have input length limits (typically 512 tokens).
- A whole-document embedding averages many topics into mush; retrieval precision
  suffers ("dilution").
- The LLM's context window is finite and attention degrades over long contexts
  (Section 11) — you want small, dense, relevant pieces.

The tension: **small chunks retrieve precisely but may lack context to answer; large
chunks carry context but retrieve imprecisely.** Every chunking strategy is a
different resolution of this trade-off.

### Strategies

| Strategy | How it works | Trade-off |
|---|---|---|
| **Fixed-size + overlap** | Split every N words/tokens, share M words between neighbors | Simple, predictable; ignores semantic boundaries — can cut ideas in half |
| **Recursive / structural** | Split on paragraphs → sentences → words, respecting document structure (headings, code blocks) | Respects natural boundaries; chunk sizes vary |
| **Semantic chunking** | Embed sentences, cut where consecutive-sentence similarity drops (topic shift) | Boundaries match topics; extra embedding cost |
| **Sentence-window** | Retrieve single sentences, but hand the LLM a window of surrounding sentences | Precise retrieval + sufficient context; more bookkeeping |
| **Parent-document** | Index small child chunks; on match, return the larger parent chunk | Same idea as sentence-window at coarser grain |
| **Late chunking** (Günther et al., 2024, arXiv:2409.04701) | Run a long-context embedding model over the whole document, then pool token embeddings into chunk vectors | Each chunk's vector "knows" full-document context (e.g., resolves pronouns) |

**Overlap** exists because a fixed-size cut can land mid-sentence or mid-idea; sharing
words between adjacent chunks means every idea appears intact in at least one chunk.

Empirical studies (and practitioner consensus) put useful chunk sizes around 128–512
tokens with ~10–20% overlap, but the honest answer from the literature is: **optimal
chunking is corpus- and query-dependent — measure it** (Section 10).

Your script: fixed-size, 200 words, 40-word (20%) overlap — the standard baseline.

---

## 5. Representing Text: Sparse vs. Dense Retrieval

The heart of retrieval: turn text into vectors such that "relevant to the same
question" ≈ "close in vector space."

### 5.1 Sparse retrieval (lexical)

Vectors have one dimension per vocabulary word; almost all entries are zero.

**TF-IDF** (Spärck Jones, 1972, "A statistical interpretation of term specificity"):

- **TF** (term frequency): how often a word occurs in this document, normalized by
  document length. Captures "this document is about this word."
- **IDF** (inverse document frequency): `log(N / df)` where df = number of documents
  containing the word. Rare words are informative; words in every document ("the",
  "is") carry no signal and get weight ≈ 0.
- Score = TF × IDF per word; document similarity = cosine between vectors.

**BM25** (Robertson & Zaragoza, 2009, "The Probabilistic Relevance Framework: BM25
and Beyond") — the industrial-strength successor and default lexical baseline in
every search engine (Lucene, Elasticsearch). Improves on raw TF-IDF with:

- **TF saturation**: the 10th occurrence of a word adds less than the 2nd
  (diminishing returns, controlled by parameter k₁ ≈ 1.2–2.0).
- **Length normalization**: long documents don't win just by containing more words
  (parameter b ≈ 0.75).

BM25 is *shockingly* hard to beat. The BEIR benchmark (Section 10) showed BM25
outperforming many dense retrievers on out-of-domain corpora.

**Strengths of sparse**: exact matching of rare terms, IDs, names, code symbols; no
training; interpretable; cheap.
**Weakness**: the **vocabulary mismatch problem** — "car" and "automobile" are
orthogonal dimensions. A query phrased differently from the document retrieves
nothing. This is the precise weakness of your TF-IDF implementation.

### 5.2 Dense retrieval (semantic)

A neural network maps text to a low-dimensional dense vector (384–4096 dims) where
*meaning*, not word identity, determines position. "Car" and "automobile" land close
together.

Key papers:

- **DPR** (arXiv:2004.04906) — the bi-encoder blueprint: separate encoders for query
  and passage, trained contrastively (pull question toward its gold passage, push
  away "hard negatives" — e.g., BM25-retrieved passages that *don't* contain the
  answer).
- **Sentence-BERT — Reimers & Gurevych, 2019** (arXiv:1908.10084) — showed how to
  turn BERT into a practical sentence-embedding model with siamese networks; spawned
  the `sentence-transformers` library, the standard tool for local embeddings (e.g.,
  the 22M-parameter `all-MiniLM-L6-v2`, 384 dims).
- **Contriever — Izacard et al., 2021** (arXiv:2112.09118) — unsupervised contrastive
  pretraining for retrieval; no labeled QA pairs needed.
- **E5** (Wang et al., 2022, arXiv:2212.03533), **GTE**, **BGE** (Xiao et al., 2023,
  arXiv:2309.07597) — modern open embedding families trained on massive weakly
  supervised pairs; dominate the MTEB embedding leaderboard (Muennighoff et al.,
  2022, arXiv:2210.07316). Note E5-style models require prefixes ("query: ...",
  "passage: ...") because question-space and document-space are asymmetric.
- **Matryoshka embeddings** (Kusupati et al., 2022, arXiv:2205.13147) — train so that
  prefixes of the vector are themselves usable embeddings; lets you truncate 1024-dim
  vectors to 256 dims for cheap search with minor quality loss.

**Strengths of dense**: handles synonyms, paraphrase, cross-lingual matching;
"what's the biggest planet" matches "Jupiter is the largest..." with zero shared
keywords.
**Weaknesses**: can miss exact rare tokens (product IDs, function names); needs a
model; out-of-domain performance degrades (BEIR finding).

### 5.3 Late interaction: ColBERT

**ColBERT — Khattab & Zaharia, 2020** (arXiv:2004.12832; ColBERTv2:
arXiv:2112.01488). Middle ground between bi-encoders (one vector per text, fast but
lossy) and cross-encoders (full attention between query and doc, accurate but slow).
ColBERT keeps **one vector per token** and scores via "MaxSim": each query token
finds its best-matching document token; sum the maxima. More accurate than
bi-encoders, far cheaper than cross-encoders; higher storage cost.

### 5.4 Learned sparse: SPLADE

**SPLADE — Formal et al., 2021** (arXiv:2107.05720). A neural model that outputs
*sparse* vectors over the vocabulary, learning to expand documents with related terms
(a document about "cars" gets weight on "automobile" too). Combines lexical
precision with learned semantics; runs on standard inverted-index infrastructure.

### 5.5 Hybrid retrieval

Production consensus: **run sparse (BM25) and dense in parallel, merge results.**
Sparse catches exact matches; dense catches paraphrase. The standard merge is
**Reciprocal Rank Fusion (RRF)** — Cormack et al., 2009: each document's score is
Σ 1/(k + rank_i) across the result lists (k ≈ 60). RRF needs no score calibration
between systems, which is why it won over weighted score sums.

---

## 6. Vector Search at Scale

With 6 chunks, you compare the query against every chunk (brute force — exactly what
`matrix @ query_vec` does). With 100 million chunks, you cannot. **Approximate
Nearest Neighbor (ANN)** search trades a little recall for orders-of-magnitude speed:

- **HNSW — Malkov & Yashunin, 2016** (arXiv:1603.09320). Hierarchical Navigable
  Small World graphs: a multi-layer proximity graph you greedily descend. The
  default ANN algorithm in almost every vector database.
- **IVF** (inverted file index): cluster vectors with k-means; search only the
  nearest clusters.
- **Product Quantization — Jégou et al., 2011**: compress vectors into compact codes
  for memory-bound corpora.
- **FAISS — Johnson et al., 2017** (arXiv:1702.08734) — Meta's library implementing
  all of the above; the reference implementation for billion-scale search.
- **ScaNN — Guo et al., 2020** (arXiv:1908.10396) — Google's anisotropic
  quantization; optimizes the quantization loss that actually matters for ranking.

Vector databases (Chroma, Qdrant, Weaviate, Milvus, pgvector, Pinecone) wrap ANN
indexes with persistence, metadata filtering ("only search docs where team=legal"),
and hybrid search. **Metadata filtering** matters more in practice than raw recall:
combining vector similarity with structured filters is where real systems win.

Rule of thumb: below ~100k chunks, brute-force exact search with numpy/FAISS-flat is
fast enough and has perfect recall. ANN complexity only pays past that.

---

## 7. Reranking

Bi-encoder retrieval compresses each text to one vector *before* seeing the query —
information is lost. A **cross-encoder** feeds query and passage *together* through a
transformer, letting every query token attend to every passage token. Far more
accurate, far too slow to run over the whole corpus.

The standard pattern — **retrieve-then-rerank**:

1. Fast retriever (BM25/dense/hybrid) fetches top 50–100 candidates.
2. Cross-encoder re-scores those candidates.
3. Top 3–10 after reranking go to the LLM.

Key work: **monoBERT/monoT5 — Nogueira et al.** (arXiv:1901.04085, arXiv:2003.06713)
established cross-encoder and seq2seq reranking; **RankGPT — Sun et al., 2023**
(arXiv:2304.09542) showed LLMs themselves can rerank via listwise prompting.
Practical tools: `sentence-transformers` CrossEncoder models (ms-marco-MiniLM),
Cohere Rerank, BGE-reranker.

Reranking is usually the single highest-leverage upgrade to a basic RAG pipeline:
it converts "answer is somewhere in top-50" into "answer is in top-3."

---

## 8. Generation: Using Retrieved Context

### Prompt construction

The near-universal template (yours included):

```
Answer the question using ONLY the context below.
If the context does not contain the answer, say so.

Context:
[source1] chunk text...
[source2] chunk text...

Question: ...
Answer:
```

Design decisions that matter:

- **Grounding instruction** ("ONLY the context") — the anti-hallucination lever. The
  model's parametric knowledge is treated as untrusted.
- **Refusal path** ("say so") — gives the model a legitimate out when retrieval
  failed. Without it, models bridge gaps with confabulation.
- **Source tags** per chunk — enables citations in the answer ("according to
  [solar_system.txt]...") and auditability.
- **Chunk ordering** — put the most relevant chunks at the beginning or end of the
  context, not the middle (see "lost in the middle," Section 11).

### Faithfulness vs. parametric knowledge

A grounded model must arbitrate conflicts: what if the context contradicts what the
model "knows"? Research on this tension: **Longpre et al., 2021, "Entity-Based
Knowledge Conflicts in QA"** (arXiv:2109.05052) showed models often ignore provided
context in favor of parametric memory — a core failure mode RAG prompting must fight.

### Attribution

Production systems increasingly require the model to emit citations per claim, then
*verify* the citation actually supports the claim (attribution evaluation: **Rashkin
et al., 2021, "Measuring Attribution in NLI"**; **Gao et al., 2023, "Enabling Large
Language Models to Generate Text with Citations"**, arXiv:2305.14627 — the ALCE
benchmark).

---

## 9. Advanced RAG Techniques

The 2023–2024 literature exploded with refinements. Organized by pipeline stage:

### 9.1 Query transformation (before retrieval)

- **HyDE — Gao et al., 2022, "Precise Zero-Shot Dense Retrieval without Relevance
  Labels"** (arXiv:2212.10496). Ask the LLM to write a *hypothetical answer* to the
  query, embed that, retrieve with it. Rationale: a fake answer looks more like a
  real document than a question does — bridging the query-document style gap.
- **Multi-query**: LLM rewrites the question 3–5 ways; retrieve with each; union the
  results. Attacks phrasing sensitivity.
- **Step-back prompting — Zheng et al., 2023** (arXiv:2310.06117): abstract the
  question ("What school did X attend in 1995?" → "What is X's education history?")
  and retrieve with the abstraction too.
- **Query decomposition**: split multi-hop questions ("Did the director of Inception
  also direct Interstellar?") into sub-questions, retrieve for each, compose.

### 9.2 Adaptive & self-correcting retrieval

- **Self-RAG — Asai et al., 2023** (arXiv:2310.11511). Train the model to emit
  *reflection tokens*: decide whether retrieval is needed at all, critique whether
  retrieved passages are relevant, and whether its own generation is supported.
  Retrieval becomes conditional, not mandatory.
- **CRAG (Corrective RAG) — Yan et al., 2024** (arXiv:2401.15884). A lightweight
  evaluator grades retrieved documents; on low confidence, trigger a fallback (e.g.,
  web search) and decompose-then-recompose the evidence.
- **FLARE — Jiang et al., 2023** (arXiv:2305.06983). *Active* retrieval during
  generation: when the model's next-sentence confidence drops, pause, retrieve for
  the upcoming content, continue. Retrieval interleaved with generation instead of
  once upfront.

### 9.3 Index-structure innovations

- **RAPTOR — Sarthi et al., 2024** (arXiv:2401.18059). Recursively cluster chunks
  and summarize each cluster, building a tree of abstractions. Queries can retrieve
  at any level — details from leaves, themes from summary nodes. Fixes RAG's
  weakness on "summarize the whole corpus" questions.
- **GraphRAG — Edge et al., 2024 (Microsoft), "From Local to Global"**
  (arXiv:2404.16130). LLM extracts an entity-relationship knowledge graph from the
  corpus; community detection (Leiden) clusters it; pre-generated community
  summaries answer *global* questions ("what are the main themes?") that chunk
  retrieval fundamentally cannot.
- **Contextual retrieval (Anthropic, 2024)**: before embedding, prepend each chunk
  with an LLM-generated sentence situating it in its document ("This chunk is from
  the Q3 earnings section discussing..."). Large reduction in retrieval failures;
  cheap with prompt caching.

### 9.4 Agentic RAG

Instead of a fixed pipeline, an LLM agent with a search *tool* decides when to
search, what to search for, reads results, and searches again until it has enough —
retrieval in a loop, driven by reasoning. Precursors: **ReAct — Yao et al., 2022**
(arXiv:2210.03629), Toolformer (arXiv:2302.04761). This is increasingly the dominant
pattern for complex/multi-hop questions, at the price of latency and token cost.

### 9.5 Naming the landscape

**Gao et al., 2023, "Retrieval-Augmented Generation for Large Language Models: A
Survey"** (arXiv:2312.10997) — the standard survey; introduced the widely used
taxonomy **Naive RAG → Advanced RAG → Modular RAG**. Your project is deliberately at
stage 1; everything in this section is stage 2–3.

---

## 10. Evaluating RAG Systems

You cannot improve what you don't measure. Evaluate the two stages separately.

### Retrieval metrics

Given questions with known gold passages/documents:

- **Recall@k** — is the gold passage in the top k? (The metric that matters most:
  if it's not there, generation is doomed.)
- **MRR** (mean reciprocal rank) — 1/rank of the first relevant hit, averaged.
- **nDCG@k** — rank-discounted gain; standard in IR when relevance is graded.
- **Precision@k / context precision** — how much of what you retrieved is relevant
  (irrelevant chunks waste context and distract the model).

### Generation metrics (the RAG triad)

- **Faithfulness / groundedness** — is every claim in the answer supported by the
  retrieved context? (Measures hallucination.)
- **Answer relevance** — does the answer actually address the question?
- **Context relevance** — was the retrieved context relevant to the question?

These are typically scored by an **LLM-as-judge** — using a strong LLM to grade
outputs (validated in **Zheng et al., 2023, "Judging LLM-as-a-Judge with MT-Bench and
Chatbot Arena"**, arXiv:2306.05685).

### Frameworks & benchmarks

- **RAGAS — Es et al., 2023** (arXiv:2309.15217) — reference-free RAG evaluation
  implementing the triad above; the de-facto standard library.
- **BEIR — Thakur et al., 2021** (arXiv:2104.08663) — 18-dataset zero-shot retrieval
  benchmark; source of the finding that BM25 is a brutal baseline out-of-domain.
- **KILT — Petroni et al., 2020** (arXiv:2009.02252) — knowledge-intensive task
  suite with provenance annotations.
- **Natural Questions** (real Google queries), **HotpotQA** (multi-hop),
  **MS MARCO** (the dataset most embedding models are trained on).

### Practical recipe for your project

Build a small eval set: ~20 questions, each labeled with the file (or chunk) that
answers it. Measure Recall@3 and MRR on the TF-IDF index. Every future change —
different chunk size, neural embeddings, hybrid, reranking — reruns the same set.
Now upgrades are *provable*, not vibes.

---

## 11. Known Failure Modes

1. **Vocabulary mismatch** (sparse retrieval): query words ≠ document words → zero
   similarity. *Fix: dense/hybrid retrieval.*
2. **Lost in the middle — Liu et al., 2023** (arXiv:2307.03172): LLMs attend well to
   the beginning and end of long contexts and poorly to the middle. Stuffing 20
   chunks in makes the model *miss* the relevant one placed 10th. *Fix: retrieve
   fewer, better chunks; put the best first/last; rerank.*
3. **Bad chunking**: answer split across a chunk boundary; or a chunk lacks the
   context to be interpretable ("He then resigned" — who?). *Fix: overlap,
   structural/semantic chunking, contextual retrieval, parent-document.*
4. **Retrieval-generation mismatch**: chunk is topically similar but doesn't contain
   the answer; model hallucinates the gap. *Fix: reranking, groundedness checks,
   refusal instruction.*
5. **Knowledge conflict**: context contradicts parametric memory; model may ignore
   the context (Longpre et al., 2021). *Fix: strong grounding prompts; models
   trained for faithfulness.*
6. **Global questions**: "summarize the corpus," "what are the main themes" —
   top-k chunk retrieval structurally cannot answer these. *Fix: RAPTOR, GraphRAG.*
7. **Stale index**: documents changed, embeddings didn't. *Fix: re-indexing
   pipelines tied to document updates.*
8. **Distracting context**: irrelevant retrieved chunks measurably *lower* accuracy
   versus no context at all (shown across the RAG-robustness literature). *Fix:
   context precision thresholds — retrieve fewer when scores are low.*

---

## 12. RAG vs. Long Context vs. Fine-Tuning

Three ways to give an LLM knowledge; they answer different problems.

| | RAG | Long context (paste everything) | Fine-tuning |
|---|---|---|---|
| Knowledge freshness | Update index instantly | Fresh per call | Frozen at training |
| Cost per query | Low (small context) | High (pay for every token, every call) | Low at inference |
| Corpus size limit | ~Unbounded | Context window (and attention quality degrades) | Bounded by training capacity |
| Attribution | Natural (show chunks) | Possible | None |
| Best for | Facts, documents, freshness | Small corpora, whole-doc reasoning | Style, format, domain *behavior* |

Findings from the debate: with strong long-context models, pasting whole small
corpora can beat RAG on quality (no retrieval misses) but at 10–100× token cost;
"lost in the middle" and cost keep retrieval relevant at scale. Fine-tuning is
consistently *bad* at injecting facts but good at teaching behavior. Production
answer: **they compose** — fine-tune for behavior, RAG for knowledge, long context
for generous retrieval budgets.

---

## 13. Your Naive_rag.py, Mapped to This Guide

| Your code | Guide section | What it is | Standard upgrade |
|---|---|---|---|
| `load_and_chunk()` — 200 words, 40 overlap | §4 | Fixed-size chunking with overlap | Structural/semantic chunking; parent-document |
| `tokenize()` — lowercase, `[a-z0-9]+` | §5.1 | Basic normalization | Stemming/lemmatization; subword tokens |
| `TfidfIndex.__init__` — TF × IDF, L2-normalized | §5.1 | Sparse lexical retrieval (1972 tech, honest baseline) | BM25; dense bi-encoder (sentence-transformers); hybrid + RRF |
| `matrix @ query_vec` | §6 | Brute-force exact search — correct choice at 6 chunks | FAISS/HNSW past ~100k chunks |
| `retrieve()` top-3 | §7 | Single-stage retrieval | Retrieve 50 → cross-encoder rerank → top 3 |
| `build_prompt()` — "ONLY the context... say so", `[source]` tags | §8 | Grounded prompt with refusal path and provenance — already best-practice shape | Citation verification; chunk ordering |
| `generate()` → Ollama qwen3.5 | §8 | Frozen instruction-tuned LLM, in-context RAG (Ram et al., 2023) | Larger model; Self-RAG-style checks |
| *(missing)* | §10 | Evaluation | ~20-question eval set; Recall@3, MRR; RAGAS later |

Architecturally, your script is **DrQA (2017) with a 2024 generator** — sparse
retrieval feeding a neural reader. That is a legitimate, well-studied design point,
and it's the ideal baseline: every technique in Section 9 exists because of a failure
you can now reproduce on your own machine. Try: ask `"Who invented Python?"` —
TF-IDF finds it ("created" vs "invented" — enough words overlap). Then ask
`"Which beverage comes from roasted seeds?"` without the word "coffee" — and watch
sparse retrieval struggle. That experience *is* the DPR paper's motivation.

Sensible learning order from here (matches the research chronology):

1. **Eval set first** (§10) — otherwise upgrades are unmeasurable.
2. **Dense embeddings** (§5.2) — sentence-transformers, compare against TF-IDF on
   the eval set. Re-live the 2020 DPR moment.
3. **Hybrid + RRF** (§5.5) — see why production keeps both.
4. **Reranking** (§7) — biggest single quality jump.
5. **One advanced technique** (§9) — HyDE is the easiest to implement (one extra
   LLM call).

---

## 14. Reading List

Chronological; ★ = read these five first.

| Year | Paper | arXiv | Contribution |
|---|---|---|---|
| 2009 | Robertson & Zaragoza — BM25 | (Foundations & Trends in IR) | The lexical baseline |
| 2017 | ★ Chen et al. — DrQA | 1704.00051 | Retriever + neural reader template |
| 2019 | ★ Reimers & Gurevych — Sentence-BERT | 1908.10084 | Practical sentence embeddings |
| 2020 | Guu et al. — REALM | 2002.08909 | Retrieval in LM pretraining |
| 2020 | ★ Karpukhin et al. — DPR | 2004.04906 | Dense bi-encoder retrieval |
| 2020 | ★ Lewis et al. — RAG | 2005.11401 | Named and formalized RAG |
| 2020 | Khattab & Zaharia — ColBERT | 2004.12832 | Late interaction |
| 2021 | Izacard & Grave — FiD | 2007.01282 | Fusion-in-decoder |
| 2021 | Thakur et al. — BEIR | 2104.08663 | Zero-shot retrieval benchmark |
| 2021 | Formal et al. — SPLADE | 2107.05720 | Learned sparse retrieval |
| 2022 | Gao et al. — HyDE | 2212.10496 | Hypothetical document embeddings |
| 2023 | Ram et al. — In-Context RALM | 2302.00083 | RAG with frozen LLMs via prompting |
| 2023 | Liu et al. — Lost in the Middle | 2307.03172 | Long-context position bias |
| 2023 | Es et al. — RAGAS | 2309.15217 | Reference-free RAG evaluation |
| 2023 | Asai et al. — Self-RAG | 2310.11511 | Reflective, conditional retrieval |
| 2023 | ★ Gao et al. — RAG Survey | 2312.10997 | Naive→Advanced→Modular taxonomy |
| 2024 | Yan et al. — CRAG | 2401.15884 | Corrective retrieval |
| 2024 | Sarthi et al. — RAPTOR | 2401.18059 | Recursive summary trees |
| 2024 | Edge et al. — GraphRAG | 2404.16130 | Knowledge-graph RAG for global questions |
| 2024 | Günther et al. — Late Chunking | 2409.04701 | Context-aware chunk embeddings |

All arXiv papers: `https://arxiv.org/abs/<id>`.

---

*Companion to `RAG/Naive_rag.py`. Written 2026-07-19.*
