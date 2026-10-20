# Stage 7 — Enterprise RAG (what companies actually need)

**Decision (2026-10-19):** stop tuning retrieval quality on the synthetic gold
set. From here on, build what a company deployment needs. Retrieval changes are
only in scope when they fall out of an enterprise requirement (e.g. metadata
filters for permissions also fix wrong-company misses).

## Why pivot

Stage 6 closed at **80.3% end-to-end** (`rag_stage_6/v4_a3_rejudge.json`,
after the judge fix: full gold passage instead of `text[:1500]`). That number
does **not** predict production accuracy:

1. **Synthetic questions.** Gold Qs are LLM-generated from one chunk each
   (`_gen`), name the entity, and have a single gold passage. Real users ask
   vague, comparative, multi-doc, follow-up, out-of-scope questions.
2. **40% corpus.** Full corpus = more look-alike docs = more wrong-entity
   confusion (already the #1 miss cause).
3. **Tuned on the test set.** `grader_chars`, `top_k` etc. were picked on the
   same 318 Q. No held-out split → optimistic.
4. **Noise floor.** 279 positives → 95% CI ≈ ±4.7 pts. Recent 1–2 pt moves are
   inside it. Fixing 6 or 24 questions is tuning to noise.
5. **No gold in prod.** Correctness has to be measured another way.

The gold set stays useful for **A/B and regression checks**, not for an
absolute "it works" claim.

## What carries over from stages 1–6

| Component | Reuse |
|---|---|
| Parent-child chunking, heading-aware splitting | as-is |
| pgvector + BGE-M3 dense retrieval (self-hostable) | as-is |
| CRAG loop: grade → refine → **refuse** | as-is; refusal matters more than accuracy in prod |
| Failure attribution (`retrieval_miss` / `refused_on_hit` / `answered_wrong`) | as-is |
| Spend guards (`--max-usd`), LLM cache, `--rejudge` | as-is |
| Lessons: BM25/reranker null on this data, metadata is the real lever | guides priorities |

Not reused: tuned knob values, the 80.3% number, LLM-generated eval questions.

## Gaps — ranked by "can't ship without it"

| # | Gap | Why companies need it |
|---|---|---|
| 1 | **Access control (ACL)** | User must never retrieve a doc they can't open (HR, finance, legal). #1 blocker. |
| 2 | **Metadata layer** | company/dept, doc type, date, version, owner, ACL on every chunk. Foundation for 1, 4, and wrong-entity misses. |
| 3 | **Ingestion + freshness** | Connectors (Confluence, SharePoint, Drive, Slack, tickets); incremental re-index on change; **delete** on removal. |
| 4 | **Version conflicts** | Old and new policy both match. Prefer latest / flag conflict. Same root cause as wrong-company. |
| 5 | **Citations** | Every answer links to source chunk. Trust + auditability. |
| 6 | **Multi-turn** | "what about last year?" → condense query with chat history. |
| 7 | **Prod monitoring** | Reference-free faithfulness (answer grounded in retrieved ctx), refusal rate, thumbs, weekly sampled review. |
| 8 | **PII / data residency** | Decide what may leave the network to the Claude API; what stays local. |
| 9 | **Serving** | API, auth, latency budget, tracing, cost per query. |

## Plan (phases, each with a verify gate)

0. **Honest eval first** → verify: frozen `qa_real_v1.json`, 50–100 realistic
   Qs (vague, comparative, multi-doc, follow-up, out-of-scope), never tuned on.
   Score current pipeline once = real baseline. Plus reference-free
   faithfulness scorer.
1. **Metadata layer** → verify: every chunk carries `{entity, doc_type, date,
   version, acl}`; filtered retrieval via pgvector jsonb; wrong-entity misses
   drop on v4, no regression on `qa_real_v1`.
2. **ACL enforcement** → verify: test with 2 synthetic users / 2 permission
   groups; 0 cross-group chunks retrieved across all eval Qs (hard gate, not a %).
3. **Incremental ingestion** → verify: edit / add / delete a doc → index
   reflects it, deleted doc never retrieved again.
4. **Version handling** → verify: seeded old/new policy pair; answer cites newest.
5. **Citations + API** → verify: every non-refusal answer has ≥1 valid source link.
6. **Multi-turn** → verify: follow-up Qs in `qa_real_v1` answered with history.
7. **Monitoring** → verify: faithfulness + refusal + cost per query logged per request.
8. **Tool-using agent (agentic level 2)** → planner decomposes multi-part /
   comparative Qs and picks tools: filtered search, calculator, SQL over
   extracted financial tables. Fixes LLM arithmetic (e.g. Bancorp
   pre-dividend sum). → verify: multi-hop + numeric slice of `qa_real_v1`
   improves; single-hop no regression; latency/cost per Q within budget.
9. **Learn from mistakes (agentic level 3)** → needs phase 5 citations + a
   feedback signal (thumbs, user corrections, low faithfulness, logged
   refusals). Cheapest mechanism first, stop at the first that moves the gate:
   1. correction memory — failed Q + approved fix, retrieved as hint for similar Qs
   2. query-rewrite memory — reuse rewrites that succeeded
   3. doc-level boost/demote from feedback
   4. offline prompt optimization over logged failures (weekly)
   5. embedder/reranker fine-tune on logged (query, good/bad doc) pairs

   Guardrails: memory is **ACL-scoped** (a correction citing an HR doc never
   reaches users outside that group); human approval before a correction
   enters memory (no poisoning); learning runs as an **offline loop** (log →
   review → update → eval gate → deploy), never self-modifying online.
   → verify: frozen `qa_real_v1` improves over phase 8 baseline; ACL test from
   phase 2 still 0 cross-group leaks with memory on.

### Agentic levels

| Level | Meaning | Status |
|---|---|---|
| 1 | Self-correct within one query (grade → refine → re-retrieve → refuse) | ✅ CRAG, stage 6 |
| 2 | Tool-using planner (decompose, filtered search, calculator, SQL) | phase 8 |
| 3 | Learns across queries from feedback | phase 9 (after ACL + feedback capture) |

## Non-goals (for now)

Retrieval-only tuning on v4 (BM25, rerankers, embedder A/B, chunk-size sweeps),
web fallback, any fine-tuning before phase 9.

## Open decisions

- Target company data shape for phase 0: keep SEC/arxiv/docs corpus as a
  stand-in, or get a sample of real internal-style docs (wiki, policies, tickets)?
- PII policy: Claude API for everything, or local model for sensitive tenants?
