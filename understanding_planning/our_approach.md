# Our Approach — Deep Research Agent

> Our complete plan: the tech stack we picked, why we picked it, the end-to-end pipeline, and how every assignment requirement maps to a concrete piece of code.

---

## 1. Guiding Principles

Before picking any tool, we locked down five principles that drive every decision:

1. **Zero cost** — every API, model, and host must have a real, sustainable free tier (no trial credits that expire).
2. **No frameworks** — agent loop, planner, context builder, and session store are all written by hand in pure Python.
3. **Speed matters** — the user must see useful progress in under 2 seconds and a full answer in under 30 seconds for typical queries.
4. **Honesty over confidence** — the agent must admit when evidence is weak or conflicting; we will not let it bluff.
5. **Reviewer-friendly** — one `git clone`, one `pip install`, one secret in `.env`, one `streamlit run`. No exotic setup.

---

## 2. Final Tech Stack (with Rationale)

| Layer | Choice | Why we picked it (over alternatives) |
|------|--------|--------------------------------------|
| **Language** | Python 3.11 | Required by assignment; 3.11 has the best async + typing story. |
| **LLM (primary)** | **Groq — Llama 3.3 70B Versatile** (free tier) | ~300 tokens/sec via LPU hardware — by far the fastest free option, perfect for a streaming UX. 30 RPM, 14.4K requests/day. No card. |
| **LLM (fallback)** | **Google Gemini 2.5 Flash** (free tier) | 1,500 requests/day and a 1M-token context window for long-context fallback. We auto-failover when Groq rate-limits. |
| **Web search** | **Tavily Search API** (free tier) | Purpose-built for AI agents — returns clean, LLM-ready text. 1,000 credits/month, no card. Basic search = 1 credit. Comes with content-extraction endpoint for free. Serper would force us to re-process raw SERPs; Parallel has weaker docs. |
| **Page fetching** | **httpx (async)** | Modern, async-first, HTTP/2, much faster concurrent fetches than `requests`. We fetch up to 6 pages in parallel. |
| **HTML → text** | **Trafilatura 2.x** | Highest benchmarked F1 (0.958) for article extraction. Used by HuggingFace, IBM, Microsoft, Stanford. Falls back to readability + jusText internally. |
| **Embeddings (context selection)** | **sentence-transformers/all-MiniLM-L6-v2** | 384 dims, ~22M params, runs on CPU in milliseconds. Good enough for snippet re-ranking; keeps everything free and local. |
| **Token counting** | **tiktoken** | Reliable token counts to enforce hard context limits. |
| **Session storage** | **SQLite** (via `sqlite3` stdlib) | Atomic, persistent, queryable, zero dependencies. JSON files would race and corrupt on concurrent turns. |
| **Data validation** | **Pydantic v2** | Strict typed models for `SearchResult`, `Snippet`, `Turn`, `Session`. Catches bad data at boundaries. |
| **UI** | **Streamlit 1.x** | `st.status` + `st.write_stream` are purpose-built for our exact streaming-progress pattern. Better DX than Gradio for this. |
| **Concurrency** | **asyncio + httpx.AsyncClient** | Parallel page fetching is the single biggest latency win. |
| **Config** | **python-dotenv** | Simple `.env` locally; HF Spaces secrets in prod. |
| **Testing** | **pytest** | Standard for the evaluation harness. |
| **Hosting** | **Hugging Face Spaces** (free) | 2 CPU / 16 GB RAM / 50 GB disk — substantially better than Streamlit Cloud's 690 MB–2.7 GB. Native Streamlit support, built-in secrets, sleeps only after 48h (vs 12h on Streamlit Cloud). |

### Why these, not the obvious alternatives

- **Groq vs OpenAI** — OpenAI has no real free tier in 2026; Groq is genuinely free and 5–10× faster.
- **Tavily vs Serper** — Serper returns raw Google SERPs that we'd have to parse and clean. Tavily returns cleaned, LLM-optimized text out of the box, saving us code and credits.
- **Trafilatura vs BeautifulSoup** — Soup gives us a tree, not "the article". Trafilatura's heuristics correctly strip nav/ads/footers across thousands of site layouts.
- **SQLite vs JSON files** — Concurrent turns + crash safety are real risks; SQLite handles both for free.
- **HF Spaces vs Streamlit Cloud** — Streamlit Cloud's 690 MB RAM is too small to load `sentence-transformers` reliably. HF gives us 16 GB.
- **Streamlit vs Gradio** — `st.status` containers are exactly the "phase labels with rolling logs" pattern the assignment asks for.

---

## 3. System Architecture (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (app.py)                       │
│   chat input  •  session selector  •  live status  •  citations     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ user query + session_id
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Agent Orchestrator (agent.py)                   │
│   handwritten loop:  PLAN → SEARCH → FETCH → SELECT → ANSWER        │
│   yields ProgressEvent objects on every phase change                │
└──┬──────────────┬───────────────┬──────────────┬───────────────┬────┘
   │              │               │              │               │
   ▼              ▼               ▼              ▼               ▼
┌──────┐   ┌──────────┐   ┌────────────┐  ┌────────────┐  ┌──────────┐
│Planner│   │  Search  │   │   Fetch    │  │  Context   │  │  Answer  │
│(LLM) │   │ (Tavily) │   │ (httpx +   │  │  Selector  │  │   (LLM,  │
│      │   │          │   │ Trafilatura│  │ (MiniLM    │  │ grounded)│
│      │   │          │   │            │  │  re-rank)  │  │          │
└──────┘   └──────────┘   └────────────┘  └────────────┘  └─────┬────┘
                                                                 │
                                                                 ▼
                                              ┌───────────────────────────┐
                                              │   Citation Formatter      │
                                              │   [Title — domain](url)   │
                                              └───────────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Session Store  (SQLite — session.db)               │
│   • sessions     • messages (with timestamps)                       │
│   • turns        • snippets (with full provenance)                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. The Agent Loop (Hand-Built, No Frameworks)

The orchestrator is a single Python function that runs as an **async generator**, yielding `ProgressEvent`s after every phase. The UI consumes these events with `st.status` and `st.write_stream`. Internally:

### Phase 1 — Plan
- Input: the user query + a rolling summary of prior conversation.
- LLM call (Groq, low temperature) with a **strict JSON-only system prompt** asking for:
  - 2–4 sub-questions covering the topic
  - 3–6 concrete search queries
  - A short "research goal" sentence
- Output is validated by Pydantic. If parsing fails, we retry once with stricter formatting instructions.
- **Why this matters**: Without a plan, the agent issues lazy single-query searches and misses multi-hop questions.

### Phase 2 — Search
- For each planned search query, call **Tavily's `search` endpoint** in parallel (asyncio).
- Each result is normalized to a `SearchResult(title, url, snippet, score, domain, query)`.
- We **deduplicate by URL** and keep the top N (default 8) by score after merging across queries.
- Tavily returns `score` so we use it as the initial relevance signal.

### Phase 3 — Fetch & Extract
- For each selected URL, asynchronously:
  1. `httpx.AsyncClient` with a 10 s timeout, browser-like User-Agent, and follow_redirects=True.
  2. Run **Trafilatura** on the HTML to extract clean main-text content.
  3. Tag the result with `url, title, domain, retrieved_at`.
- Failures (timeouts, 4xx, JS-only pages) are recorded but do not crash the run.
- We chunk extracted text into ~600-token windows with 80-token overlap so the re-ranker has clean snippet candidates.

### Phase 4 — Context Selection (the most important step)
This is where we earn "Quality of context selection" marks. We do **three passes**:

1. **Semantic re-rank** — Embed the user query with `all-MiniLM-L6-v2`; embed every snippet; compute cosine similarity; rank.
2. **Diversity penalty (MMR)** — Apply Maximal Marginal Relevance so we don't pick 5 snippets from the same domain. Each pick is penalized by similarity to already-picked snippets.
3. **Token budget** — Greedy-fill until we hit our hard budget (default 4,000 tokens for snippets). Snippets that overflow are dropped, not truncated mid-sentence.

Output: an ordered `List[Snippet]` where every snippet carries `(id, text, url, title, domain, retrieved_at, score)`.

### Phase 5 — Answer (Grounded Generation)
- Build the **final prompt** with five sections:
  1. System: agent role + citation rules + uncertainty rules
  2. Rolling summary of prior conversation (if any)
  3. Recent relevant turns
  4. Current user query
  5. Selected snippets, each labeled `[S1]…[Sn]` with its title, domain, URL
- The model is instructed to:
  - Cite using `[S#]` inline, expanded to `[Title — domain](URL)` in the final pass.
  - Explicitly flag disagreements (e.g., *"Source A claims X while Source B claims Y…"*).
  - Say *"I could not find sufficient evidence"* when applicable, and propose next research steps.
- We **stream the answer** with `st.write_stream` on top of the Groq streaming SDK.

### Phase 6 — Persist Turn
After streaming completes, we write **one atomic SQLite transaction** containing:
- The user message and assistant message (with timestamps)
- The full Turn record: query, search_queries, urls_opened, selected_snippets, final_answer
- Each snippet is stored with its full provenance for the audit trail.

### Phase 7 — Conversation Summarization (when needed)
- If the cumulative conversation tokens exceed a threshold (default 3,000 tokens), we summarize older turns into a `rolling_summary` field on the session.
- This summary becomes part of every future prompt instead of the full transcript.
- Summarization itself is a cheap Groq call with a fixed prompt template.

---

## 5. Streaming Progress Pattern

We emit a typed `ProgressEvent` for each operational step. The UI maps these to `st.status` updates:

| Phase | UI Label | Payload Shown |
|-------|----------|---------------|
| `PLAN_START` | "Planning research strategy…" | (none) |
| `PLAN_DONE` | "Plan ready" | sub-questions, search queries |
| `SEARCH_START` | "Searching the web…" | the queries being issued |
| `SEARCH_DONE` | "Found N results" | result titles + domains |
| `FETCH_START` | "Fetching N pages…" | URLs |
| `FETCH_PROGRESS` | "Fetched X of N" | per-URL OK/FAIL |
| `SELECT_DONE` | "Selected K snippets from D domains" | picked snippets preview |
| `ANSWER_START` | "Generating grounded answer…" | (token stream begins) |
| `ANSWER_TOKEN` | (streamed inline) | text delta |
| `DONE` | "Done in T seconds" | latency + citation count |

**Important:** we never stream the LLM's internal chain-of-thought — only operational status, as the assignment requires.

---

## 6. Citation Strategy

Every claim that cites a source uses the format the assignment specifies:

> `[Title — domain](URL)`

To enforce this:
1. The system prompt embeds the exact format and forbids invented URLs.
2. We pass the LLM a `[S#]` placeholder map. After generation, a **post-processor** expands every `[S#]` into the full `[Title — domain](URL)` and verifies that the URL exists in the snippet table for the current turn.
3. Any `[S#]` that doesn't map to a real snippet is **rejected** — we re-prompt the model once asking it to remove unsupported claims.
4. If sources conflict, the prompt template includes a *"Conflicts"* sub-section that the model is asked to fill explicitly.

This gives us strong **citation integrity** — every reference in the final answer is guaranteed to point to a page the agent actually fetched in this turn.

---

## 7. Session & History Schema (SQLite)

```sql
CREATE TABLE sessions (
  session_id      TEXT PRIMARY KEY,
  created_at      TEXT NOT NULL,
  rolling_summary TEXT
);

CREATE TABLE messages (
  message_id  INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT NOT NULL REFERENCES sessions(session_id),
  role        TEXT NOT NULL,        -- 'user' | 'assistant'
  content     TEXT NOT NULL,
  ts          TEXT NOT NULL
);

CREATE TABLE turns (
  turn_id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id      TEXT NOT NULL REFERENCES sessions(session_id),
  query           TEXT NOT NULL,
  plan_json       TEXT,
  search_queries  TEXT,             -- JSON array
  urls_opened     TEXT,             -- JSON array
  final_answer    TEXT,
  ts              TEXT NOT NULL,
  latency_ms      INTEGER
);

CREATE TABLE snippets (
  snippet_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  turn_id      INTEGER NOT NULL REFERENCES turns(turn_id),
  url          TEXT NOT NULL,
  title        TEXT,
  domain       TEXT,
  text         TEXT NOT NULL,
  score        REAL,
  retrieved_at TEXT NOT NULL
);
```

This schema gives the reviewer one file (`session.db`) they can open to inspect the *entire* run history including every snippet selected.

---

## 8. Evaluation Harness Plan

A standalone script `eval/run_eval.py` that exercises the agent on a curated dataset and produces a Markdown + JSON report.

### Dataset (≈ 20 questions, hand-curated)
Each question is tagged with a `category` and `expected_behavior`:

| Category | Example | What we check |
|---------|---------|---------------|
| Factual | *"When was the Sarvam-1 model released?"* | Single correct fact, single citation. |
| Multi-hop | *"Which Indian states had GDP growth above 8% in FY24, and which one had the highest IT-sector contribution among them?"* | Multiple sources combined, all cited. |
| Comparison | *"Compare context length of Llama 3.3 70B and Gemini 2.5 Flash."* | Both entities covered, both cited. |
| Insufficient evidence | *"What is the home address of Sarvam AI's CEO?"* | Agent must refuse / say insufficient evidence. |
| Conflicting sources | *"Is intermittent fasting safe for diabetics?"* | Must flag the disagreement and cite both sides. |
| Recency-sensitive | *"What is the latest version of Llama released?"* | Must prefer recent sources. |

### Metrics (each scored 0–1; final report shows per-category averages)

| Metric | How we compute it | Why we chose it |
|--------|-------------------|-----------------|
| **Citation Coverage** | % of fact-bearing sentences in the answer that carry at least one citation. Computed via regex + sentence-splitter. | Directly measures grounding; cheap and automatic. |
| **Citation Validity** | % of cited URLs that (a) exist in the turn's snippet table and (b) return HTTP 200 on a re-fetch. | Catches fabricated URLs — the single biggest risk in RAG. |
| **Faithfulness (LLM-as-Judge)** | A judge LLM is shown the snippets + answer and asked: *"For each claim in the answer, is it supported by the cited snippet? (Yes / No / Partial)"* | Industry-standard RAG faithfulness measure. |
| **Source Diversity** | Number of unique domains cited ÷ number of citations. | Penalizes one-domain answers. |
| **Conflict Handling** | On the `conflicting_sources` subset, did the answer explicitly use words like "disagree", "however", "conflicting", AND cite both sides? Binary. | Directly tests the assignment's conflict requirement. |
| **Refusal Quality** | On the `insufficient_evidence` subset, did the agent refuse appropriately? Binary. | Tests honesty. |
| **Multi-turn Robustness** | A 3-turn follow-up dialogue per question; we check that turn 2 and 3 correctly reference earlier turns. | Tests session/context management. |
| **Latency** | Wall-clock seconds per turn (mean, p95). | Tracks user experience. |

The script outputs:
- `eval/results.json` — per-question raw scores
- `eval/report.md` — pretty summary with per-category tables
- `eval/sample_runs/` — full transcripts for the reviewer to spot-check

---

## 9. Repository Layout (Clean & Modular)

```
SarvamAi/
├── README.md                    # full design note, setup, demo link
├── requirements.txt
├── .env.example
├── app.py                       # Streamlit UI
├── deep_research/
│   ├── __init__.py
│   ├── agent.py                 # the orchestrator + async generator
│   ├── llm.py                   # Groq + Gemini clients with failover
│   ├── search.py                # Tavily wrapper
│   ├── fetch.py                 # async httpx + trafilatura
│   ├── context.py               # MMR re-ranker, token budgeter
│   ├── prompts.py               # all prompt templates (one place)
│   ├── citations.py             # [S#] → [Title — domain](URL) expander
│   ├── session.py               # SQLite store
│   ├── summarizer.py            # rolling-summary helper
│   ├── models.py                # Pydantic models
│   └── progress.py              # ProgressEvent types
├── eval/
│   ├── dataset.json             # ~20 questions, tagged
│   ├── run_eval.py
│   ├── judges.py                # LLM-as-judge prompts
│   ├── results.json             # produced by run_eval
│   └── report.md                # produced by run_eval
└── understanding_planning/
    ├── assignment.txt
    ├── what_is_asked.md
    └── our_approach.md          # (this file)
```

Every module has a single responsibility. Nothing imports across modules in ways that hide control flow.

---

## 10. Day-Plan (Single Day, Realistic)

| Block | Work | Output |
|------|------|--------|
| **Hour 0** | This research + planning (already done) | the two `.md` files in this folder |
| **Hour 1** | Repo skeleton, `requirements.txt`, `.env.example`, Pydantic models, SQLite schema | importable empty modules |
| **Hour 2** | `llm.py` (Groq + Gemini with failover), `search.py` (Tavily) | working LLM + search calls in a notebook |
| **Hour 3** | `fetch.py` (async httpx + Trafilatura), `context.py` (MMR + token budget) | end-to-end fetch-and-select working |
| **Hour 4** | `agent.py` orchestrator with `ProgressEvent` async generator, `citations.py` post-processor | full agent loop working in CLI |
| **Hour 5** | `app.py` Streamlit UI with `st.status` + `st.write_stream` + session selector | demo-ready local app |
| **Hour 6** | `session.py` persistence, conversation summarization, multi-turn handling | sessions survive restarts |
| **Hour 7** | `eval/dataset.json` (20 questions) + `eval/run_eval.py` + LLM-as-judge | reports generated |
| **Hour 8** | README (with design note), polish error messages, record video demo | submission-ready |
| **Hour 9** | Deploy to Hugging Face Spaces, add secrets, smoke-test | live URL |
| **Hour 10** | Buffer for fixes, write submission PDF | final deliverable |

---

## 11. Risk Register & Mitigations

| Risk | Mitigation |
|------|-----------|
| Tavily rate limit (1k/month) | Cache search results per `(query, day)` in SQLite. Each eval run shouldn't burn more than ~150 credits. |
| Groq rate limit (30 RPM) | Exponential backoff + automatic failover to Gemini Flash. |
| Page fetch failures (paywalls, JS-only sites) | Try `trafilatura.fetch_url` first; fall back to httpx + trafilatura on raw HTML; never block on a single failed URL. |
| Token-window overflow | Hard budget enforcement in `context.py` + rolling summary for old turns. |
| Citation hallucination | Strict `[S#]` placeholder + post-validate every citation against the snippet table; reject + re-prompt if invalid. |
| Conflicting sources missed | Dedicated *"Conflicts"* sub-section in the answer prompt; eval metric explicitly checks for it. |
| HF Space cold start (48 h sleep) | Add a small note in README; cold start ≈ 30 s, acceptable for a demo. |
| Secret leakage | `.env` git-ignored locally; `st.secrets` + HF Space secrets in prod. Never log API keys. |

---

## 12. Why This Plan Wins on the Rubric

The reviewer scores us on five criteria. Here's exactly how our plan maps to each:

| Rubric criterion | Where it's earned |
|------------------|-------------------|
| **Soundness of evaluation metrics** | Section 8 — 8 metrics covering grounding, validity, conflict, refusal, diversity, multi-turn, latency. Each has a written rationale. |
| **Citation integrity** | Section 6 — strict `[S#]` placeholders, post-validation against the snippet table, automatic re-prompt on invalid citations. |
| **Quality of context selection & conflict handling** | Section 4 Phase 4 — semantic re-rank + MMR diversity + token budget. Conflict handling baked into the answer prompt. |
| **Session & context management** | Sections 4, 7 — SQLite schema with full turn provenance; rolling summarization when history grows. |
| **Code quality & modularity** | Section 9 — one module per responsibility; Pydantic at boundaries; typed events; no framework opacity. |

---

## 13. One-Sentence Summary of Our Approach

> A single-file Streamlit UI on Hugging Face Spaces drives a hand-written Python async agent that plans → searches with Tavily → fetches and extracts with Trafilatura → re-ranks snippets with MiniLM embeddings → answers with Groq Llama 3.3 70B under strict citation rules → and persists every turn to SQLite, with a custom 8-metric evaluation harness proving it works.
