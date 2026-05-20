---
title: Deep Research Agent
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.57.0
app_file: app.py
pinned: false
short_description: A from-scratch web research agent with citations, sessions, and a custom eval harness.
---

# Deep Research Agent

> A from-scratch Python web-research AI agent. Searches the live web, reads the actual
> pages, picks the best snippets, and writes citation-grounded answers — with full
> session persistence, streaming intermediate progress, and a custom evaluation harness.

**No agent framework used.** No LangChain, LangGraph, CrewAI, LlamaIndex, or Haystack.
The agent loop is hand-written Python.

- **Live demo:** `<add Hugging Face Spaces URL here>`
- **Video walkthrough:** `<add 2–3 minute Loom/YouTube link here>`

---

## Table of contents

1. [Quickstart](#quickstart)
2. [Design Note (Part 1)](#design-note-part-1)
3. [Architecture](#architecture)
4. [Repository layout](#repository-layout)
5. [Configuration](#configuration)
6. [Example conversations](#example-conversations)
7. [Evaluation harness](#evaluation-harness)
8. [Limitations & future improvements](#limitations--future-improvements)
9. [Deployment — Hugging Face Spaces](#deployment--hugging-face-spaces)
10. [Assumptions](#assumptions)

---

## Quickstart

### 1. Requirements

- Python **3.11+** (works on 3.11 / 3.12 / 3.13)
- A Tavily key (free tier, no credit card) — https://app.tavily.com
- A Groq key (free tier, no credit card) — https://console.groq.com
- Optional — pick one of these as a fallback:
  - **Local Ollama** (recommended for offline / unlimited use) — https://ollama.com
  - **Google Gemini** (free, 1,500 req/day) — https://aistudio.google.com

The LLM fallback order is **Groq → Gemini → Ollama**; each tier is silently skipped if unconfigured.

### 2. Install

```bash
git clone <your-repo-url>
cd SarvamAi
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# open .env and paste your three API keys
```

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), create a session in the sidebar,
and ask a question. You will see live phase updates (planning → searching → fetching →
selecting context → streaming answer) with citations expanded inline and a "Sources"
appendix added automatically.

### 4. Run the evaluation

```bash
python -m eval.run_eval                 # full dataset (~17 questions + follow-ups)
python -m eval.run_eval --limit 3       # quick smoke test
python -m eval.run_eval --skip-judge    # skip LLM-as-judge (only auto-metrics)
```

Outputs are written to `eval/results.json` and `eval/report.md`.

### 5. Run the offline smoke tests (no API keys needed)

```bash
python tests/test_smoke.py        # exercises every module in isolation
python tests/test_smoke_e2e.py    # full agent pipeline with mocked LLM/Search/Fetch
```

---

## Design Note (Part 1)

### Target users and the problem being solved

**Who:** technical knowledge workers (researchers, analysts, founders, journalists,
engineers) who need to make a defensible factual claim *quickly* and want to see *where*
the claim came from. They are not satisfied with a generic chatbot answer — they need
**citable sources** and they need to know when the agent is **unsure**.

**Problem:** stock LLMs hallucinate, are stale, and never tell you *where* a fact came
from. Existing search engines dump ten blue links and leave the synthesis to you.
Browsing-enabled assistants help but often pick a single dubious source. A *deep research
agent* sits between the two: it plans, performs multiple targeted searches, opens
several pages in parallel, picks the most relevant passages from different domains,
and writes a citation-grounded answer that says "I'm not sure" when the evidence is
weak or conflicting.

### Our definition of "deep research"

For this implementation, **deep research** means an answer that satisfies all five
properties:

1. **Multi-query planning** — the agent decomposes the user question into 3–6 concrete
   search queries before fetching.
2. **Multi-source reading** — content is fetched from at least 4–6 distinct URLs and
   the final context spans **multiple domains** (enforced by MMR + domain penalty).
3. **Grounded synthesis** — every factual sentence is tied to a snippet that was
   actually fetched in this turn (verified by post-processor).
4. **Honest uncertainty** — when evidence is weak, missing, or contradictory, the
   agent says so *explicitly* and proposes next steps rather than bluffing.
5. **Transparent process** — the user sees the plan, the queries, the URLs opened,
   the snippets selected, and the latency for every turn.

### Success metrics (5)

We picked metrics that together cover correctness *and* trustworthiness:

| Metric | What it measures | How computed |
|---|---|---|
| **Citation coverage** | Are factual sentences cited at all? | Automated: regex on raw `[S#]` placeholders against factual-looking sentences. |
| **Citation validity** | Do the cited ids correspond to real fetched snippets? | Automated: cross-check every `[S#]` against the turn's snippet table. |
| **Faithfulness** (LLM-as-judge) | Are the claims actually supported by the cited text? | A judge LLM scores `supported_claims / total_claims`. |
| **Source diversity** | How many distinct domains contribute to citations? | Unique cited domains / total citations. |
| **Conflict & refusal correctness** | On the dedicated categories, does the agent flag conflicts and refuse appropriately? | Pattern match for hedging/refusal phrases + ≥ 2 domains on conflict-class items. |

A **latency mean + p95** is also reported alongside — quality at unbounded latency is
not interesting.

### Data flow and components

```
┌──────────────┐  query  ┌────────────────────┐  plan  ┌──────────────┐
│ Streamlit UI │ ──────▶ │ Orchestrator       │ ─────▶ │  Tavily      │
│  app.py      │         │ deep_research/     │        │  search.py   │
│              │ ◀────── │   agent.py         │ ◀───── │              │
└──────────────┘ events  └────────────────────┘ results└──────────────┘
                              │  ▲                              │
                       URLs to│  │ chosen snippets              │
                       fetch  │  │                              ▼
                              ▼  │                       ┌──────────────┐
                       ┌────────────┐  fetched pages     │ httpx async  │
                       │ context.py │ ◀───────────────── │  + Trafilat. │
                       │ MMR + MiniLM│                    │  fetch.py    │
                       │ + budget    │                    └──────────────┘
                       └────────────┘
                              │
                              ▼
                       ┌────────────┐    streamed   ┌───────────────┐
                       │ Answer LLM │ ─────────────▶│ Citation post-│
                       │ Groq/Gemini│               │ processor +   │
                       └────────────┘               │ validator     │
                                                    └───────────────┘
                                                             │
                                                             ▼
                                                    ┌───────────────┐
                                                    │ SQLite store  │
                                                    │  session.db   │
                                                    └───────────────┘
```

Phases (one async generator inside `agent.py`):

1. **Plan** — Groq returns strict JSON with `research_goal`, `sub_questions`, `search_queries`.
2. **Search** — Tavily called in parallel for every planned query; results deduped by URL.
3. **Fetch** — `httpx.AsyncClient` fetches the top N URLs concurrently; Trafilatura extracts main-article text; BeautifulSoup is the fallback.
4. **Select context** — text is chunked into ~220-word overlapping windows; fastembed (BGE-small) embeds query + chunks; MMR + a domain-diversity penalty pick a budget-respecting set; chunks become `Snippet`s with stable ids `S1, S2, …`.
5. **Answer** — Groq streams a markdown answer that cites with `[S#]`. The post-processor expands each `[S#]` to `[Title — domain](URL)`, drops invalid refs, and appends a Sources list.
6. **Persist** — message + turn + every selected snippet are saved atomically in SQLite.
7. **Summarize** — when cumulative conversation tokens exceed 3,000, older turns are folded into a rolling summary used by future prompts.

### Risks and limitations

| Risk | Mitigation in this implementation |
|---|---|
| **Search-provider rate limits** (Tavily 1k credits/month) | Each query batch is capped at 4–6 advanced calls. Eval reuses sessions and keeps the dataset size moderate (~17 + follow-ups). |
| **LLM rate limits** (Groq 30 RPM / 14.4K RPD) | Automatic failover to Gemini 2.5 Flash. Token budgets keep prompts small. |
| **Low-quality sources** (SEO spam, AI-generated content) | Tavily already filters aggressively; we still apply MMR which prefers diverse domains. We display every URL in the audit panel for human spot-checking. |
| **Conflicting sources** | Answer prompt mandates a "Conflicts" sub-section; eval has a dedicated category with a pass/fail check. |
| **Context length blowup** in long sessions | Rolling summarization at 3,000 tokens; only the last two (user, assistant) pairs are kept verbatim. |
| **Page-fetch failures** (JS-only sites, paywalls, 4xx) | Failures are recorded but never block the rest of the run; we proceed with the pages we did get. |
| **Citation hallucination** | Strict `[S#]` placeholders + post-processor that strips any unknown id. |

### Future improvements (2)

1. **Re-prompting on invalid citations.** Today we silently drop unknown `[S#]`
   references. A second pass that re-prompts the model with the list of dropped
   references would push citation validity even higher.
2. **Self-critic with web-search verification.** Add a final verification phase
   where the agent picks 2–3 high-value claims from its own answer and re-issues
   targeted searches to confirm them, raising or lowering its hedging accordingly.

---

## Architecture

```
SarvamAi/
├── app.py                       Streamlit UI + async-to-sync event pump
├── requirements.txt
├── .env.example
├── deep_research/
│   ├── agent.py                 Hand-built orchestrator (async generator)
│   ├── llm.py                   Groq + Gemini clients with failover
│   ├── search.py                Tavily async wrapper (httpx)
│   ├── fetch.py                 Async page fetch + Trafilatura extraction + chunking
│   ├── context.py               MMR re-rank + domain diversity + token budget
│   ├── citations.py             [S#] expansion + validator + Sources appendix
│   ├── session.py               SQLite store (sessions / messages / turns / snippets)
│   ├── summarizer.py            Rolling conversation summarizer
│   ├── prompts.py               All prompt templates (planner, answer, summarizer, judge)
│   ├── progress.py              Typed ProgressEvent
│   └── models.py                Pydantic models (SearchResult, Snippet, Plan, Turn, …)
├── eval/
│   ├── dataset.json             ~17 hand-curated questions + multi-turn follow-ups
│   ├── run_eval.py              Runs the agent + scores it, writes report.md
│   ├── judges.py                LLM-as-judge wrapper
│   ├── results.json             (produced by run_eval)
│   └── report.md                (produced by run_eval)
└── understanding_planning/
    ├── assignment.txt
    ├── what_is_asked.md
    └── our_approach.md          (the full plan, written before any code)
```

---

## Configuration

All knobs are exposed via environment variables (or `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `TAVILY_API_KEY` | — (required) | Web search |
| `GROQ_API_KEY` | — (required) | Primary LLM |
| `GEMINI_API_KEY` | — (optional) | Cloud fallback LLM |
| `OLLAMA_URL` | `http://localhost:11434` | Local fallback LLM endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name (must be `ollama pull`-ed first) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | |
| `GEMINI_MODEL` | `gemini-2.5-flash` | |
| `MAX_SEARCH_RESULTS` | `8` | Cap on results merged across queries |
| `MAX_PAGES_TO_FETCH` | `6` | Cap on pages opened per turn |
| `MAX_CONTEXT_TOKENS` | `4000` | Hard budget for snippet tokens in the LLM prompt |
| `SESSION_DB_PATH` | `session.db` | Local SQLite file |

### Setting up Ollama as a fallback (optional)

```bash
# Install (macOS)
brew install ollama
# Or: download from https://ollama.com/download

# Start the daemon (keep this terminal open, or run as a service)
ollama serve

# In another terminal, pull a small fast model
ollama pull llama3.1:8b
```

The agent will auto-detect Ollama on `localhost:11434` and use it whenever
both Groq and Gemini fail or are unconfigured. Ollama runs entirely on your
machine — no rate limits, no API costs.

On Hugging Face Spaces the keys are configured under **Settings → Variables and secrets**.

---

## Example conversations

Once the app is running:

- "Compare the free-tier daily request limits of Groq, Gemini, and Cerebras."
- "Who founded Sarvam AI and when?"
- "Is intermittent fasting safe for people with type 2 diabetes?" *(expect explicit conflict acknowledgement)*
- "What is the current home address of Sarvam AI's CEO?" *(expect a refusal)*
- "What is the latest open-weights Llama release?"

Each answer shows an expandable **Audit panel** with the plan, every search query, every
opened URL, every selected snippet, and the per-turn latency.

---

## Evaluation harness

The eval harness is in `eval/`.

### Dataset

`eval/dataset.json` — ~17 hand-curated questions across:

- **factual** — single fact lookup
- **multi_hop** — combines facts from multiple sources
- **comparison** — compare two entities
- **recency** — needs the most recent information
- **conflicting_sources** — must acknowledge disagreement
- **insufficient_evidence** — must refuse
- **multi_turn** — three-turn conversation testing session memory

### Metrics

| Metric | Type | Notes |
|---|---|---|
| `citation_coverage` | automatic | % of factual sentences with at least one `[S#]` |
| `citation_validity` | automatic | % of `[S#]` ids that map to a real fetched snippet |
| `source_diversity` | automatic | unique cited domains ÷ total citations |
| `faithfulness` | LLM judge | claims supported by cited snippets / total claims |
| `relevance` | LLM judge | does the answer address the user's question |
| `conflict_handled` | automatic + judge | for conflict-class items |
| `appropriate_refusal` | automatic + judge | for insufficient-evidence items |
| `latency_ms` | automatic | wall-clock per turn (mean + p95) |

### Running

```bash
python -m eval.run_eval --skip-judge      # auto-only, ~3–5 min
python -m eval.run_eval                   # full, including LLM-as-judge
```

`eval/report.md` is the human-readable summary. `eval/results.json` keeps the full
per-item transcript (search queries, opened URLs, selected snippets, raw answer with
`[S#]`, expanded answer, judge JSON, latency).

---

## Limitations & future improvements

**Limitations**

- **JS-rendered pages** (single-page apps) often yield little text via static fetch. We
  do not run a headless browser by default.
- **Paywalls and login-walled sites** can return useless extracted content. We rely on
  Tavily's filtering and on MMR to push us toward more accessible domains.
- **No persistent vector index.** Each turn re-embeds chunks. Acceptable at this scale,
  but a Chroma / SQLite-vss layer would help if we wanted to reuse pages across turns.
- **Single-region rate limits.** Heavy use can still trip Tavily's monthly cap.
- **Citation expansion is markdown.** Non-markdown surfaces (e.g. plain text export)
  need an additional renderer.

**Future improvements** (beyond the two highlighted in the design note)

- Invalid-citation re-prompting (described above).
- Self-critic verification pass on key claims.
- Headless-browser fallback (Playwright) when static fetch returns < N characters.
- Persistent cross-session retrieval cache so popular topics are answered faster.
- A `claim-level provenance` view showing every sentence and its supporting snippet text.

---

## Deployment — Hugging Face Spaces

We chose Hugging Face Spaces for its 16 GB RAM (vs Streamlit Cloud's 690 MB–2.7 GB),
built-in secrets, and 48-hour idle window.

### One-time setup

1. Sign in at https://huggingface.co (free).
2. Click your avatar → **New Space**.
3. Fill in:
   - **Space name**: `deep-research-agent` (or whatever you like)
   - **License**: MIT
   - **Select the Space SDK**: **Streamlit**
   - **Hardware**: CPU basic (free)
   - **Public** or **Private** — your call.
4. Click **Create Space**.

### Push the code

Hugging Face Spaces is a git repository. Two options:

**Option A — via Hugging Face CLI (recommended)**

```bash
pip install -U "huggingface_hub[cli]"
hf auth login                                    # paste a write token from huggingface.co/settings/tokens

# from inside the SarvamAi/ folder
git init -b main
git add . && git commit -m "Initial deep research agent"
git remote add space https://huggingface.co/spaces/<your-username>/deep-research-agent
git push space main
```

**Option B — via the web UI**

Drag-and-drop every file in `SarvamAi/` into the Space's **Files** tab via the
"Add file" button. Make sure `app.py` and `requirements.txt` are at the top level.

### Add your secrets

In the Space, go to **Settings → Variables and secrets → New secret** and add:

| Name | Value |
|---|---|
| `TAVILY_API_KEY` | your `tvly-…` key |
| `GROQ_API_KEY` | your `gsk_…` key |
| `GEMINI_API_KEY` | (optional) your Gemini key |

`OLLAMA_URL` is **not** used on Spaces (the local Ollama daemon won't be reachable
from a hosted container). The Groq → Gemini fallback chain is what matters in
production.

### First boot

After saving the secrets, the Space rebuilds automatically. The first build takes
2–3 minutes (downloads `fastembed`'s ~130 MB BGE-small model on first request).
After that the app is live; subsequent cold starts are ~30 seconds.

> Sessions and the SQLite file live on the Space's ephemeral disk. They persist
> while the Space is warm but reset across hard restarts. Acceptable for a demo;
> a small Postgres add-on or HF Persistent Storage would make them durable.

---

## Assumptions

- We assume the reviewer has free-tier accounts on Tavily and Groq (both are no-card signups).
- We assume English-language web sources; the agent will still work on other languages but the embedder is English-centric.
- We deliberately limit `MAX_PAGES_TO_FETCH` to 6 to stay well within Tavily's free monthly credit budget across an eval run + interactive demos.
- The LLM-as-judge is the same Groq model used for answering. This is a known caveat in RAG evaluation; we mitigate by also reporting automatic metrics that do not depend on the judge.
