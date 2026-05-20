# What the Assignment is Asking — In Simple Language

> A clean, complete, no-fluff breakdown of the Sarvam AI "Deep Research Agent" challenge so we never lose track of any requirement.

---

## 1. The Big Picture (One Paragraph)

We have to build an **AI Agent** (a small program that thinks in steps) that, given a user's question, goes to the **live internet**, searches for information, **reads the actual web pages**, picks the most useful parts, and writes a clear answer with **citations** (links + titles + domains) — exactly like ChatGPT's "Deep Research" or Perplexity does. It must also **remember the conversation**, **stream live progress updates** while it's working, and come with our own **evaluation system** that proves it works well.

---

## 2. The Hard Rules (Non-Negotiables)

| Rule | What it Means |
|------|---------------|
| **Language** | Python only |
| **Time** | 2 days (we are doing it in 1 day) |
| **No frameworks** | Cannot use LangChain, LangGraph, CrewAI, LlamaIndex, Haystack, or any other agent framework. We must hand-write the agent loop. |
| **LLM** | Any free LLM is allowed (open-source, local, or free-tier APIs). |
| **Search** | Must use Tavily, Serper, or Parallel as the web search layer. |
| **Submission** | One PDF with all links and assumptions, submitted before the deadline. |

---

## 3. The Three Parts of the Assignment

### Part 1 — Design Note (25% of marks)

A short 1–2 page write-up (can be a section in the README) covering:

1. **Who is this for and what problem does it solve?**
2. **What does "deep research" mean in our implementation?** (Our own definition)
3. **3–5 success metrics** we will use to judge quality.
4. **Data flow diagram**: search → page fetch → context selection → answer.
5. **Risks and limitations** (rate limits, low-quality sources, conflicting sources, context length) + **2 future improvements**.

### Part 2 — Technical Implementation (75% of marks)

This is the actual working code. It has several modules:

#### A. Web Research

- **Search module** — Sends queries to Tavily/Serper/Parallel and returns results that include at minimum: `title`, `url`, `snippet`, and an optional `relevance_score`.
- **Page-fetch module** — Downloads the full page for selected URLs, converts HTML to clean readable text, and stores metadata: `url`, `title`, `retrieved_at`, optional `domain`.

#### B. Context Construction

- From everything we fetched, **pick the most useful snippets** to send to the LLM.
- Selection should balance **relevance + recency + source diversity**.
- **Limit total context size** (by tokens or characters) so we don't exceed the LLM's window.
- **Keep metadata attached** to every snippet so we can cite it later.

#### C. Citations (Strict Rules)

- Every fact-heavy answer must cite its source as either:
  - `[Title — domain]` with the URL included, **or**
  - `(domain, URL)`
- If **two sources disagree**, the agent must explicitly say so and cite both.

#### D. Session Management & History

- Persist everything to disk using **`session_id`** (JSON or SQLite).
- **Conversation history**: every user message + assistant message with a timestamp.
- **Turn history**: for every user query, save:
  - The query itself
  - The search queries we issued
  - The URLs we opened
  - The context snippets we picked
  - The final answer
  - Timestamp

#### E. Context Builder (the brain that picks what to send to the LLM)

For each LLM call, decide what to include from:
- The **current user query**
- A **rolling summary** of older conversation (when history gets long)
- The **most relevant prior turns**
- The **selected web snippets** with their metadata
- A **hard maximum context length** with **summarization fallback** when we exceed it.

#### F. Agent Flow (Hand-Built, No Frameworks)

For each user query the agent must do these four steps:

1. **Plan** — Write a short plan of what to research and which search queries to issue.
2. **Search** — Use Tavily/Serper/Parallel to get web results.
3. **Acquire & Select Context** — Fetch pages, pull the most relevant snippets.
4. **Answer** — Generate a final response that is grounded in the selected snippets and properly cited. If evidence is weak/missing/conflicting, **explicitly say so** and propose next steps.

#### G. Streaming Progress Updates

While the agent works, the UI must show live updates like:
- "Planning…"
- "Searching the web…"
- "Fetching sources…"
- "Selecting relevant context…"
- "Generating answer with citations…"

This can be done via Streamlit updates, CLI prints, or FastAPI streaming.

**Important constraint:** Do **NOT** stream the LLM's hidden chain-of-thought. Only stream operational status.

### Part 3 — Evaluation Harness (Required)

A runnable script that proves our agent works:

1. **Build a dataset** of questions that test the agent across these categories:
   - Factual (single-fact lookup)
   - Multi-hop (needs combining info from multiple sources)
   - Comparison (compare two things)
   - Insufficient evidence (agent should admit "I don't know")
   - Conflicting sources (agent should flag the disagreement)
2. **Define our own evaluation metrics** that measure:
   - Grounding & citation quality
   - Correctness & usefulness
   - Handling of uncertainty & conflicts
   - Robustness across turns and sessions
3. **Write a script** that:
   - Runs the agent on every question in the dataset
   - Records outputs, citations, and any intermediate artifacts
   - Produces a clear summary of results

> The assignment explicitly says: *"There is no prescribed metric set — part of the assignment is demonstrating good judgment in choosing what to measure and why."* So our choice of metrics itself is being graded.

---

## 4. Deliverables Checklist (What to Hand In)

- [ ] Working app — Streamlit / Gradio / custom UI
- [ ] Web research ingestion via Tavily / Serper / Parallel
- [ ] Persistent sessions with conversation + turn history
- [ ] Citation-grounded answers with URL / title / domain
- [ ] Streaming intermediate step updates in the UI
- [ ] Evaluation harness with metrics and results
- [ ] **README** containing:
  - Video demo
  - Setup & run instructions
  - Design note (Part 1)
  - Example conversations
  - Evaluation methodology & findings
  - Limitations & future improvements
- [ ] **One PDF** with all submission links and assumptions

---

## 5. How We Will Be Judged (Evaluation Criteria)

The reviewers will score us on:

1. **Soundness of the evaluation metrics we chose** and our rationale for them.
2. **Citation integrity** — do citations actually point to real web sources that support the claim?
3. **Quality of context selection** and how we handle conflicting sources.
4. **Session & context management** across the agent's full flow.
5. **Code quality** — modularity, error handling, cleanliness.

---

## 6. The Trap Cards (Things That Are Easy to Miss)

These are subtle requirements that look small but are explicitly scored:

1. **Source diversity** — don't pick 5 snippets from the same domain. We need variety.
2. **Recency awareness** — if a question is about something recent, prefer newer pages.
3. **Conflict handling** — must *explicitly* mention disagreements, not silently pick one side.
4. **Insufficient evidence** — must be willing to say "I cannot find enough information" rather than make things up.
5. **Rolling summary** — when history grows long, summarize old turns instead of dumping the whole transcript into the prompt.
6. **No streamed chain-of-thought** — only operational status messages.
7. **Timestamps** — every message and turn must have one.
8. **Atomic metadata** — every snippet/source must remain linked to its `url`, `title`, `domain`, and `retrieved_at` from start to finish, so citations are never fake.
9. **Free LLM** — must not silently require a paid OpenAI/Anthropic key to run.
10. **Re-runnable evaluation** — the evaluation script must be executable end-to-end by the reviewer.

---

## 7. One-Sentence Summary

> Build a Python web-research AI agent (from scratch, no frameworks) that searches the live web with Tavily, reads pages, picks the best snippets, answers with real citations, remembers the conversation, streams live progress, and ships with our own evaluation harness — all running on a free LLM and hosted for free.
