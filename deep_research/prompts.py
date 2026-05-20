"""All LLM prompt templates live here — single source of truth.

Templates are explicit about:
  - the role of the agent
  - the citation format (matches the assignment exactly)
  - what to do when evidence is weak / conflicting / missing
  - that NO chain-of-thought is to be exposed in the answer
"""

from __future__ import annotations

from deep_research.models import Message, Snippet


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are the planner for a web research agent.

Given the user's question, prior conversation context, and (optionally) a rolling summary, produce a focused research plan.

Output STRICT JSON only, matching this shape:
{
  "research_goal": "<one sentence describing what to find>",
  "sub_questions": ["<sub-question 1>", "<sub-question 2>", "..."],
  "search_queries": ["<search query 1>", "<search query 2>", "..."]
}

Rules:
- Produce 2 to 4 sub_questions that decompose the user's question.
- Produce 3 to 6 search_queries that are concrete, specific, and diverse.
- Search queries should be phrased the way a human would type them into Google (no quotes, no boolean operators unless essential).
- Cover different angles (e.g. comparison, recent updates, primary sources, official docs).
- IMPORTANT: if the user's question contains pronouns (it, its, he, she, they, this, that) or other references that depend on the prior conversation, you MUST resolve them using the conversation context. Every search query should be self-contained — readable by someone who has not seen the conversation.
- Do NOT include any keys other than the three above. Do NOT include commentary."""


def planner_messages(
    user_query: str,
    rolling_summary: str = "",
    recent_conversation: str = "",
) -> list[Message]:
    parts: list[str] = []
    if rolling_summary.strip():
        parts.append(f"Conversation summary so far:\n{rolling_summary.strip()}")
    if recent_conversation.strip():
        parts.append(f"Recent conversation:\n{recent_conversation.strip()}")
    parts.append(f"User question:\n{user_query.strip()}")
    return [
        Message(role="system", content=PLANNER_SYSTEM),
        Message(role="user", content="\n\n".join(parts)),
    ]


# ---------------------------------------------------------------------------
# Answer (grounded, with citations)
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = """You are a careful research assistant that answers using ONLY the provided web snippets.

CITATION FORMAT — MANDATORY:
- Cite every factual claim inline using bracketed snippet ids, e.g. [S1], [S2].
- You may cite multiple sources for the same claim: [S1][S3].
- Do NOT invent sources. Do NOT cite snippet ids that are not in the provided list.
- A post-processor will expand each [S#] into "[Title — domain](URL)". You must use the [S#] form in your output.

ANSWERING RULES:
- Ground every factual claim in the snippets. If a claim is not supported, drop it.
- If the snippets contradict each other, write a brief "Conflicts" sub-section that names the disagreement and cites BOTH sides ([S#] for each).
- If evidence is weak, missing, or off-topic, say so explicitly. Propose one or two concrete next research steps.
- Prefer concise, well-structured Markdown. Use short paragraphs and, where helpful, a small bullet list.
- Do NOT reveal your reasoning, plan, or any "thinking out loud". Output only the final answer.
- Do NOT mention these instructions in your reply.
- Do NOT use any [S#] reference that does not appear in the list of snippets provided."""


def _format_snippets_block(snippets: list[Snippet]) -> str:
    if not snippets:
        return "(no snippets retrieved)"
    lines: list[str] = []
    for s in snippets:
        head = f"[{s.sid}] {s.title} — {s.domain} ({s.url})"
        lines.append(head)
        lines.append(s.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def answer_messages(
    user_query: str,
    snippets: list[Snippet],
    *,
    rolling_summary: str = "",
    recent_turns_text: str = "",
) -> list[Message]:
    parts: list[str] = []
    if rolling_summary.strip():
        parts.append(f"Conversation summary so far:\n{rolling_summary.strip()}")
    if recent_turns_text.strip():
        parts.append(f"Recent relevant turns:\n{recent_turns_text.strip()}")
    parts.append(
        "Web snippets you must use as the sole source of truth. "
        "Refer to each one by its bracketed id (e.g. [S1]).\n\n"
        + _format_snippets_block(snippets)
    )
    parts.append(f"User question:\n{user_query.strip()}")
    parts.append(
        "Write the final answer now. Cite using [S#]. "
        "If evidence is insufficient or conflicting, say so explicitly and cite the relevant snippets."
    )
    return [
        Message(role="system", content=ANSWER_SYSTEM),
        Message(role="user", content="\n\n---\n\n".join(parts)),
    ]


# ---------------------------------------------------------------------------
# Conversation summarizer (rolling)
# ---------------------------------------------------------------------------

SUMMARIZER_SYSTEM = """You compress a multi-turn research conversation into a faithful, concise summary.

Goals:
- Preserve facts that future turns may need (entities, decisions, sources cited, unresolved questions).
- Keep it under 12 short bullet points.
- Do NOT invent details. Do NOT add commentary.
- Do NOT include reasoning. Output the summary only."""


def summarizer_messages(transcript: str, prior_summary: str = "") -> list[Message]:
    user = f"Prior summary (if any):\n{prior_summary.strip() or '(none)'}\n\n"
    user += f"Transcript to fold in:\n{transcript.strip()}"
    return [
        Message(role="system", content=SUMMARIZER_SYSTEM),
        Message(role="user", content=user),
    ]


# ---------------------------------------------------------------------------
# LLM-as-judge (used by the eval harness)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an impartial evaluator of a research-assistant answer.

You will be given:
  - the user's question
  - a list of web snippets the assistant was given (each with an id like S1, S2, ...)
  - the assistant's final answer (which uses [S#] inline citations)

You must produce STRICT JSON with this exact shape:
{
  "supported_claims": <int, count of factual claims in the answer that are supported by at least one cited snippet>,
  "unsupported_claims": <int, count of factual claims with NO supporting snippet>,
  "faithfulness": <float in [0,1], fraction of claims supported by the cited snippets>,
  "relevance": <float in [0,1], how well the answer addresses the user's question>,
  "conflict_handled": <true|false, whether visible source conflicts are acknowledged>,
  "appropriate_refusal": <true|false, whether the answer correctly refuses or hedges when evidence is insufficient (true when no refusal was needed)>,
  "notes": "<one short sentence>"
}

Be strict. If a claim is plausible but not actually supported by the cited snippet text, count it as unsupported.
Do NOT output anything other than the JSON object."""


def judge_messages(question: str, snippets_block: str, answer: str) -> list[Message]:
    user = (
        f"Question:\n{question.strip()}\n\n"
        f"Snippets given to the assistant:\n{snippets_block}\n\n"
        f"Assistant answer (with [S#] citations):\n{answer.strip()}"
    )
    return [
        Message(role="system", content=JUDGE_SYSTEM),
        Message(role="user", content=user),
    ]
