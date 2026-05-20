"""Rolling conversation summarizer.

When the cumulative message tokens grow beyond a threshold, we fold older
turns into a compact summary so we never blow the LLM context window.
"""

from __future__ import annotations

import logging
from typing import Optional

from deep_research.llm import LLMClient, count_tokens
from deep_research.models import Message
from deep_research.prompts import summarizer_messages

log = logging.getLogger(__name__)


DEFAULT_TRIGGER_TOKENS = 3000
DEFAULT_RECENT_KEEP = 2  # keep last N (user, assistant) pairs verbatim


def messages_token_count(messages: list[Message]) -> int:
    return sum(count_tokens(m.content) for m in messages)


def needs_summarization(messages: list[Message], threshold: int = DEFAULT_TRIGGER_TOKENS) -> bool:
    return messages_token_count(messages) > threshold


def _transcript(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        role = "User" if m.role == "user" else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


async def summarize_if_needed(
    llm: LLMClient,
    messages: list[Message],
    prior_summary: str = "",
    *,
    trigger_tokens: int = DEFAULT_TRIGGER_TOKENS,
    keep_recent: int = DEFAULT_RECENT_KEEP,
) -> Optional[str]:
    """Return a new rolling summary if summarization was triggered, else None."""
    if not needs_summarization(messages, trigger_tokens):
        return None

    # Keep the last N pairs (2N messages) verbatim; summarize the older head.
    head_end = max(0, len(messages) - 2 * keep_recent)
    head = messages[:head_end]
    if not head:
        return None

    transcript = _transcript(head)
    msgs = summarizer_messages(transcript=transcript, prior_summary=prior_summary)
    try:
        new_summary = await llm.chat(msgs, temperature=0.1, max_tokens=600)
    except Exception:  # noqa: BLE001
        log.exception("Summarizer failed; keeping previous summary")
        return None
    return new_summary.strip()


def recent_turns_text(messages: list[Message], pairs: int = DEFAULT_RECENT_KEEP) -> str:
    """Render the last N (user, assistant) pairs for inclusion in the answer prompt."""
    if not messages:
        return ""
    tail = messages[-2 * pairs :]
    return _transcript(tail)
