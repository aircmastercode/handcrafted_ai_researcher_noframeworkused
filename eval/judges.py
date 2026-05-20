"""LLM-as-judge wrapper used by the evaluation harness."""

from __future__ import annotations

import logging
from typing import Any

from deep_research.llm import LLMClient
from deep_research.models import Snippet
from deep_research.prompts import _format_snippets_block, judge_messages  # type: ignore

log = logging.getLogger(__name__)


async def judge_answer(
    llm: LLMClient,
    *,
    question: str,
    snippets: list[Snippet],
    answer_with_placeholders: str,
) -> dict[str, Any]:
    """Ask the judge LLM to score one answer. Returns a dict with the JSON fields."""
    snippets_block = _format_snippets_block(snippets)
    msgs = judge_messages(
        question=question, snippets_block=snippets_block, answer=answer_with_placeholders
    )
    try:
        data = await llm.chat_json(msgs, temperature=0.0, max_tokens=400)
    except Exception as e:  # noqa: BLE001
        log.warning("Judge LLM call failed: %s", e)
        return {
            "supported_claims": None,
            "unsupported_claims": None,
            "faithfulness": None,
            "relevance": None,
            "conflict_handled": None,
            "appropriate_refusal": None,
            "notes": f"judge_error: {e!s}",
        }

    def _f(v: Any, lo: float, hi: float) -> float | None:
        try:
            x = float(v)
        except Exception:
            return None
        return max(lo, min(hi, x))

    return {
        "supported_claims": data.get("supported_claims"),
        "unsupported_claims": data.get("unsupported_claims"),
        "faithfulness": _f(data.get("faithfulness"), 0.0, 1.0),
        "relevance": _f(data.get("relevance"), 0.0, 1.0),
        "conflict_handled": bool(data.get("conflict_handled")) if data.get("conflict_handled") is not None else None,
        "appropriate_refusal": bool(data.get("appropriate_refusal")) if data.get("appropriate_refusal") is not None else None,
        "notes": str(data.get("notes", "")),
    }
