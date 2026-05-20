"""Typed data models used across the agent.

Every boundary between modules is a Pydantic model — bad data fails loud and early.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, HttpUrl, field_validator


def now_iso() -> str:
    """UTC timestamp in ISO-8601 with seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def domain_of(url: str) -> str:
    """Best-effort domain extraction. Strips leading 'www.'."""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """One result returned by the search provider."""

    title: str
    url: str
    snippet: str = ""
    score: float = 0.0
    domain: str = Field(default="", validate_default=True)
    query: str = Field(..., description="The query that produced this result")

    @field_validator("domain", mode="before")
    @classmethod
    def _fill_domain(cls, v, info):
        if v:
            return v
        url = info.data.get("url", "")
        return domain_of(url)


# ---------------------------------------------------------------------------
# Fetched content & snippets
# ---------------------------------------------------------------------------


class FetchedPage(BaseModel):
    """Raw page after fetch + extraction. Pre-chunking."""

    url: str
    title: str = ""
    domain: str = ""
    text: str = ""
    retrieved_at: str = Field(default_factory=now_iso)
    ok: bool = True
    error: Optional[str] = None


class Snippet(BaseModel):
    """A bounded chunk of page text, ready to feed the LLM with full provenance."""

    sid: str = Field(..., description="Stable id like 'S1', used for citations")
    text: str
    url: str
    title: str
    domain: str
    retrieved_at: str
    score: float = 0.0  # final selection score (semantic + diversity)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class Plan(BaseModel):
    """The agent's research plan for a single user query.

    For non-research inputs (greetings, chitchat, meta questions about the agent
    itself), `is_research` is False and `direct_response` carries a short reply
    that the agent emits without doing any web search.
    """

    research_goal: str
    sub_questions: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    is_research: bool = True
    direct_response: str = ""


# ---------------------------------------------------------------------------
# Conversation & turns
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """One conversation message persisted to the session."""

    role: str  # 'user' | 'assistant'
    content: str
    ts: str = Field(default_factory=now_iso)


class Turn(BaseModel):
    """One full user-query -> agent-answer cycle, with full audit trail."""

    query: str
    plan: Optional[Plan] = None
    search_queries: list[str] = Field(default_factory=list)
    urls_opened: list[str] = Field(default_factory=list)
    snippets: list[Snippet] = Field(default_factory=list)
    final_answer: str = ""
    ts: str = Field(default_factory=now_iso)
    latency_ms: int = 0


class Session(BaseModel):
    """Persistent session container."""

    session_id: str
    created_at: str = Field(default_factory=now_iso)
    rolling_summary: str = ""
    messages: list[Message] = Field(default_factory=list)
    turns: list[Turn] = Field(default_factory=list)
