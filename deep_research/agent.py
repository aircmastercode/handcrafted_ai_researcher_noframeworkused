"""Hand-built agent orchestrator (no framework).

`DeepResearchAgent.run(...)` is an async generator that drives the entire
research turn and yields `ProgressEvent` objects after every phase. The UI
consumes these to render live status + a streaming answer.

Pipeline:
    PLAN  →  SEARCH  →  FETCH  →  SELECT CONTEXT  →  ANSWER  →  PERSIST
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator, Optional

from deep_research.citations import (
    CitationReport,
    citation_coverage,
    expand_and_validate,
)
from deep_research.context import select_context
from deep_research.fetch import fetch_pages
from deep_research.llm import LLMClient, LLMError
from deep_research.models import (
    FetchedPage,
    Message,
    Plan,
    SearchResult,
    Snippet,
    Turn,
    now_iso,
)
from deep_research.progress import Phase, ProgressEvent, evt
from deep_research.prompts import answer_messages, planner_messages
from deep_research.search import TavilySearch
from deep_research.session import SessionStore
from deep_research.summarizer import (
    recent_turns_text,
    summarize_if_needed,
)

log = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def _word_stream(text: str):
    """Yield word-sized chunks (with trailing space) for a typewriter-feel UI."""
    parts = text.split(" ")
    for i, w in enumerate(parts):
        yield (w if i == 0 else " " + w)


def _humanize_llm_error(err: str) -> str:
    """Turn raw provider error strings into a friendly message for the UI."""
    msg = err.lower()
    if "rate_limit_exceeded" in msg or "rate limit" in msg or "429" in msg:
        if "tokens per minute" in msg or "tpm" in msg:
            return (
                "The LLM hit its per-minute token limit on the free tier. "
                "Wait ~60 seconds and try again, or set GEMINI_API_KEY as a fallback "
                "(see .env.example)."
            )
        return (
            "The LLM provider rate-limited us. Wait a moment and try again, "
            "or configure a fallback LLM (Gemini / Ollama)."
        )
    if "context length" in msg or "too large" in msg or " 413" in msg or "413," in msg:
        return (
            "The request was too large for the free-tier LLM. We've already tried to "
            "shrink the context — try a more specific question or set MAX_CONTEXT_TOKENS lower."
        )
    if "unauthorized" in msg or "invalid api key" in msg or "401" in msg:
        return "The LLM API key was rejected. Check that GROQ_API_KEY (or GEMINI_API_KEY) is correct."
    return f"LLM provider error: {err[:300]}"


class DeepResearchAgent:
    """End-to-end agent: orchestrates plan → search → fetch → select → answer."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        search: Optional[TavilySearch] = None,
        store: Optional[SessionStore] = None,
        *,
        max_search_results: Optional[int] = None,
        max_pages_to_fetch: Optional[int] = None,
        max_snippets: int = 6,
        max_context_tokens: Optional[int] = None,
        fetch_concurrency: int = 6,
    ) -> None:
        self.llm = llm or LLMClient()
        self.search = search or TavilySearch()
        self.store = store or SessionStore()

        self.max_search_results = max_search_results or _env_int("MAX_SEARCH_RESULTS", 6)
        self.max_pages_to_fetch = max_pages_to_fetch or _env_int("MAX_PAGES_TO_FETCH", 5)
        self.max_snippets = max_snippets
        self.max_context_tokens = max_context_tokens or _env_int("MAX_CONTEXT_TOKENS", 2500)
        self.fetch_concurrency = fetch_concurrency

    # ------------------------------------------------------------------ public API

    async def run(self, session_id: str, query: str) -> AsyncIterator[ProgressEvent]:
        """Drive a single research turn. Yields progress events."""
        t0 = time.perf_counter()
        query = (query or "").strip()
        if not query:
            yield evt(Phase.ERROR, "Empty query.")
            return

        # Make sure the session exists (idempotent)
        self.store.create_session(session_id)
        rolling_summary = self.store.get_rolling_summary(session_id)
        prior_messages = self.store.get_messages(session_id)
        recent_text = recent_turns_text(prior_messages)

        # --- Phase 1: PLAN ---------------------------------------------
        yield evt(Phase.PLAN_START, "Planning research strategy")
        try:
            plan = await self._plan(query, rolling_summary, recent_text)
        except Exception as e:  # noqa: BLE001
            log.exception("Planner failed; falling back to a trivial plan")
            plan = Plan(
                research_goal=query,
                sub_questions=[query],
                search_queries=[query],
            )
            yield evt(Phase.PLAN_DONE, f"Planner fallback used ({e!s})", plan=plan.model_dump())
        else:
            yield evt(Phase.PLAN_DONE, "Plan ready", plan=plan.model_dump())

        # --- Short-circuit: non-research input (greeting/chitchat/meta) -------
        if not plan.is_research:
            async for ev in self._stream_direct_response(
                session_id=session_id,
                query=query,
                plan=plan,
                t0=t0,
            ):
                yield ev
            return

        # --- Phase 2: SEARCH -------------------------------------------
        yield evt(
            Phase.SEARCH_START,
            f"Searching the web ({len(plan.search_queries)} queries)",
            queries=plan.search_queries,
        )
        try:
            results = await self.search.multi_search(
                plan.search_queries,
                per_query=max(3, self.max_search_results // 2),
                search_depth="advanced",
            )
        except Exception as e:  # noqa: BLE001
            log.exception("Search failed")
            yield evt(Phase.ERROR, f"Search failed: {e!s}")
            return

        results = results[: self.max_search_results]
        yield evt(
            Phase.SEARCH_DONE,
            f"Found {len(results)} unique results",
            results=[r.model_dump() for r in results],
        )

        if not results:
            yield evt(Phase.ERROR, "No web results found for this query.")
            return

        # --- Phase 3: FETCH --------------------------------------------
        urls_to_fetch = [r.url for r in results[: self.max_pages_to_fetch]]
        yield evt(
            Phase.FETCH_START,
            f"Fetching {len(urls_to_fetch)} pages",
            urls=urls_to_fetch,
        )

        progress_queue: asyncio.Queue[tuple[int, int, FetchedPage]] = asyncio.Queue()

        def _on_progress(done: int, total: int, page: FetchedPage) -> None:
            progress_queue.put_nowait((done, total, page))

        fetch_task = asyncio.create_task(
            fetch_pages(
                urls_to_fetch,
                concurrency=self.fetch_concurrency,
                on_progress=_on_progress,
            )
        )

        # Drain progress events while the fetch task runs.
        while not fetch_task.done():
            try:
                done, total, page = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
                yield evt(
                    Phase.FETCH_PROGRESS,
                    f"Fetched {done}/{total}: {page.domain or page.url}",
                    done=done,
                    total=total,
                    url=page.url,
                    ok=page.ok,
                    error=page.error,
                )
            except asyncio.TimeoutError:
                continue
        # Drain any leftover events.
        while not progress_queue.empty():
            done, total, page = progress_queue.get_nowait()
            yield evt(
                Phase.FETCH_PROGRESS,
                f"Fetched {done}/{total}: {page.domain or page.url}",
                done=done,
                total=total,
                url=page.url,
                ok=page.ok,
                error=page.error,
            )

        pages: list[FetchedPage] = await fetch_task
        n_ok = sum(1 for p in pages if p.ok)
        yield evt(
            Phase.FETCH_DONE,
            f"Fetched {n_ok}/{len(pages)} pages successfully",
            urls=[p.url for p in pages],
            ok_count=n_ok,
        )

        # --- Phase 4: SELECT CONTEXT ------------------------------------
        yield evt(Phase.SELECT_START, "Selecting relevant context")
        snippets = await asyncio.to_thread(
            select_context,
            query,
            pages,
            max_snippets=self.max_snippets,
            max_tokens=self.max_context_tokens,
        )
        domains_used = sorted({s.domain for s in snippets if s.domain})
        yield evt(
            Phase.SELECT_DONE,
            f"Selected {len(snippets)} snippets from {len(domains_used)} domains",
            snippets=[s.model_dump() for s in snippets],
            domains=domains_used,
        )

        if not snippets:
            yield evt(
                Phase.ERROR,
                "Could not extract usable content from any of the fetched pages.",
            )
            return

        # --- Phase 5: ANSWER (streaming) -------------------------------
        yield evt(Phase.ANSWER_START, "Generating grounded answer")
        prompt_messages = answer_messages(
            user_query=query,
            snippets=snippets,
            rolling_summary=rolling_summary,
            recent_turns_text=recent_text,
        )

        raw_chunks: list[str] = []
        try:
            async for delta in self.llm.chat_stream(
                prompt_messages, temperature=0.2, max_tokens=1200
            ):
                raw_chunks.append(delta)
                yield evt(Phase.ANSWER_TOKEN, delta=delta)
        except LLMError as e:
            log.exception("LLM stream failed")
            yield evt(Phase.ERROR, _humanize_llm_error(str(e)))
            return

        raw_answer = "".join(raw_chunks).strip()
        coverage = citation_coverage(raw_answer)
        final_answer, report = expand_and_validate(raw_answer, snippets)
        yield evt(
            Phase.ANSWER_DONE,
            "Answer complete",
            raw_answer=raw_answer,
            final_answer=final_answer,
            citation_coverage=coverage,
            cited_sids=report.unique_sids,
            invalid_sids=report.invalid_sids,
            unique_domains=report.unique_domains,
        )

        # --- Phase 6: PERSIST ------------------------------------------
        latency_ms = int((time.perf_counter() - t0) * 1000)
        turn = Turn(
            query=query,
            plan=plan,
            search_queries=plan.search_queries,
            urls_opened=[p.url for p in pages if p.ok],
            snippets=snippets,
            final_answer=final_answer,
            ts=now_iso(),
            latency_ms=latency_ms,
        )
        self.store.append_message(session_id, Message(role="user", content=query))
        self.store.append_message(session_id, Message(role="assistant", content=final_answer))
        self.store.append_turn(session_id, turn)

        # --- Phase 7: SUMMARIZE (only if conversation grew long) -------
        try:
            new_summary = await summarize_if_needed(
                self.llm,
                self.store.get_messages(session_id),
                prior_summary=rolling_summary,
            )
            if new_summary:
                self.store.set_rolling_summary(session_id, new_summary)
        except Exception:  # noqa: BLE001
            log.exception("Background summarization failed (non-fatal)")

        # --- Done ------------------------------------------------------
        yield evt(
            Phase.DONE,
            f"Done in {latency_ms / 1000:.1f}s",
            latency_ms=latency_ms,
            final_answer=final_answer,
            n_snippets=len(snippets),
            n_domains=len(domains_used),
            citation_coverage=coverage,
            invalid_sids=report.invalid_sids,
        )

    # ------------------------------------------------------------------ helpers

    async def _plan(
        self,
        query: str,
        rolling_summary: str,
        recent_conversation: str = "",
    ) -> Plan:
        msgs = planner_messages(
            query,
            rolling_summary=rolling_summary,
            recent_conversation=recent_conversation,
        )
        data = await self.llm.chat_json(msgs, temperature=0.1, max_tokens=600)
        try:
            is_research = bool(data.get("is_research", True))
            direct = str(data.get("direct_response") or "").strip()
            search_queries = [str(x) for x in (data.get("search_queries") or [])][:6]
            if is_research and not search_queries:
                # Planner said research but gave no queries — fall back to the user's text.
                search_queries = [query]
            return Plan(
                research_goal=str(data.get("research_goal") or (query if is_research else "")),
                sub_questions=[str(x) for x in (data.get("sub_questions") or [])][:6],
                search_queries=search_queries,
                is_research=is_research,
                direct_response=direct,
            )
        except Exception:  # noqa: BLE001
            return Plan(
                research_goal=query,
                sub_questions=[query],
                search_queries=[query],
                is_research=True,
            )

    async def _stream_direct_response(
        self,
        *,
        session_id: str,
        query: str,
        plan: Plan,
        t0: float,
    ) -> AsyncIterator[ProgressEvent]:
        """Emit a tiny event stream for non-research inputs (greetings, meta, chitchat)."""
        reply = (plan.direct_response or "").strip() or (
            "I am a web-research assistant. Ask me a factual question and I will search "
            "the web, read the sources, and give you a cited answer."
        )

        yield evt(Phase.ANSWER_START, "Responding")
        # Stream the canned reply word-by-word so the UI feels alive.
        for token in _word_stream(reply):
            yield evt(Phase.ANSWER_TOKEN, delta=token)
        yield evt(
            Phase.ANSWER_DONE,
            "Answer complete",
            raw_answer=reply,
            final_answer=reply,
            citation_coverage=1.0,
            cited_sids=[],
            invalid_sids=[],
            unique_domains=[],
        )

        # Persist as a normal turn so it shows up in the conversation history.
        latency_ms = int((time.perf_counter() - t0) * 1000)
        turn = Turn(
            query=query,
            plan=plan,
            search_queries=[],
            urls_opened=[],
            snippets=[],
            final_answer=reply,
            ts=now_iso(),
            latency_ms=latency_ms,
        )
        self.store.append_message(session_id, Message(role="user", content=query))
        self.store.append_message(session_id, Message(role="assistant", content=reply))
        self.store.append_turn(session_id, turn)

        yield evt(
            Phase.DONE,
            f"Done in {latency_ms / 1000:.1f}s",
            latency_ms=latency_ms,
            final_answer=reply,
            n_snippets=0,
            n_domains=0,
            citation_coverage=1.0,
            invalid_sids=[],
        )
