"""Tavily web search wrapper (async, with retries, parallel queries, and dedup).

We use the REST API directly via httpx so the entire agent can stay fully async.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional

import httpx

from deep_research.models import SearchResult, domain_of


@dataclass
class SearchBatch:
    """Result of `multi_search`: merged results + the errors we observed."""

    results: list[SearchResult] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

log = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class SearchError(Exception):
    pass


class TavilySearch:
    """Thin async client over Tavily's /search endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        if not self.api_key:
            raise SearchError(
                "TAVILY_API_KEY is not set. Add it to your .env (see .env.example)."
            )
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------ single query

    async def search(
        self,
        query: str,
        max_results: int = 6,
        search_depth: str = "advanced",
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """Issue one Tavily search and return normalized SearchResult objects."""
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": int(max_results),
            "include_answer": False,
            "include_raw_content": False,
        }
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(TAVILY_SEARCH_URL, json=payload)
                # Permanent errors (auth, payment, forbidden) — don't retry; surface body.
                if resp.status_code in (400, 401, 402, 403, 404, 422):
                    body = (resp.text or "")[:300].replace("\n", " ").strip()
                    raise SearchError(
                        f"Tavily HTTP {resp.status_code} (no retry): {body}"
                    )
                # Transient — retry with backoff.
                if resp.status_code in (429, 500, 502, 503, 504):
                    body = (resp.text or "")[:300].replace("\n", " ").strip()
                    raise SearchError(f"Tavily HTTP {resp.status_code} (transient): {body}")
                resp.raise_for_status()
                data = resp.json()
                return _parse_tavily_response(data, query=query)
            except SearchError as e:
                last_exc = e
                # Permanent errors — abort immediately
                if "(no retry)" in str(e):
                    break
                if attempt < self.max_retries:
                    backoff = 0.6 * (2**attempt)
                    log.warning(
                        "Tavily search retry %d/%d for %r after %.1fs (%s)",
                        attempt + 1, self.max_retries, query, backoff, e,
                    )
                    await asyncio.sleep(backoff)
                else:
                    log.error("Tavily search permanently failed for %r: %s", query, e)
            except httpx.HTTPError as e:
                last_exc = e
                if attempt < self.max_retries:
                    backoff = 0.6 * (2**attempt)
                    log.warning(
                        "Tavily search retry %d/%d for %r after %.1fs (%s)",
                        attempt + 1, self.max_retries, query, backoff, e,
                    )
                    await asyncio.sleep(backoff)
                else:
                    log.error("Tavily search permanently failed for %r: %s", query, e)
        raise SearchError(f"Tavily search failed for {query!r}: {last_exc!s}")

    # ------------------------------------------------------------------ batched queries

    async def multi_search(
        self,
        queries: Iterable[str],
        per_query: int = 5,
        search_depth: str = "advanced",
        concurrency: int = 4,
    ) -> "SearchBatch":
        """Run several queries concurrently, merge, dedup by URL, keep best score per URL.

        Returns a SearchBatch with both the merged results and a list of per-query
        errors so callers can surface them to the UI rather than silently dropping them.
        """
        qs = [q.strip() for q in queries if q and q.strip()]
        if not qs:
            return SearchBatch(results=[], errors=[])

        sem = asyncio.Semaphore(concurrency)
        errors: list[tuple[str, str]] = []

        async def _one(q: str) -> list[SearchResult]:
            async with sem:
                try:
                    return await self.search(q, max_results=per_query, search_depth=search_depth)
                except Exception as e:  # noqa: BLE001
                    log.warning("multi_search: dropping query %r due to %s", q, e)
                    errors.append((q, str(e)))
                    return []

        all_results = await asyncio.gather(*(_one(q) for q in qs))

        merged: dict[str, SearchResult] = {}
        for batch in all_results:
            for r in batch:
                key = _canonical_url(r.url)
                if key in merged:
                    if r.score > merged[key].score:
                        merged[key] = r
                else:
                    merged[key] = r
        ranked = sorted(merged.values(), key=lambda r: r.score, reverse=True)
        return SearchBatch(results=ranked, errors=errors)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_tavily_response(data: dict, query: str) -> list[SearchResult]:
    out: list[SearchResult] = []
    for item in data.get("results") or []:
        url = item.get("url") or ""
        if not url:
            continue
        out.append(
            SearchResult(
                title=item.get("title") or url,
                url=url,
                snippet=item.get("content") or "",
                score=float(item.get("score") or 0.0),
                domain=domain_of(url),
                query=query,
            )
        )
    return out


def _canonical_url(url: str) -> str:
    """A loose dedup key: strip fragments + common tracking params."""
    try:
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

        p = urlparse(url)
        drop = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "ref"}
        kept = [(k, v) for (k, v) in parse_qsl(p.query) if k.lower() not in drop]
        cleaned = p._replace(fragment="", query=urlencode(kept))
        return urlunparse(cleaned).rstrip("/")
    except Exception:
        return url.rstrip("/")
