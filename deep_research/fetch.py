"""Async page fetcher + HTML-to-text extractor + chunker.

- Fetches multiple URLs concurrently with httpx.
- Extracts main-article text with Trafilatura (with a BeautifulSoup fallback).
- Splits extracted text into bounded snippets ready for re-ranking.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Iterable

import httpx

from deep_research.models import FetchedPage, domain_of, now_iso

log = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------


async def fetch_pages(
    urls: Iterable[str],
    *,
    concurrency: int = 6,
    timeout: float = 12.0,
    on_progress=None,
) -> list[FetchedPage]:
    """Fetch and extract a batch of URLs concurrently. Failures are returned as `ok=False`."""
    url_list = [u for u in urls if u]
    if not url_list:
        return []

    sem = asyncio.Semaphore(concurrency)
    results: list[FetchedPage] = [None] * len(url_list)  # type: ignore[list-item]
    done_count = 0
    lock = asyncio.Lock()

    async with httpx.AsyncClient(
        timeout=timeout,
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
        http2=False,
    ) as client:

        async def _one(i: int, url: str) -> None:
            nonlocal done_count
            async with sem:
                page = await _fetch_one(client, url)
            results[i] = page
            async with lock:
                done_count += 1
                if on_progress:
                    try:
                        on_progress(done_count, len(url_list), page)
                    except Exception:  # noqa: BLE001 (callback must not break the run)
                        log.exception("on_progress callback raised; ignoring")

        await asyncio.gather(*(_one(i, u) for i, u in enumerate(url_list)))

    return [p for p in results if p is not None]


async def _fetch_one(client: httpx.AsyncClient, url: str) -> FetchedPage:
    try:
        resp = await client.get(url)
        if resp.status_code >= 400:
            return FetchedPage(
                url=url, domain=domain_of(url),
                ok=False, error=f"HTTP {resp.status_code}",
            )
        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype and "xml" not in ctype and ctype:
            return FetchedPage(
                url=url, domain=domain_of(url),
                ok=False, error=f"Unsupported content-type: {ctype}",
            )
        html = resp.text
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        return FetchedPage(url=url, domain=domain_of(url), ok=False, error=str(e))

    text, title = await asyncio.to_thread(_extract_text_and_title, html, url)
    if not text:
        return FetchedPage(
            url=url, domain=domain_of(url),
            ok=False, error="Empty extraction",
        )
    return FetchedPage(
        url=url,
        title=title or url,
        domain=domain_of(url),
        text=text,
        retrieved_at=now_iso(),
        ok=True,
    )


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def _extract_text_and_title(html: str, url: str) -> tuple[str, str]:
    """Try Trafilatura first, then fall back to BeautifulSoup."""
    title = ""
    text = ""
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
            with_metadata=False,
        )
        if extracted:
            text = extracted
        meta = trafilatura.extract_metadata(html)
        if meta and getattr(meta, "title", None):
            title = meta.title or ""
    except Exception:  # noqa: BLE001
        log.exception("Trafilatura extraction failed for %s", url)

    if not text:
        text = _bs4_fallback_text(html)
    if not title:
        title = _bs4_title(html) or url

    return _clean(text), _clean_one_line(title)


def _bs4_fallback_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return text
    except Exception:  # noqa: BLE001
        return ""


def _bs4_title(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        t = soup.find("title")
        return t.get_text(strip=True) if t else ""
    except Exception:  # noqa: BLE001
        return ""


_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    if not text:
        return ""
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def _clean_one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    *,
    target_words: int = 220,
    overlap_words: int = 40,
) -> list[str]:
    """Split text into overlapping word-windows. Approx 220 words ≈ 280 tokens."""
    if not text:
        return []
    # Split by paragraphs first; then pack into windows by word count.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    windows: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        windows.append(" ".join(buf).strip())
        if overlap_words > 0:
            # keep tail as overlap into next window
            tail_words = " ".join(buf).split()[-overlap_words:]
            buf = [" ".join(tail_words)]
            buf_len = len(tail_words)
        else:
            buf = []
            buf_len = 0

    for p in paragraphs:
        n = len(p.split())
        if buf_len + n > target_words and buf:
            flush()
        buf.append(p)
        buf_len += n
    flush()

    # Deduplicate near-identical windows that can arise from overlap on short docs
    seen: set[str] = set()
    out: list[str] = []
    for w in windows:
        key = w[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out
