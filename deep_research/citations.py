"""Citation expansion + validation.

The LLM emits inline placeholders like [S1], [S2]. This module:
  - expands every recognised placeholder into a markdown link
    `[Title — domain](URL)` (the format the assignment requires);
  - strips any [S#] that does not map to a real snippet;
  - returns audit data (which sids were referenced, which were missing).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from deep_research.models import Snippet


_CITE_RE = re.compile(r"\[\s*[sS](\d+)\s*\]")


@dataclass
class CitationReport:
    """Audit data returned alongside the expanded answer."""

    cited_sids: list[str]
    unique_sids: list[str]
    invalid_sids: list[str]
    unique_domains: list[str]


def expand_and_validate(
    answer: str,
    snippets: Iterable[Snippet],
) -> tuple[str, CitationReport]:
    """Expand `[S#]` placeholders inline and build a sources appendix.

    Returns: (final_markdown_answer, report)
    """
    snippet_map: dict[str, Snippet] = {s.sid: s for s in snippets}

    cited_sids: list[str] = []
    invalid_sids: list[str] = []

    def _repl(m: re.Match[str]) -> str:
        sid = f"S{m.group(1)}"
        s = snippet_map.get(sid)
        if not s:
            invalid_sids.append(sid)
            return ""  # silently drop unknown refs
        cited_sids.append(sid)
        # Markdown link: [Title — domain](URL)
        title = (s.title or s.domain or s.url).strip().replace("[", "(").replace("]", ")")
        return f"[{title} — {s.domain}]({s.url})"

    body = _CITE_RE.sub(_repl, answer)

    # Build a "Sources" appendix with the snippets that were actually used.
    unique_sids = list(dict.fromkeys(cited_sids))
    appendix_lines: list[str] = []
    if unique_sids:
        appendix_lines.append("\n\n---\n\n**Sources**")
        for i, sid in enumerate(unique_sids, start=1):
            s = snippet_map[sid]
            title = (s.title or s.domain or s.url).strip()
            appendix_lines.append(f"{i}. [{title} — {s.domain}]({s.url})")
    final = body.rstrip() + ("\n" + "\n".join(appendix_lines) if appendix_lines else "")

    unique_domains = list(dict.fromkeys(snippet_map[sid].domain for sid in unique_sids))
    return final, CitationReport(
        cited_sids=cited_sids,
        unique_sids=unique_sids,
        invalid_sids=list(dict.fromkeys(invalid_sids)),
        unique_domains=unique_domains,
    )


# ---------------------------------------------------------------------------
# Cheap, automatic coverage metric (used by the eval harness)
# ---------------------------------------------------------------------------


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter; good enough for coverage stats."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    parts = _SENT_RE.split(cleaned)
    return [p.strip() for p in parts if p.strip()]


def citation_coverage(answer_with_placeholders: str) -> float:
    """Fraction of non-trivial sentences that contain at least one [S#] citation.

    Note: call this on the RAW model output (before expansion) so the
    placeholders are still present.
    """
    sentences = _split_sentences(answer_with_placeholders)
    if not sentences:
        return 0.0
    factual = [s for s in sentences if _looks_factual(s)]
    if not factual:
        return 1.0  # nothing claim-like to cite
    cited = sum(1 for s in factual if _CITE_RE.search(s))
    return cited / len(factual)


_DISCOURSE_RE = re.compile(
    r"^\s*("
    r"however|so|therefore|thus|in summary|in short|in conclusion|"
    r"overall|let me know|next steps|note|caveat"
    r")[,:\s]",
    re.IGNORECASE,
)


def _looks_factual(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < 12:
        return False
    if _DISCOURSE_RE.match(s):
        return False
    # Heuristic: a sentence with no nouns/numbers is unlikely to be a factual claim.
    if not re.search(r"[A-Za-z]{4,}|\d", s):
        return False
    return True
