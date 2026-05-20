"""Context selection: semantic re-rank + MMR diversity + token budget.

Given the user query and a batch of fetched pages, we produce a ranked,
diverse, budget-respecting list of `Snippet` objects with stable ids (S1, S2, …)
ready to feed the LLM.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from deep_research.fetch import chunk_text
from deep_research.llm import count_tokens
from deep_research.models import FetchedPage, Snippet

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedder (cached singleton; fastembed CPU/ONNX)
# ---------------------------------------------------------------------------


class _Embedder:
    """Lazy-loaded fastembed singleton with a word-overlap fallback."""

    _instance: Optional["_Embedder"] = None

    def __init__(self) -> None:
        self._fe = None
        self._dim: Optional[int] = None
        self._init_fastembed()

    def _init_fastembed(self) -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore

            # BAAI/bge-small-en-v1.5 — small, fast, retrieval-tuned. 384 dims.
            self._fe = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self._dim = 384
            log.info("fastembed BGE-small loaded")
        except Exception as e:  # noqa: BLE001
            log.warning("fastembed unavailable (%s); falling back to bag-of-words", e)
            self._fe = None

    @classmethod
    def get(cls) -> "_Embedder":
        if cls._instance is None:
            cls._instance = _Embedder()
        return cls._instance

    @property
    def ok(self) -> bool:
        return self._fe is not None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._fe is not None:
            return [list(v) for v in self._fe.embed(texts)]
        # Fallback: cheap normalized bag-of-words vector via a hashing trick.
        return [_bow_vector(t) for t in texts]


def _bow_vector(text: str, dim: int = 256) -> list[float]:
    """Hashed-bag-of-words vector. Crude but lets the pipeline still run offline."""
    vec = [0.0] * dim
    for tok in _tokenize_words(text):
        h = hash(tok) % dim
        vec[h] += 1.0
    return _l2_normalize(vec)


def _tokenize_words(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _l2_normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    # fastembed vectors are already unit-norm; bow we normalize ourselves.
    return float(s)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_context(
    query: str,
    pages: list[FetchedPage],
    *,
    max_snippets: int = 8,
    max_tokens: int = 4000,
    mmr_lambda: float = 0.65,
    target_words: int = 220,
    overlap_words: int = 40,
) -> list[Snippet]:
    """Return budget-respecting, diverse, semantically-ranked snippets with stable ids."""
    if not pages:
        return []

    # 1) Chunk each fetched page into windows
    candidates: list[dict] = []
    for p in pages:
        if not p.ok or not p.text:
            continue
        for chunk in chunk_text(p.text, target_words=target_words, overlap_words=overlap_words):
            candidates.append(
                {
                    "text": chunk,
                    "url": p.url,
                    "title": p.title,
                    "domain": p.domain,
                    "retrieved_at": p.retrieved_at,
                }
            )

    if not candidates:
        return []

    # 2) Embed query and chunks
    emb = _Embedder.get()
    q_vec = emb.embed([query])[0]
    doc_vecs = emb.embed([c["text"] for c in candidates])

    # 3) Initial relevance scores
    sims = [_cosine(q_vec, d) for d in doc_vecs]

    # 4) MMR with a mild domain-level diversity bonus
    chosen_idx = _mmr_with_domain_diversity(
        candidates=candidates,
        sims=sims,
        doc_vecs=doc_vecs,
        k=max(max_snippets * 2, max_snippets),  # over-select; budget will trim
        lambda_=mmr_lambda,
    )

    # 5) Enforce token budget and finalize Snippet objects
    snippets: list[Snippet] = []
    used_tokens = 0
    sid_counter = 0
    for i in chosen_idx:
        if len(snippets) >= max_snippets:
            break
        c = candidates[i]
        tok = count_tokens(c["text"])
        if used_tokens + tok > max_tokens and snippets:
            continue  # try a smaller next candidate rather than truncating
        sid_counter += 1
        snippets.append(
            Snippet(
                sid=f"S{sid_counter}",
                text=c["text"],
                url=c["url"],
                title=c["title"],
                domain=c["domain"],
                retrieved_at=c["retrieved_at"],
                score=float(sims[i]),
            )
        )
        used_tokens += tok

    return snippets


def _mmr_with_domain_diversity(
    *,
    candidates: list[dict],
    sims: list[float],
    doc_vecs: list[list[float]],
    k: int,
    lambda_: float,
) -> list[int]:
    """MMR selection with a small extra penalty for repeated domains."""
    n = len(candidates)
    if n == 0:
        return []
    remaining = set(range(n))
    selected: list[int] = []
    domain_counts: dict[str, int] = {}

    # First pick = highest similarity
    first = max(remaining, key=lambda i: sims[i])
    selected.append(first)
    remaining.discard(first)
    domain_counts[candidates[first]["domain"]] = 1

    while len(selected) < min(k, n) and remaining:
        best_i = None
        best_score = -1e18
        for i in remaining:
            sim_q = sims[i]
            sim_max_selected = max(_cosine(doc_vecs[i], doc_vecs[j]) for j in selected)
            domain_penalty = 0.1 * domain_counts.get(candidates[i]["domain"], 0)
            score = (lambda_ * sim_q) - ((1 - lambda_) * sim_max_selected) - domain_penalty
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.discard(best_i)
        d = candidates[best_i]["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1

    return selected
