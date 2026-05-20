"""End-to-end mocked agent run.

Replaces the LLM, search, and fetch modules with deterministic stubs and
drives `DeepResearchAgent.run()` to completion. Verifies the full pipeline
including event yielding, citation expansion, and SQLite persistence.

Run with:
    python tests/test_smoke_e2e.py
    # or
    python -m tests.test_smoke_e2e
"""

import asyncio
import tempfile
import sys
from pathlib import Path
from typing import AsyncIterator

# Make the project root importable when this script is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.agent import DeepResearchAgent
from deep_research.fetch import FetchedPage
from deep_research.models import Message, Plan, SearchResult, now_iso
from deep_research.progress import Phase
from deep_research.search import SearchBatch
from deep_research.session import SessionStore


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockLLM:
    """Fake LLM client with deterministic outputs for the planner and the answer."""

    async def chat_json(self, messages, temperature=0.1, max_tokens=600, max_retries=1):
        return {
            "research_goal": "Find facts about topic Z and Y.",
            "sub_questions": ["What is Z?", "What is Y?"],
            "search_queries": ["topic Z definition", "topic Y overview"],
        }

    async def chat_stream(self, messages, temperature=0.2, max_tokens=1500) -> AsyncIterator[str]:
        # Stream a deterministic answer with valid + one invalid citation
        chunks = [
            "Topic Z is well-established [S1]. ",
            "Topic Y has multiple definitions. ",
            "Some sources disagree [S2][S3]. ",
            "An invalid ref [S99] should be silently dropped.",
        ]
        for c in chunks:
            await asyncio.sleep(0.001)
            yield c

    async def chat(self, messages, temperature=0.2, max_tokens=1200) -> str:
        out = []
        async for d in self.chat_stream(messages, temperature, max_tokens):
            out.append(d)
        return "".join(out)


class MockSearch:
    async def multi_search(self, queries, per_query=5, search_depth="advanced", concurrency=4):
        results = []
        for i, q in enumerate(queries):
            for j in range(3):
                url = f"https://example{i}-{j}.com/article"
                results.append(
                    SearchResult(
                        title=f"Article {i}-{j}",
                        url=url,
                        snippet=f"Snippet for {q}",
                        score=1.0 - 0.05 * (i + j),
                        domain=f"example{i}-{j}.com",
                        query=q,
                    )
                )
        return SearchBatch(results=results, errors=[])


def make_mock_fetch(label_prefix: str = "Article body"):
    async def _fetch_pages(urls, *, concurrency=6, timeout=12.0, on_progress=None):
        pages = []
        for i, u in enumerate(urls):
            page = FetchedPage(
                url=u,
                title=f"Title for {u}",
                domain=u.split("//", 1)[-1].split("/", 1)[0],
                text=(f"{label_prefix} {i}: details about topic Z. " * 60),
                retrieved_at=now_iso(),
                ok=True,
            )
            pages.append(page)
            if on_progress:
                on_progress(i + 1, len(urls), page)
        return pages

    return _fetch_pages


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


async def run_one_turn():
    import deep_research.agent as agent_mod

    # Patch the fetch_pages dependency the agent imports at the top of agent.py
    agent_mod.fetch_pages = make_mock_fetch()  # type: ignore[attr-defined]

    with tempfile.TemporaryDirectory() as td:
        store = SessionStore(db_path=Path(td) / "e2e.db")
        agent = DeepResearchAgent(
            llm=MockLLM(),  # type: ignore[arg-type]
            search=MockSearch(),  # type: ignore[arg-type]
            store=store,
            max_search_results=4,
            max_pages_to_fetch=3,
            max_snippets=3,
            max_context_tokens=2000,
        )
        sid = store.create_session()
        phases_seen: list[str] = []
        answer_tokens: list[str] = []
        final = ""
        async for event in agent.run(sid, "What are topics Z and Y?"):
            phases_seen.append(event.phase.value)
            if event.phase == Phase.ANSWER_TOKEN:
                answer_tokens.append(event.data.get("delta", ""))
            if event.phase == Phase.DONE:
                final = event.data.get("final_answer", "")

        # --- Assertions ---
        for required in (
            "plan_start", "plan_done", "search_start", "search_done",
            "fetch_start", "fetch_progress", "fetch_done",
            "select_start", "select_done",
            "answer_start", "answer_token", "answer_done", "done",
        ):
            assert required in phases_seen, f"missing phase: {required}"

        assert "".join(answer_tokens), "no streamed tokens"
        assert "[S1]" not in final, "raw [S1] placeholder should have been expanded"
        assert "[S99]" not in final, "invalid [S99] should have been dropped"
        # The expansion adds a markdown link with the domain in it.
        assert "example0-0.com" in final or "example" in final, "no expanded citation"

        # Persistence check
        turns = store.get_turns(sid)
        assert len(turns) == 1
        t0 = turns[0]
        assert t0.plan is not None and t0.plan.search_queries
        assert t0.urls_opened, "urls_opened should not be empty"
        assert t0.snippets, "snippets should not be empty"
        assert t0.final_answer == final

        # Messages persisted
        msgs = store.get_messages(sid)
        assert len(msgs) == 2 and msgs[0].role == "user" and msgs[1].role == "assistant"

        print("  phases observed:", phases_seen)
        print("  snippets selected:", len(t0.snippets))
        print("  domains used:", sorted({s.domain for s in t0.snippets}))
        print("  final answer preview:", final[:200].replace("\n", " "))


def main() -> int:
    try:
        print("--- end-to-end mocked agent run ----------------------")
        asyncio.run(run_one_turn())
        print("\n  E2E PASSED")
        return 0
    except AssertionError as e:
        print(f"\n  E2E FAILED (assertion): {e}")
        return 1
    except Exception as e:
        print(f"\n  E2E FAILED (exception): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
