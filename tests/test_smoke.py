"""Offline smoke test — exercises every module without hitting any external API.

Run with:
    python tests/test_smoke.py
    # or
    python -m tests.test_smoke
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Make the project root importable when this script is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def banner(name: str) -> None:
    print(f"\n--- {name} -------------------------------------------------------")


def test_imports() -> None:
    banner("imports")
    from deep_research import DeepResearchAgent  # noqa: F401
    from deep_research import (  # noqa: F401
        agent, citations, context, fetch, llm, models, progress,
        prompts, search, session, summarizer,
    )
    print("  all modules import OK")


def test_models() -> None:
    banner("models / pydantic")
    from deep_research.models import (
        Plan, SearchResult, Snippet, Turn, Message, domain_of, now_iso,
    )
    assert domain_of("https://www.example.com/path") == "example.com"
    assert domain_of("https://docs.python.org/3/") == "docs.python.org"
    p = Plan(research_goal="g", sub_questions=["a"], search_queries=["q"])
    assert p.research_goal == "g"
    s = Snippet(sid="S1", text="hi", url="https://a.com", title="A",
                domain="a.com", retrieved_at=now_iso())
    assert s.sid == "S1"
    sr = SearchResult(title="t", url="https://b.com/x", snippet="snip", query="q")
    assert sr.domain == "b.com"  # auto-filled
    print("  Plan, Snippet, SearchResult, Turn, Message OK")


def test_session_store() -> None:
    banner("session store (SQLite)")
    from deep_research.session import SessionStore
    from deep_research.models import Message, Plan, Snippet, Turn, now_iso

    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        store = SessionStore(db_path=db)
        sid = store.create_session()
        assert sid

        store.append_message(sid, Message(role="user", content="hello"))
        store.append_message(sid, Message(role="assistant", content="hi"))
        msgs = store.get_messages(sid)
        assert len(msgs) == 2
        assert msgs[0].role == "user"

        turn = Turn(
            query="q",
            plan=Plan(research_goal="g", sub_questions=["a"], search_queries=["q1"]),
            search_queries=["q1"],
            urls_opened=["https://a.com/", "https://b.com/"],
            snippets=[
                Snippet(sid="S1", text="t1", url="https://a.com/", title="A",
                        domain="a.com", retrieved_at=now_iso(), score=0.9),
                Snippet(sid="S2", text="t2", url="https://b.com/", title="B",
                        domain="b.com", retrieved_at=now_iso(), score=0.8),
            ],
            final_answer="A — a.com",
            ts=now_iso(),
            latency_ms=1234,
        )
        turn_id = store.append_turn(sid, turn)
        assert turn_id > 0

        store.set_rolling_summary(sid, "some summary")
        assert store.get_rolling_summary(sid) == "some summary"

        turns = store.get_turns(sid)
        assert len(turns) == 1 and len(turns[0].snippets) == 2
        full = store.get_session(sid)
        assert full and len(full.messages) == 2 and len(full.turns) == 1
        all_sessions = store.list_sessions()
        assert len(all_sessions) == 1 and all_sessions[0]["n_turns"] == 1
        print("  insert + read + summary + list OK")


def test_chunker() -> None:
    banner("chunker")
    from deep_research.fetch import chunk_text

    text = ("Paragraph A. " * 50) + "\n\n" + ("Paragraph B. " * 50) + "\n\n" + ("Paragraph C. " * 50)
    chunks = chunk_text(text, target_words=80, overlap_words=15)
    assert len(chunks) >= 2
    print(f"  {len(chunks)} chunks produced")


def test_html_extraction() -> None:
    banner("html extraction (trafilatura + bs4 fallback)")
    from deep_research.fetch import _extract_text_and_title  # noqa: PLC2701

    html = """<!doctype html><html><head><title>My Title</title></head><body>
    <nav>NAV LINK</nav>
    <article>
      <h1>Hello</h1>
      <p>This is the main body of the article that talks about a particular fact.</p>
      <p>It also has a second paragraph that elaborates further on the topic.</p>
    </article>
    <footer>SITE FOOTER</footer>
    </body></html>"""
    text, title = _extract_text_and_title(html, "https://example.com/article")
    assert "main body" in text or "main body of the article" in text
    assert title  # something got extracted
    print(f"  title='{title}', text len={len(text)}")


def test_citation_expand() -> None:
    banner("citation expansion + validation")
    from deep_research.citations import expand_and_validate, citation_coverage
    from deep_research.models import Snippet, now_iso

    snippets = [
        Snippet(sid="S1", text="t1", url="https://a.com/", title="A",
                domain="a.com", retrieved_at=now_iso()),
        Snippet(sid="S2", text="t2", url="https://b.com/", title="B",
                domain="b.com", retrieved_at=now_iso()),
    ]
    raw = (
        "The sky is blue [S1]. Water boils at 100 C [S2]. "
        "An invalid reference [S99] should be dropped silently. "
        "Multiple citations work [S1][S2]."
    )
    expanded, report = expand_and_validate(raw, snippets)
    assert "[A — a.com](https://a.com/)" in expanded
    assert "[B — b.com](https://b.com/)" in expanded
    assert "[S99]" not in expanded
    assert "S99" in report.invalid_sids
    assert set(report.unique_sids) == {"S1", "S2"}
    cov = citation_coverage(raw)
    assert 0.0 <= cov <= 1.0
    print(f"  expansion OK; invalid={report.invalid_sids}; coverage={cov:.2f}")


def test_prompts() -> None:
    banner("prompts assembly")
    from deep_research.prompts import (
        answer_messages, planner_messages, summarizer_messages, judge_messages,
    )
    from deep_research.models import Snippet, now_iso

    snippets = [
        Snippet(sid="S1", text="text1", url="https://a.com", title="A",
                domain="a.com", retrieved_at=now_iso()),
    ]
    p = planner_messages("What is X?", rolling_summary="earlier we talked about Y")
    assert any(m.role == "system" for m in p)
    a = answer_messages("What is X?", snippets, rolling_summary="", recent_turns_text="")
    assert "[S1]" in a[-1].content
    s = summarizer_messages("User: hi\nAssistant: hello")
    assert len(s) == 2
    j = judge_messages("Q?", "[S1] block", "answer [S1]")
    assert "[S1]" in j[-1].content
    print("  planner, answer, summarizer, judge all build OK")


def test_context_selection() -> None:
    banner("context selection (MMR + budget)")
    from deep_research.context import select_context
    from deep_research.models import FetchedPage, now_iso

    pages = [
        FetchedPage(
            url=f"https://example{i}.com/",
            title=f"Title {i}",
            domain=f"example{i}.com",
            text=(f"This is article {i} about topic Z. " * 60),
            retrieved_at=now_iso(),
            ok=True,
        )
        for i in range(4)
    ]
    snippets = select_context("What is topic Z?", pages, max_snippets=4, max_tokens=1000)
    assert 1 <= len(snippets) <= 4
    sids = [s.sid for s in snippets]
    assert sids == [f"S{i+1}" for i in range(len(sids))]
    domains = {s.domain for s in snippets}
    print(f"  picked {len(snippets)} snippets from {len(domains)} domains: {sids}")


def test_event_pump() -> None:
    banner("async generator event pump")

    async def gen():
        for i in range(3):
            await asyncio.sleep(0.001)
            yield i

    async def drive():
        out = []
        async for v in gen():
            out.append(v)
        return out

    result = asyncio.run(drive())
    assert result == [0, 1, 2]
    print("  async generator drives correctly")


def main() -> int:
    try:
        test_imports()
        test_models()
        test_session_store()
        test_chunker()
        test_html_extraction()
        test_citation_expand()
        test_prompts()
        test_context_selection()
        test_event_pump()
    except AssertionError as e:
        print(f"\n  FAILED: assertion: {e}")
        return 1
    except Exception as e:
        print(f"\n  FAILED: exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n--- all smoke tests passed -------------------------------------------")
    return 0


if __name__ == "__main__":
    sys.exit(main())
