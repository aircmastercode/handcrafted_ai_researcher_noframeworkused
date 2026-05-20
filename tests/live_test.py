"""Live integration test against the real Tavily + Groq APIs.

Usage:
    python tests/live_test.py
    python tests/live_test.py "Your custom question here"

Requires .env with TAVILY_API_KEY and GROQ_API_KEY (or other configured LLM).
Will burn ~3-6 Tavily credits per run (well under the 1k/month free budget).
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from deep_research.agent import DeepResearchAgent  # noqa: E402
from deep_research.progress import Phase  # noqa: E402
from deep_research.session import SessionStore  # noqa: E402

DEFAULT_QUESTION = "Who founded Sarvam AI and when was the company founded?"


def _truncate(s: str, n: int = 140) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


async def run(question: str) -> int:
    load_dotenv()
    with tempfile.TemporaryDirectory() as td:
        store = SessionStore(db_path=Path(td) / "live.db")
        agent = DeepResearchAgent(store=store)
        sid = store.create_session()

        print(f"\nSession {sid}")
        print(f"Question: {question}\n")
        print("-" * 80)

        answer_chunks: list[str] = []
        final_answer = ""
        async for event in agent.run(sid, question):
            ph = event.phase
            if ph == Phase.PLAN_DONE:
                plan = event.data.get("plan") or {}
                qs = plan.get("search_queries", [])
                print("[PLAN] search queries:")
                for q in qs:
                    print(f"   - {q}")
            elif ph == Phase.SEARCH_DONE:
                results = event.data.get("results") or []
                print(f"[SEARCH] {len(results)} unique result(s):")
                for r in results[:5]:
                    print(f"   - {r['domain']:<30} {_truncate(r['title'], 80)}")
            elif ph == Phase.FETCH_PROGRESS:
                status = "OK" if event.data.get("ok", True) else "FAIL"
                print(f"[FETCH] {status}  {event.data.get('url')}")
            elif ph == Phase.SELECT_DONE:
                snips = event.data.get("snippets") or []
                doms = event.data.get("domains") or []
                print(f"[CONTEXT] {len(snips)} snippet(s) from {len(doms)} domain(s):")
                for s in snips:
                    print(f"   - [{s['sid']}] {s['domain']:<25} score={s['score']:.3f}")
            elif ph == Phase.ANSWER_START:
                print("\n[ANSWER stream] ↓\n")
            elif ph == Phase.ANSWER_TOKEN:
                delta = event.data.get("delta", "")
                answer_chunks.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()
            elif ph == Phase.ANSWER_DONE:
                final_answer = event.data.get("final_answer") or "".join(answer_chunks)
                print("\n")
            elif ph == Phase.DONE:
                print("-" * 80)
                lat = event.data.get("latency_ms", 0) / 1000
                cov = event.data.get("citation_coverage", 0.0)
                inv = event.data.get("invalid_sids") or []
                doms = event.data.get("n_domains", 0)
                snip = event.data.get("n_snippets", 0)
                print(
                    f"[DONE] latency={lat:.1f}s · {snip} snippets · {doms} domains · "
                    f"citation coverage={cov:.0%}"
                    + (f" · {len(inv)} invalid refs dropped" if inv else "")
                )
            elif ph == Phase.ERROR:
                print(f"\n[ERROR] {event.message}")
                return 1

        print("\n--- Final expanded answer (with markdown citations) ---\n")
        print(final_answer)
        print("\n-------------------------------------------------------\n")
        return 0


def main() -> int:
    q = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    return asyncio.run(run(q))


if __name__ == "__main__":
    sys.exit(main())
