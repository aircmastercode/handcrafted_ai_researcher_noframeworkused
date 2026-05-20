"""Live multi-turn test — verifies session memory across follow-ups.

Asks an initial question and two follow-ups in the SAME session, checking that
the agent uses prior conversation context (the follow-ups don't restate the topic).
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


async def _one(agent, sid, q: str) -> str:
    print(f"\n>>> {q}\n")
    final = ""
    async for event in agent.run(sid, q):
        if event.phase == Phase.ANSWER_TOKEN:
            sys.stdout.write(event.data.get("delta", ""))
            sys.stdout.flush()
        elif event.phase == Phase.DONE:
            final = event.data.get("final_answer", "")
            lat = event.data.get("latency_ms", 0) / 1000
            cov = event.data.get("citation_coverage", 0.0)
            print(f"\n\n[turn done — {lat:.1f}s, coverage={cov:.0%}]")
        elif event.phase == Phase.ERROR:
            print(f"\n[ERROR] {event.message}")
            return ""
    return final


async def main():
    load_dotenv()
    with tempfile.TemporaryDirectory() as td:
        store = SessionStore(db_path=Path(td) / "mt.db")
        agent = DeepResearchAgent(store=store)
        sid = store.create_session()
        print(f"Session: {sid}")
        print("=" * 80)

        await _one(agent, sid, "What is Sarvam AI?")
        print("\n" + "=" * 80)
        await _one(agent, sid, "Who are its founders?")
        print("\n" + "=" * 80)
        await _one(agent, sid, "What language models has it released?")

        print("\n" + "=" * 80)
        # Verify session state
        turns = store.get_turns(sid)
        msgs = store.get_messages(sid)
        print(f"\nFinal session state:  {len(turns)} turn(s),  {len(msgs)} message(s)")
        for i, t in enumerate(turns, 1):
            doms = sorted({s.domain for s in t.snippets})
            print(f"  turn {i}: q={t.query[:60]!r}")
            print(f"          {len(t.snippets)} snippets, domains={doms}")
            print(f"          latency={t.latency_ms} ms")


if __name__ == "__main__":
    asyncio.run(main())
