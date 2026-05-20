"""Run the evaluation harness end-to-end.

Usage:
    python -m eval.run_eval                       # run the bundled dataset
    python -m eval.run_eval --dataset other.json  # use a custom dataset
    python -m eval.run_eval --skip-judge          # skip the LLM-as-judge step
    python -m eval.run_eval --limit 5             # quick smoke test on first 5

Outputs:
    eval/results.json   — per-item raw scores + agent transcripts
    eval/report.md      — human-readable summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from deep_research.agent import DeepResearchAgent
from deep_research.citations import _CITE_RE, citation_coverage  # type: ignore
from deep_research.llm import LLMClient
from deep_research.models import Snippet
from deep_research.progress import Phase
from deep_research.session import SessionStore
from eval.judges import judge_answer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval")

HERE = Path(__file__).parent
DEFAULT_DATASET = HERE / "dataset.json"
RESULTS_PATH = HERE / "results.json"
REPORT_PATH = HERE / "report.md"


# ---------------------------------------------------------------------------
# Refusal & conflict keyword heuristics
# ---------------------------------------------------------------------------

_REFUSAL_PATTERNS = re.compile(
    r"\b("
    r"cannot find|could not find|insufficient evidence|not (?:publicly )?available|"
    r"not enough (?:reliable )?information|i don'?t know|"
    r"cannot (?:reliably )?predict|cannot determine|"
    r"unable to (?:find|verify|determine)|no reliable source|not disclosed"
    r")\b",
    re.IGNORECASE,
)

_CONFLICT_PATTERNS = re.compile(
    r"\b("
    r"disagree|disagreement|conflicting|contradict|however|"
    r"on the other hand|in contrast|mixed (?:evidence|results)"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Agent driver
# ---------------------------------------------------------------------------


async def run_single_turn(
    agent: DeepResearchAgent, session_id: str, query: str
) -> dict[str, Any]:
    """Run one turn and capture everything the eval needs."""
    t0 = time.perf_counter()
    raw_answer = ""
    final_answer = ""
    snippets_payload: list[dict] = []
    queries: list[str] = []
    urls: list[str] = []
    coverage = 0.0
    invalid_sids: list[str] = []
    error: str | None = None

    async for event in agent.run(session_id, query):
        if event.phase == Phase.PLAN_DONE:
            queries = (event.data.get("plan") or {}).get("search_queries", [])
        elif event.phase == Phase.FETCH_DONE:
            urls = event.data.get("urls", [])
        elif event.phase == Phase.SELECT_DONE:
            snippets_payload = event.data.get("snippets", [])
        elif event.phase == Phase.ANSWER_DONE:
            raw_answer = event.data.get("raw_answer", "")
            final_answer = event.data.get("final_answer", "")
            coverage = float(event.data.get("citation_coverage", 0.0))
            invalid_sids = event.data.get("invalid_sids", [])
        elif event.phase == Phase.ERROR:
            error = event.message
            break

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "query": query,
        "raw_answer": raw_answer,
        "final_answer": final_answer,
        "search_queries": queries,
        "urls_opened": urls,
        "snippets": snippets_payload,
        "citation_coverage": coverage,
        "invalid_sids": invalid_sids,
        "latency_ms": latency_ms,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Automatic per-item metrics
# ---------------------------------------------------------------------------


def auto_metrics(turn: dict[str, Any], category: str) -> dict[str, Any]:
    snippet_sids = {s["sid"] for s in turn["snippets"]}
    snippet_urls = {s["url"] for s in turn["snippets"]}
    snippet_domains_by_sid = {s["sid"]: s["domain"] for s in turn["snippets"]}

    cited_sids = list(dict.fromkeys(m.group(0)[1:-1].upper() for m in _CITE_RE.finditer(turn["raw_answer"])))
    cited_sids = [s if s.startswith("S") else "S" + s[1:] for s in cited_sids]

    # Citation validity: every cited [S#] must map to a real snippet.
    n_cites = len(cited_sids)
    n_valid = sum(1 for s in cited_sids if s in snippet_sids)
    cite_validity = (n_valid / n_cites) if n_cites else (1.0 if turn["final_answer"] else 0.0)

    # Source diversity: unique cited domains / total citations (capped at 1.0).
    unique_domains = {snippet_domains_by_sid.get(s, "") for s in cited_sids if s in snippet_sids}
    unique_domains.discard("")
    diversity = (len(unique_domains) / max(1, n_cites)) if n_cites else 0.0

    coverage = float(turn.get("citation_coverage", 0.0))

    answer_text = turn.get("raw_answer", "") + " " + turn.get("final_answer", "")
    conflict_handled = bool(_CONFLICT_PATTERNS.search(answer_text))
    refused = bool(_REFUSAL_PATTERNS.search(answer_text))

    category_pass: bool | None = None
    if category == "conflicting_sources":
        # Pass = answer must explicitly acknowledge disagreement AND cite at least two domains.
        category_pass = conflict_handled and len(unique_domains) >= 2
    elif category == "insufficient_evidence":
        category_pass = refused

    return {
        "n_citations": n_cites,
        "citation_validity": cite_validity,
        "citation_coverage": coverage,
        "source_diversity": diversity,
        "unique_cited_domains": sorted(unique_domains),
        "conflict_handled": conflict_handled,
        "refused": refused,
        "category_pass": category_pass,
        "snippet_urls_count": len(snippet_urls),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    cleaned = [v for v in values if v is not None]
    return statistics.fmean(cleaned) if cleaned else None


def _p95(values: list[float]) -> float | None:
    cleaned = sorted(v for v in values if v is not None)
    if not cleaned:
        return None
    idx = max(0, int(round(0.95 * (len(cleaned) - 1))))
    return cleaned[idx]


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)

    summary: dict[str, Any] = {"per_category": {}, "overall": {}}
    for cat, rows in by_cat.items():
        summary["per_category"][cat] = {
            "n": len(rows),
            "citation_coverage": _mean([r["auto"]["citation_coverage"] for r in rows]),
            "citation_validity": _mean([r["auto"]["citation_validity"] for r in rows]),
            "source_diversity": _mean([r["auto"]["source_diversity"] for r in rows]),
            "faithfulness": _mean([r.get("judge", {}).get("faithfulness") for r in rows]),
            "relevance": _mean([r.get("judge", {}).get("relevance") for r in rows]),
            "latency_ms_mean": _mean([r["turn"]["latency_ms"] for r in rows]),
            "latency_ms_p95": _p95([r["turn"]["latency_ms"] for r in rows]),
            "category_pass_rate": _mean(
                [
                    1.0 if r["auto"]["category_pass"] else 0.0
                    for r in rows
                    if r["auto"]["category_pass"] is not None
                ]
            ),
        }

    all_rows = items
    summary["overall"] = {
        "n": len(all_rows),
        "citation_coverage": _mean([r["auto"]["citation_coverage"] for r in all_rows]),
        "citation_validity": _mean([r["auto"]["citation_validity"] for r in all_rows]),
        "source_diversity": _mean([r["auto"]["source_diversity"] for r in all_rows]),
        "faithfulness": _mean([r.get("judge", {}).get("faithfulness") for r in all_rows]),
        "relevance": _mean([r.get("judge", {}).get("relevance") for r in all_rows]),
        "latency_ms_mean": _mean([r["turn"]["latency_ms"] for r in all_rows]),
        "latency_ms_p95": _p95([r["turn"]["latency_ms"] for r in all_rows]),
    }
    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(v: float | None, *, pct: bool = False, digits: int = 2) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.0f}%"
    return f"{v:.{digits}f}"


def write_report(items: list[dict[str, Any]], agg: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Deep Research Agent — Evaluation Report\n")
    lines.append(f"_Dataset size: **{agg['overall']['n']}** items_\n")

    lines.append("## Overall\n")
    o = agg["overall"]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Citation coverage (auto)   | {_fmt(o['citation_coverage'], pct=True)} |")
    lines.append(f"| Citation validity (auto)   | {_fmt(o['citation_validity'], pct=True)} |")
    lines.append(f"| Source diversity (auto)    | {_fmt(o['source_diversity'])} |")
    lines.append(f"| Faithfulness (LLM judge)   | {_fmt(o['faithfulness'])} |")
    lines.append(f"| Relevance (LLM judge)      | {_fmt(o['relevance'])} |")
    lines.append(f"| Latency mean (ms)          | {_fmt(o['latency_ms_mean'], digits=0)} |")
    lines.append(f"| Latency p95 (ms)           | {_fmt(o['latency_ms_p95'], digits=0)} |")
    lines.append("")

    lines.append("## Per category\n")
    lines.append("| Category | n | Coverage | Validity | Diversity | Faithfulness | Relevance | Category pass | Latency p95 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cat, row in agg["per_category"].items():
        lines.append(
            f"| {cat} | {row['n']} | {_fmt(row['citation_coverage'], pct=True)} | "
            f"{_fmt(row['citation_validity'], pct=True)} | "
            f"{_fmt(row['source_diversity'])} | {_fmt(row['faithfulness'])} | "
            f"{_fmt(row['relevance'])} | "
            f"{_fmt(row['category_pass_rate'], pct=True)} | "
            f"{_fmt(row['latency_ms_p95'], digits=0)} |"
        )
    lines.append("")

    lines.append("## Per item\n")
    for it in items:
        a = it["auto"]
        j = it.get("judge", {})
        lines.append(f"### `{it['id']}` — {it['category']}")
        lines.append(f"**Question:** {it['turn']['query']}")
        lines.append("")
        if it["turn"].get("error"):
            lines.append(f"> Error: {it['turn']['error']}")
        lines.append(
            "- coverage: {cov} · validity: {val} · diversity: {div} · "
            "faithfulness: {fa} · relevance: {rel} · latency: {lat} ms".format(
                cov=_fmt(a["citation_coverage"], pct=True),
                val=_fmt(a["citation_validity"], pct=True),
                div=_fmt(a["source_diversity"]),
                fa=_fmt(j.get("faithfulness")),
                rel=_fmt(j.get("relevance")),
                lat=it["turn"]["latency_ms"],
            )
        )
        if a["category_pass"] is not None:
            lines.append(f"- category pass: **{'yes' if a['category_pass'] else 'no'}**")
        lines.append(f"- cited domains: {', '.join(a['unique_cited_domains']) or '—'}")
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(it["turn"]["final_answer"] or "_(no answer)_")
        lines.append("\n---\n")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv()
    dataset_path = Path(args.dataset)
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    items_in = data["items"]
    if args.limit:
        items_in = items_in[: args.limit]

    db_path = args.db_path or os.getenv("EVAL_DB_PATH", "eval_session.db")
    store = SessionStore(db_path=db_path)
    agent = DeepResearchAgent(store=store)
    judge_llm = LLMClient() if not args.skip_judge else None

    results: list[dict[str, Any]] = []

    for i, item in enumerate(items_in, start=1):
        qid = item["id"]
        question = item["question"]
        category = item["category"]
        follow_ups = item.get("follow_ups", [])
        run_session = f"eval-{uuid.uuid4().hex[:8]}-{qid}"
        log.warning("[%d/%d] %s (%s)", i, len(items_in), qid, category)

        try:
            turn = await run_single_turn(agent, run_session, question)
        except Exception as e:  # noqa: BLE001
            log.exception("Agent crashed on %s", qid)
            turn = {
                "query": question, "raw_answer": "", "final_answer": "",
                "search_queries": [], "urls_opened": [], "snippets": [],
                "citation_coverage": 0.0, "invalid_sids": [],
                "latency_ms": 0, "error": str(e),
            }

        snippets_objs = [Snippet(**s) for s in turn["snippets"]]
        auto = auto_metrics(turn, category)

        judge: dict[str, Any] = {}
        if judge_llm and turn["raw_answer"]:
            try:
                judge = await judge_answer(
                    judge_llm,
                    question=question,
                    snippets=snippets_objs,
                    answer_with_placeholders=turn["raw_answer"],
                )
            except Exception as e:  # noqa: BLE001
                judge = {"error": str(e)}

        item_result = {
            "id": qid,
            "category": category,
            "expected_behavior": item.get("expected_behavior", ""),
            "turn": turn,
            "auto": auto,
            "judge": judge,
            "follow_ups": [],
        }

        # Multi-turn follow-ups (use the SAME session to test memory)
        if category == "multi_turn" and follow_ups:
            for j, fq in enumerate(follow_ups, start=1):
                log.warning("    follow-up %d/%d", j, len(follow_ups))
                try:
                    fturn = await run_single_turn(agent, run_session, fq)
                except Exception as e:  # noqa: BLE001
                    fturn = {
                        "query": fq, "raw_answer": "", "final_answer": "",
                        "search_queries": [], "urls_opened": [], "snippets": [],
                        "citation_coverage": 0.0, "invalid_sids": [],
                        "latency_ms": 0, "error": str(e),
                    }
                fsnips = [Snippet(**s) for s in fturn["snippets"]]
                fauto = auto_metrics(fturn, "multi_turn")
                fjudge: dict[str, Any] = {}
                if judge_llm and fturn["raw_answer"]:
                    try:
                        fjudge = await judge_answer(
                            judge_llm,
                            question=fq,
                            snippets=fsnips,
                            answer_with_placeholders=fturn["raw_answer"],
                        )
                    except Exception as e:  # noqa: BLE001
                        fjudge = {"error": str(e)}
                item_result["follow_ups"].append(
                    {"turn": fturn, "auto": fauto, "judge": fjudge}
                )

        results.append(item_result)

    # Flatten for aggregation (multi-turn follow-ups are added as separate rows)
    flat_for_agg: list[dict[str, Any]] = []
    for r in results:
        flat_for_agg.append(r)
        for fu in r.get("follow_ups", []):
            flat_for_agg.append(
                {
                    "id": r["id"] + "-followup",
                    "category": "multi_turn",
                    "turn": fu["turn"],
                    "auto": fu["auto"],
                    "judge": fu["judge"],
                }
            )

    agg = aggregate(flat_for_agg)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"items": results, "summary": agg}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(flat_for_agg, agg, REPORT_PATH)

    log.warning("Results -> %s", RESULTS_PATH)
    log.warning("Report  -> %s", REPORT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Deep Research Agent.")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--db-path", type=str, default="")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
