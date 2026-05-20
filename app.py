"""Streamlit UI for the Deep Research Agent.

Features:
  - Session selector (multiple persistent sessions, switch on the fly).
  - Chat-style conversation history with timestamps.
  - Live `st.status` log that streams every agent phase (planning, searching,
    fetching, selecting, answering) in real time.
  - The final answer streams token-by-token.
  - Expandable "audit" panel for each turn: plan, searched queries, opened URLs,
    selected snippets, citation report.

The agent itself is an async generator. We drive it synchronously here via a
small event-pump so Streamlit can render updates as events arrive.
"""

from __future__ import annotations

import asyncio
import os
from typing import Iterator

import streamlit as st
from dotenv import load_dotenv

from deep_research.agent import DeepResearchAgent
from deep_research.models import Message, Turn
from deep_research.progress import Phase, ProgressEvent
from deep_research.session import SessionStore

# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_secret(name: str, default: str = "") -> str:
    """Read a secret. HF Spaces injects secrets as env vars. Streamlit Cloud uses st.secrets."""
    val = os.getenv(name, "")
    if val:
        return val
    try:
        if name in st.secrets:
            return str(st.secrets[name]) or default
    except Exception:
        pass
    return default


# Push secrets into the env so the rest of the codebase can use os.getenv.
for _k in ("TAVILY_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "OLLAMA_URL", "OLLAMA_MODEL"):
    _v = _get_secret(_k)
    if _v:
        os.environ[_k] = _v


def _mask(v: str) -> str:
    """Return a masked preview of a secret, e.g. 'tvly****XXXX'."""
    if not v:
        return ""
    if len(v) <= 8:
        return "*" * len(v)
    return v[:4] + "*" * 4 + v[-4:]


@st.cache_resource(show_spinner=False)
def get_store() -> SessionStore:
    return SessionStore(db_path=os.getenv("SESSION_DB_PATH", "session.db"))


@st.cache_resource(show_spinner=False)
def get_agent() -> DeepResearchAgent:
    return DeepResearchAgent(store=get_store())


# ---------------------------------------------------------------------------
# Async-to-sync event pump
# ---------------------------------------------------------------------------


def run_agent_sync(agent: DeepResearchAgent, session_id: str, query: str) -> Iterator[ProgressEvent]:
    """Drive the async agent generator step-by-step from sync Streamlit code."""
    loop = asyncio.new_event_loop()
    try:
        agen = agent.run(session_id, query)
        while True:
            try:
                event = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
            yield event
    finally:
        try:
            loop.run_until_complete(asyncio.sleep(0))
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Sidebar — sessions + settings
# ---------------------------------------------------------------------------


def sidebar() -> str:
    st.sidebar.title("Deep Research")
    store = get_store()

    sessions = store.list_sessions()
    options = ["+ New session"] + [
        f"{s['session_id']}  ·  {s['n_turns']} turn(s)  ·  {s['created_at'][:16]}"
        for s in sessions
    ]
    default_index = 1 if sessions else 0
    choice = st.sidebar.selectbox(
        "Active session",
        options,
        index=st.session_state.get("session_select_idx", default_index),
        key="session_picker",
    )

    if choice == "+ New session":
        if st.sidebar.button("Create session", type="primary", use_container_width=True):
            sid = store.create_session()
            st.session_state["session_id"] = sid
            st.session_state["session_select_idx"] = 1
            st.rerun()
        # If no sessions exist yet, force-create one so the user can chat.
        if not sessions:
            sid = store.create_session()
            st.session_state["session_id"] = sid
            st.rerun()
        # Show a placeholder until the user clicks
        st.session_state["session_id"] = sessions[0]["session_id"] if sessions else ""
    else:
        sid = choice.split("  ·  ", 1)[0].strip()
        st.session_state["session_id"] = sid

    sid = st.session_state.get("session_id", "")
    if sid:
        st.sidebar.caption(f"session_id: `{sid}`")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Reset chat", use_container_width=True):
                store.delete_session(sid)
                new_sid = store.create_session()
                st.session_state["session_id"] = new_sid
                st.rerun()
        with col2:
            if st.button("Refresh", use_container_width=True):
                st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Status")
    st.sidebar.write(
        "Tavily: " + ("configured" if os.getenv("TAVILY_API_KEY") else "missing"),
    )
    st.sidebar.write(
        "Groq: " + ("configured" if os.getenv("GROQ_API_KEY") else "missing"),
    )
    st.sidebar.write(
        "Gemini (fallback): "
        + ("configured" if os.getenv("GEMINI_API_KEY") else "not set (optional)"),
    )
    st.sidebar.write(
        "Ollama (local fallback): "
        + (f"configured · {os.getenv('OLLAMA_MODEL', 'llama3.1:8b')}" if os.getenv("OLLAMA_URL") else "not set (optional)"),
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Built with no agent framework. "
        "Stack: Streamlit · Tavily · Groq · Ollama · Trafilatura · fastembed · SQLite."
    )
    return sid


# ---------------------------------------------------------------------------
# Conversation history rendering
# ---------------------------------------------------------------------------


def render_history(store: SessionStore, session_id: str) -> None:
    if not session_id:
        return
    messages = store.get_messages(session_id)
    turns = store.get_turns(session_id)
    turn_idx = 0
    for m in messages:
        with st.chat_message("user" if m.role == "user" else "assistant"):
            st.markdown(m.content)
            st.caption(m.ts)
            if m.role == "assistant" and turn_idx < len(turns):
                _render_turn_audit(turns[turn_idx])
                turn_idx += 1


def _render_turn_audit(turn: Turn) -> None:
    with st.expander("Audit — what this turn did", expanded=False):
        if turn.plan:
            st.markdown("**Plan**")
            st.markdown(f"- Goal: {turn.plan.research_goal}")
            if turn.plan.sub_questions:
                st.markdown("- Sub-questions:")
                for q in turn.plan.sub_questions:
                    st.markdown(f"  - {q}")
            if turn.plan.search_queries:
                st.markdown("- Search queries:")
                for q in turn.plan.search_queries:
                    st.markdown(f"  - `{q}`")
        if turn.urls_opened:
            st.markdown("**URLs opened**")
            for u in turn.urls_opened:
                st.markdown(f"- {u}")
        if turn.snippets:
            st.markdown(f"**Snippets selected** — {len(turn.snippets)}")
            for s in turn.snippets:
                st.markdown(
                    f"- `[{s.sid}]` [{s.title or s.domain or s.url}]({s.url}) "
                    f"— *{s.domain}* — score {s.score:.3f}"
                )
        if turn.latency_ms:
            st.caption(f"latency: {turn.latency_ms} ms")


# ---------------------------------------------------------------------------
# Run a turn — with live status updates
# ---------------------------------------------------------------------------


def run_turn(agent: DeepResearchAgent, session_id: str, query: str) -> None:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        status = st.status("Researching…", expanded=True)
        log_lines: list[str] = []
        log_placeholder = status.empty()
        answer_placeholder = st.empty()
        audit_placeholder = st.empty()
        answer_text = ""
        last_event: ProgressEvent | None = None

        def _log(text: str) -> None:
            log_lines.append(text)
            log_placeholder.markdown("\n\n".join(log_lines))

        try:
            for event in run_agent_sync(agent, session_id, query):
                last_event = event
                ph = event.phase
                if ph == Phase.PLAN_START:
                    _log("**Planning** research strategy…")
                elif ph == Phase.PLAN_DONE:
                    plan = event.data.get("plan") or {}
                    queries = plan.get("search_queries") or []
                    sub_qs = plan.get("sub_questions") or []
                    _log(
                        f"**Plan ready** — {len(sub_qs)} sub-question(s), "
                        f"{len(queries)} search query(ies)."
                    )
                    if queries:
                        _log("Search queries: " + ", ".join(f"`{q}`" for q in queries))
                elif ph == Phase.SEARCH_START:
                    _log("**Searching** the web…")
                elif ph == Phase.SEARCH_DONE:
                    results = event.data.get("results") or []
                    _log(f"**Search complete** — {len(results)} unique result(s).")
                elif ph == Phase.FETCH_START:
                    urls = event.data.get("urls") or []
                    _log(f"**Fetching** {len(urls)} page(s)…")
                elif ph == Phase.FETCH_PROGRESS:
                    ok = event.data.get("ok", True)
                    mark = "OK" if ok else f"FAIL ({event.data.get('error') or ''})"
                    _log(f"&nbsp;&nbsp;{event.message} — {mark}")
                elif ph == Phase.FETCH_DONE:
                    ok_count = event.data.get("ok_count", 0)
                    total = len(event.data.get("urls") or [])
                    _log(f"**Fetch complete** — {ok_count}/{total} succeeded.")
                elif ph == Phase.SELECT_START:
                    _log("**Selecting** the most relevant snippets…")
                elif ph == Phase.SELECT_DONE:
                    snippets = event.data.get("snippets") or []
                    domains = event.data.get("domains") or []
                    _log(
                        f"**Context selected** — {len(snippets)} snippet(s) "
                        f"from {len(domains)} domain(s)."
                    )
                elif ph == Phase.ANSWER_START:
                    _log("**Generating** the grounded answer with citations…")
                    status.update(label="Generating answer", state="running", expanded=True)
                elif ph == Phase.ANSWER_TOKEN:
                    answer_text += event.data.get("delta", "")
                    answer_placeholder.markdown(answer_text + " ▌")
                elif ph == Phase.ANSWER_DONE:
                    final_answer = event.data.get("final_answer") or answer_text
                    answer_placeholder.markdown(final_answer)
                elif ph == Phase.DONE:
                    final_answer = event.data.get("final_answer") or answer_text
                    answer_placeholder.markdown(final_answer)
                    n_dom = event.data.get("n_domains", 0)
                    n_snip = event.data.get("n_snippets", 0)
                    cov = event.data.get("citation_coverage", 0.0)
                    invalid = event.data.get("invalid_sids") or []
                    status.update(
                        label=(
                            f"Done · {event.data.get('latency_ms', 0) / 1000:.1f}s "
                            f"· {n_snip} snippets · {n_dom} domains · "
                            f"citation coverage {cov:.0%}"
                            + (f" · {len(invalid)} invalid refs dropped" if invalid else "")
                        ),
                        state="complete",
                        expanded=False,
                    )
                    # Render the audit panel for the just-finished turn
                    turns = get_store().get_turns(session_id)
                    if turns:
                        with audit_placeholder.container():
                            _render_turn_audit(turns[-1])
                elif ph == Phase.ERROR:
                    status.update(label="Error", state="error", expanded=True)
                    st.error(event.message)
                    return
        except Exception as e:  # noqa: BLE001
            status.update(label="Crashed", state="error", expanded=True)
            st.exception(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    session_id = sidebar()

    st.title("Deep Research Agent")
    st.caption(
        "Asks the live web, reads the actual pages, picks the best snippets, "
        "and answers with real citations."
    )

    tavily = os.getenv("TAVILY_API_KEY", "")
    groq = os.getenv("GROQ_API_KEY", "")
    gemini = os.getenv("GEMINI_API_KEY", "")
    if not (tavily and (groq or gemini)):
        st.warning(
            "Missing credentials — the agent cannot run until these are set.",
            icon=None,
        )
        with st.expander("Diagnostics — what the container actually sees", expanded=True):
            st.markdown(
                f"""
- `TAVILY_API_KEY`: **{('configured (' + _mask(tavily) + ')') if tavily else 'MISSING'}**
- `GROQ_API_KEY`: **{('configured (' + _mask(groq) + ')') if groq else 'MISSING'}**
- `GEMINI_API_KEY`: **{('configured (' + _mask(gemini) + ')') if gemini else 'not set (optional)'}**

**If you just added the secrets in Hugging Face → Settings → Variables and secrets,
the container has to be restarted for them to be injected as environment variables.**

To restart:
1. Open the Space's **Settings** tab.
2. Scroll to the very bottom → click **Factory rebuild** (or **Restart this Space**).
3. Wait ~30 seconds for the container to come back up, then refresh this page.

If after a restart this still says MISSING, double-check:
- The secret **name** is spelled exactly `TAVILY_API_KEY` / `GROQ_API_KEY` (uppercase, underscore).
- You clicked **Save** after typing the value (HF requires an explicit save).
- The value has no leading/trailing whitespace.
""",
            )
        return

    if not session_id:
        st.info("Create a new session from the sidebar to start chatting.")
        return

    render_history(get_store(), session_id)

    query = st.chat_input("Ask anything — I'll search the web, read the sources, and cite them.")
    if not query:
        return

    try:
        agent = get_agent()
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to initialize agent: {e!s}")
        return

    run_turn(agent, session_id, query)


if __name__ == "__main__":
    main()
