"""Typed progress events emitted by the agent as it runs.

The agent is an async generator that yields these events. The UI maps them to
`st.status` updates and `st.write_stream` streaming. We never emit hidden
chain-of-thought — only operational status.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Phase(str, Enum):
    PLAN_START = "plan_start"
    PLAN_DONE = "plan_done"
    SEARCH_START = "search_start"
    SEARCH_DONE = "search_done"
    FETCH_START = "fetch_start"
    FETCH_PROGRESS = "fetch_progress"
    FETCH_DONE = "fetch_done"
    SELECT_START = "select_start"
    SELECT_DONE = "select_done"
    ANSWER_START = "answer_start"
    ANSWER_TOKEN = "answer_token"
    ANSWER_DONE = "answer_done"
    DONE = "done"
    ERROR = "error"


# Human-readable labels for the UI.
PHASE_LABELS: dict[Phase, str] = {
    Phase.PLAN_START: "Planning research strategy",
    Phase.PLAN_DONE: "Plan ready",
    Phase.SEARCH_START: "Searching the web",
    Phase.SEARCH_DONE: "Search complete",
    Phase.FETCH_START: "Fetching pages",
    Phase.FETCH_PROGRESS: "Fetching pages",
    Phase.FETCH_DONE: "Fetch complete",
    Phase.SELECT_START: "Selecting relevant context",
    Phase.SELECT_DONE: "Context selected",
    Phase.ANSWER_START: "Generating grounded answer",
    Phase.ANSWER_TOKEN: "Streaming answer",
    Phase.ANSWER_DONE: "Answer complete",
    Phase.DONE: "Done",
    Phase.ERROR: "Error",
}


class ProgressEvent(BaseModel):
    """One operational update from the agent."""

    phase: Phase
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def label(self) -> str:
        return PHASE_LABELS.get(self.phase, self.phase.value)


def evt(phase: Phase, message: str = "", **data: Any) -> ProgressEvent:
    """Concise constructor used inside the agent."""
    return ProgressEvent(phase=phase, message=message, data=data)
