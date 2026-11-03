
from typing import Any, TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


def _merge_lists(a: list, b: list) -> list:
    """
    Reducer that APPENDS lists instead of replacing them.
    Used for signals and error logs so parallel agent outputs
    are preserved.
    """
    return (a or []) + (b or [])


def _merge_unique(a: list, b: list) -> list:
    """
    Merge agent names while preserving insertion order.
    Prevents duplicate agent entries during parallel execution.
    """
    return list(dict.fromkeys((a or []) + (b or [])))


class SignalOutput(TypedDict):
    agent:       str
    signal_type: str
    nct_id:      str
    summary:     str
    evidence:    list
    confidence:  float


class MosaicState(TypedDict):

    # ── TASK INPUT ─────────────────────────────────────────────
    task:        str
    nct_ids:     list[str]
    max_studies: int

    # ── CONVERSATION HISTORY ───────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── AGENT OUTPUTS ──────────────────────────────────────────
    signals:          Annotated[list[SignalOutput], _merge_lists]
    agents_activated: Annotated[list[str], _merge_unique]
    error_log:        Annotated[list, _merge_lists]

    # ── SUPERVISOR OUTPUTS ─────────────────────────────────────
    final_brief:  str
    run_complete: bool

    # ── METADATA ───────────────────────────────────────────────
    run_id: str
