import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from graph.state import MosaicState, SignalOutput
from memory.procedural_store import ProceduralStore
from memory.episodic_store import EpisodicStore
from tools.search_tools import (
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
    get_low_credibility_sponsors,
)
from tools.clinical_tools import fetch_study_details, search_studies_by_condition
from config.settings import settings
from config.logging_config import setup_logging


logger = setup_logging(__name__)

AGENT_NAME = "track_record_agent"
SIGNAL_TYPE = "low_credibility"

AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
    get_low_credibility_sponsors,
    fetch_study_details,
    search_studies_by_condition,
]

_procedural = ProceduralStore()
_episodic = EpisodicStore()

_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0.1,
    api_key=settings.openai_api_key,
).bind_tools(AGENT_TOOLS)


async def track_record_node(state: MosaicState) -> dict:
    """Evaluate sponsor credibility based on historical behavior."""

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        procedures = await _procedural.get_procedures(AGENT_NAME)
        procedures_text = "\n".join(f"- {rule}" for rule in procedures)

        system_prompt = f"""
You are the Track Record Agent for MOSAIC.

MISSION:
Evaluate research sponsor credibility based on their complete history
across clinical trials in the database.

REASONING RULES:
{procedures_text}

WORKFLOW:
1. Check past episodes.
2. Find sponsors below the credibility threshold.
3. Get their full sponsor profiles.
4. Analyze whether the behavior is systemic or isolated.
5. Generate low credibility signals for confirmed patterns.
6. Update sponsor profiles when needed.

CONFIDENCE SCORING:
- 0.9+ : Credibility < 0.4, 10+ studies, consistent pattern
- 0.8  : Credibility < 0.5, 5+ studies, clear pattern
- 0.7  : Credibility < 0.6, 3+ studies, emerging pattern
- Below 0.65: Insufficient data — send to human review

OUTPUT FORMAT:
{{
    "nct_id": "N/A or study ID",
    "signal_type": "low_credibility",
    "summary": "Sponsor name, credibility score, and pattern",
    "evidence": ["fact 1", "fact 2"],
    "confidence": 0.85
}}

If no sponsors below the threshold are found, return NO_SIGNALS_FOUND.
"""

        task = state.get(
            "task",
            "Evaluate sponsor credibility"
        )

        nct_ids = state.get("nct_ids", [])

        human_message = f"""
ANALYSIS TASK: {task}

SPECIFIC STUDIES:
{nct_ids if nct_ids else "Evaluate all sponsors in database"}

Begin the investigation.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]

        signals_found = []
        max_iterations = 10

        for iteration in range(max_iterations):
            response = await _llm.ainvoke(messages)

            messages.append(
                AIMessage(content=response.content or "")
            )

            if not response.tool_calls:
                logger.info(
                    f"{AGENT_NAME} | Analysis complete | "
                    f"iteration={iteration + 1}"
                )

                signals_found = _parse_signals(
                    response.content,
                    AGENT_NAME
                )
                break

            for tool_call in response.tool_calls:
                result = await _execute_tool(
                    tool_call,
                    AGENT_TOOLS
                )

                messages.append(
                    HumanMessage(
                        content=f"Tool result:\n{result}"
                    )
                )

        await _episodic.save_episode(
            agent_name=AGENT_NAME,
            content=(
                f"Task: {task}. "
                f"Found {len(signals_found)} low credibility signals."
            ),
            outcome=(
                "signal_generated"
                if signals_found
                else "no_signal"
            ),
        )

        logger.info(
            f"{AGENT_NAME} | Complete | "
            f"signals_found={len(signals_found)}"
        )

        return {
            "signals": state.get("signals", []) + signals_found,
            "agents_activated": (
                state.get("agents_activated", [])
                + [AGENT_NAME]
            ),
        }

    except Exception as e:
        logger.error(f"{AGENT_NAME} | Error | {e}")

        return {
            "error_log": state.get("error_log", []) + [
                f"{AGENT_NAME}: {str(e)}"
            ],
            "agents_activated": (
                state.get("agents_activated", [])
                + [AGENT_NAME]
            ),
        }


def _parse_signals(
    response_text: str,
    agent_name: str
) -> list[SignalOutput]:

    if not response_text or "NO_SIGNALS_FOUND" in response_text:
        return []

    pattern = re.compile(
        r'\{[^{}]*"signal_type"[^{}]*\}',
        re.DOTALL
    )

    signals = []

    for match in pattern.findall(response_text):
        try:
            data = json.loads(match)

            signals.append({
                "agent": agent_name,
                "signal_type": data.get(
                    "signal_type",
                    SIGNAL_TYPE
                ),
                "nct_id": data.get("nct_id", ""),
                "summary": data.get("summary", ""),
                "evidence": data.get("evidence", []),
                "confidence": float(
                    data.get("confidence", 0.5)
                ),
            })

        except (json.JSONDecodeError, ValueError):
            continue

    return signals


async def _execute_tool(
    tool_call: dict,
    available_tools: list
) -> str:

    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})

    tool_func = next(
        (
            tool
            for tool in available_tools
            if tool.name == tool_name
        ),
        None,
    )

    if tool_func is None:
        return f"Error: Tool '{tool_name}' not found."

    try:
        return str(tool_func.invoke(tool_args))

    except Exception as e:
        return f"Error executing '{tool_name}': {e}"