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
)
from tools.clinical_tools import (
    fetch_study_details,
    check_results_posted,
    get_study_amendments,
)
from config.settings import settings
from config.logging_config import setup_logging


logger = setup_logging(__name__)

AGENT_NAME = "timeline_agent"
SIGNAL_TYPE = "timeline_delay"

AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
    fetch_study_details,
    check_results_posted,
    get_study_amendments,
]

_procedural = ProceduralStore()
_episodic = EpisodicStore()

_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0.1,
    api_key=settings.openai_api_key,
).bind_tools(AGENT_TOOLS)


async def timeline_node(state: MosaicState) -> dict:
    """Detect silent delays in clinical trials."""

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        procedures = await _procedural.get_procedures(AGENT_NAME)
        procedures_text = "\n".join(f"- {rule}" for rule in procedures)

        system_prompt = f"""
You are the Timeline Analyst Agent for MOSAIC.

MISSION:
Find clinical trials significantly past their stated completion date
without an amendment or explanation for the delay.

REASONING RULES:
{procedures_text}

WORKFLOW:
1. Check past timeline investigations.
2. Search for delayed studies.
3. Fetch study details.
4. Check amendments for an official extension.
5. Flag delays greater than 180 days with no amendment.
6. Check the sponsor profile for repeated delays.
7. Update the sponsor profile.

SILENCE THRESHOLD:
Only flag delays greater than 180 days past the completion date
with no amendment filed.

CONFIDENCE SCORING:
- 0.9+ : 2+ years delayed, no amendment, no results, repeat offender
- 0.8  : 1-2 years delayed, no amendment, prior sponsor delays
- 0.7  : 6-12 months delayed, no amendment, first-time delay
- 0.6  : 6 months delayed, possible COVID overlap
- Below 0.60: Borderline timing — send to human review

OUTPUT FORMAT:
{{
    "nct_id": "NCT_ID",
    "signal_type": "timeline_delay",
    "summary": "How long delayed and whether an amendment was filed",
    "evidence": ["completion date", "current date", "amendment status"],
    "confidence": 0.80
}}

If no timeline violations are found, return NO_SIGNALS_FOUND.
"""

        task = state.get(
            "task",
            "Find trials with unexplained timeline delays"
        )

        nct_ids = state.get("nct_ids", [])

        human_message = f"""
ANALYSIS TASK: {task}

SPECIFIC STUDIES:
{nct_ids if nct_ids else "Search broadly for delayed trials"}

Begin the timeline investigation. Focus on silent delays.
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
                f"Found {len(signals_found)} timeline delay signals."
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