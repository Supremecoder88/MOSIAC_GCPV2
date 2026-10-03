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
from tools.clinical_tools import check_results_posted, fetch_study_details
from config.settings import settings
from config.logging_config import setup_logging


logger = setup_logging(__name__)

AGENT_NAME = "missing_results_agent"
SIGNAL_TYPE = "missing_results"

AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
    check_results_posted,
    fetch_study_details,
]

_procedural = ProceduralStore()
_episodic = EpisodicStore()

_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0.1,
    api_key=settings.openai_api_key,
).bind_tools(AGENT_TOOLS)


async def missing_results_node(state: MosaicState) -> dict:
    """Find completed clinical trials with missing results."""

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        procedures = await _procedural.get_procedures(AGENT_NAME)
        procedures_text = "\n".join(f"- {rule}" for rule in procedures)

        system_prompt = f"""
You are the Missing Results Agent for MOSAIC.

MISSION:
Find completed clinical trials that have never posted their results.

REASONING RULES:
{procedures_text}

WORKFLOW:
1. Search past episodes.
2. Search for completed studies with missing results.
3. Verify suspicious studies using the live API.
4. Check the sponsor profile.
5. Generate signals with confidence scores.
6. Update the sponsor profile.
7. Save the session as an episode.

OUTPUT FORMAT:
{{
    "nct_id": "NCT_ID",
    "signal_type": "missing_results",
    "summary": "Description",
    "evidence": ["fact 1", "fact 2"],
    "confidence": 0.85
}}

If there are no signals, return NO_SIGNALS_FOUND.
"""

        task = state.get(
            "task",
            "Find completed trials with missing results"
        )

        nct_ids = state.get("nct_ids", [])

        human_message = f"""
ANALYSIS TASK: {task}

SPECIFIC STUDIES:
{nct_ids if nct_ids else "Search broadly"}

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

        episode_content = (
            f"Task: {task}. "
            f"Found {len(signals_found)} signals. "
            f"Signals: {[s.get('nct_id') for s in signals_found]}"
        )

        await _episodic.save_episode(
            agent_name=AGENT_NAME,
            content=episode_content,
            outcome="signal_generated" if signals_found else "no_signal",
        )

        current_signals = state.get("signals", [])
        current_activated = state.get("agents_activated", [])

        return {
            "signals": current_signals + signals_found,
            "agents_activated": current_activated + [AGENT_NAME],
        }

    except Exception as e:
        logger.error(f"{AGENT_NAME} | Error | {e}")

        return {
            "error_log": state.get("error_log", []) + [
                f"{AGENT_NAME}: {str(e)}"
            ],
            "agents_activated": state.get(
                "agents_activated",
                []
            ) + [AGENT_NAME],
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

            signal: SignalOutput = {
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
            }

            signals.append(signal)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Could not parse signal JSON: {e}"
            )

    return signals


async def _execute_tool(
    tool_call: dict,
    available_tools: list
) -> str:

    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})

    tool_func = None

    for tool in available_tools:
        if tool.name == tool_name:
            tool_func = tool
            break

    if tool_func is None:
        return f"Error: Tool '{tool_name}' not found."

    try:
        result = tool_func.invoke(tool_args)
        return str(result)

    except Exception as e:
        logger.error(
            f"Tool execution failed | "
            f"tool={tool_name} | error={e}"
        )
        return f"Error executing tool: {e}"