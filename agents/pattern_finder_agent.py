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
    get_low_credibility_sponsors,
)
from tools.clinical_tools import (
    search_studies_by_condition,
    fetch_study_details,
)
from tools.pubmed_tools import search_pubmed_by_query
from config.settings import settings
from config.logging_config import setup_logging


logger = setup_logging(__name__)

AGENT_NAME = "pattern_finder_agent"
SIGNAL_TYPE = "cross_study_pattern"

AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    get_low_credibility_sponsors,
    search_studies_by_condition,
    fetch_study_details,
    search_pubmed_by_query,
]

_procedural = ProceduralStore()
_episodic = EpisodicStore()

_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0.2,
    api_key=settings.openai_api_key,
).bind_tools(AGENT_TOOLS)


async def pattern_finder_node(state: MosaicState) -> dict:
    """Detect patterns across multiple clinical trials."""

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        procedures = await _procedural.get_procedures(AGENT_NAME)
        procedures_text = "\n".join(f"- {rule}" for rule in procedures)

        system_prompt = f"""
You are the Pattern Finder Agent for MOSAIC.

MISSION:
Detect research patterns spanning multiple clinical trials.

REASONING RULES:
{procedures_text}

WORKFLOW:
1. Search past episodes for previously identified patterns.
2. Search studies by condition, drug, or sponsor.
3. Look for the same drug across multiple sponsors.
4. Look for failed drugs being retried in another area.
5. Look for sponsors with repeated compliance issues.
6. Use PubMed to find papers connecting multiple trials.
7. A meaningful pattern requires at least 3 studies.

CONFIDENCE SCORING:
- 0.9+ : 5+ studies, same sponsor, clear systematic pattern
- 0.8  : 3-5 studies, clear pattern with strong evidence
- 0.7  : 3 studies, emerging pattern
- Below 0.65: Weak connection — send to human review

OUTPUT FORMAT:
{{
    "nct_id": "PATTERN or primary NCT_ID",
    "signal_type": "cross_study_pattern",
    "summary": "Pattern found across the studies",
    "evidence": ["study 1 finding", "study 2 finding", "connection"],
    "confidence": 0.80
}}

If no meaningful patterns are found, return NO_SIGNALS_FOUND.
"""

        task = state.get(
            "task",
            "Find cross-study patterns"
        )

        nct_ids = state.get("nct_ids", [])

        human_message = f"""
ANALYSIS TASK: {task}

SPECIFIC STUDIES:
{nct_ids if nct_ids else "Search broadly for patterns"}

Begin pattern detection. Connect evidence across studies.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]

        signals_found = []
        max_iterations = 12

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
                f"Found {len(signals_found)} cross-study pattern signals."
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