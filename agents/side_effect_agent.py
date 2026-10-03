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
)
from tools.clinical_tools import fetch_study_details
from tools.pubmed_tools import (
    fetch_papers_for_trial,
    compare_filing_vs_papers,
)
from config.settings import settings
from config.logging_config import setup_logging


logger = setup_logging(__name__)

AGENT_NAME = "side_effect_agent"
SIGNAL_TYPE = "safety_gap"

AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    fetch_study_details,
    fetch_papers_for_trial,
    compare_filing_vs_papers,
]

_procedural = ProceduralStore()
_episodic = EpisodicStore()

_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0.1,
    api_key=settings.openai_api_key,
).bind_tools(AGENT_TOOLS)


async def side_effect_node(state: MosaicState) -> dict:
    """Find safety gaps between official filings and published papers."""

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        procedures = await _procedural.get_procedures(AGENT_NAME)
        procedures_text = "\n".join(f"- {rule}" for rule in procedures)

        system_prompt = f"""
You are the Side Effect Checker Agent for MOSAIC.

MISSION:
Find cases where official clinical trial safety reports disagree
with what researchers published in peer-reviewed journals.

REASONING RULES:
{procedures_text}

WORKFLOW:
1. Check past safety investigations.
2. Search for studies where safety is a concern.
3. Use compare_filing_vs_papers.
4. Compare official safety data with published papers.
5. Flag serious events missing from official filings.
6. Treat possible safety concerns cautiously.

LOOK FOR:
- Filing says no serious adverse events but papers mention hospitalisations.
- Filing reports mild effects but papers report severe effects.
- Papers report deaths or discontinuations missing from filings.
- Papers report results when official results were never posted.

CONFIDENCE SCORING:
- 0.9+ : No SAEs in filing, but papers describe hospitalisations/deaths
- 0.8  : Clear severity difference
- 0.7  : Additional side effects in papers
- 0.6  : Minor terminology differences
- 0.55 : Possible discrepancy requiring human review

OUTPUT FORMAT:
{{
    "nct_id": "NCT_ID",
    "signal_type": "safety_gap",
    "summary": "What the filing said vs what papers reported",
    "evidence": ["filing claim", "paper contradiction"],
    "confidence": 0.80
}}

If no gaps are found, return NO_SIGNALS_FOUND.
"""

        task = state.get(
            "task",
            "Find safety gaps between filings and papers"
        )

        nct_ids = state.get("nct_ids", [])

        human_message = f"""
ANALYSIS TASK: {task}

SPECIFIC STUDIES:
{nct_ids if nct_ids else "Search for studies with published papers"}

Begin the safety comparison.
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
                f"Found {len(signals_found)} safety gap signals."
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