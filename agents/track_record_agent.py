
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from graph.state import MosaicState, SignalOutput
from memory.procedural_store import ProceduralStore
from memory.episodic_store import EpisodicStore

from tools.search_tools import (
    search_studies_by_meaning,
    search_past_episodes,
    get_sponsor_profile,
    update_sponsor_profile,
    get_low_credibility_sponsors,
)

from tools.clinical_tools import (
    fetch_study_details,
    search_studies_by_condition,
)

from config.settings import settings
from config.logging_config import setup_logging


logger = setup_logging(__name__)

AGENT_NAME = "track_record_agent"
SIGNAL_TYPE = "low_credibility"
MAX_ITERATIONS = 10


AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
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
        # Load procedural rules
        procedures = await _procedural.get_procedures(AGENT_NAME)

        procedures_text = "\n".join(
            f"- {rule}" for rule in procedures
        )

        system_prompt = f"""
You are the Track Record Agent for MOSAIC.

MISSION:
Evaluate research sponsor credibility based on their history
across clinical trials in the database.

REASONING RULES:
{procedures_text}

WORKFLOW:
1. Check past episodes for relevant sponsor investigations.
2. Find sponsors below the credibility threshold.
3. Retrieve their complete sponsor profiles.
4. Analyze whether the behavior is systemic or isolated.
5. Generate low credibility signals for confirmed patterns.
6. Update sponsor profiles when appropriate.

CONFIDENCE SCORING:
- 0.9+ : Credibility below 0.4, 10+ studies, consistent pattern.
- 0.8  : Credibility below 0.5, 5+ studies, clear pattern.
- 0.7  : Credibility below 0.6, 3+ studies, emerging pattern.
- Below 0.65: Insufficient data — send to human review.

EVIDENCE RULES:
- Base credibility assessments on retrieved sponsor profiles.
- Do not invent sponsor statistics or study counts.
- Distinguish missing data from negative sponsor behavior.
- Do not classify a sponsor as low credibility based on one
  isolated incident without supporting evidence.
- Consider the number of studies and available evidence.
- Do not generate signals when evidence is insufficient.
- Do not assume a missing sponsor profile means low credibility.

OUTPUT FORMAT:
Return valid JSON objects for detected signals:

{{
    "nct_id": "A valid NCT ID associated with the evidence, or null if unavailable",
    "signal_type": "low_credibility",
    "summary": "Sponsor name, credibility score, and pattern",
    "evidence": ["fact 1", "fact 2"],
    "confidence": 0.85
}}

If no verified signals are found, return:
NO_SIGNALS_FOUND

Return the final result only after completing the investigation.
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
        analysis_completed = False

        # LLM tool-calling loop
        for iteration in range(MAX_ITERATIONS):

            logger.info(
                f"{AGENT_NAME} | LLM iteration "
                f"{iteration + 1}/{MAX_ITERATIONS}"
            )

            response = await _llm.ainvoke(messages)

            # Preserve original tool-call metadata
            messages.append(response)

            # Final LLM response
            if not response.tool_calls:

                signals_found = _parse_signals(
                    response.content,
                    AGENT_NAME
                )

                analysis_completed = True

                logger.info(
                    f"{AGENT_NAME} | Analysis completed "
                    f"| iteration={iteration + 1}"
                )

                break

            # Execute requested tools
            for tool_call in response.tool_calls:

                tool_name = tool_call.get("name", "")

                logger.info(
                    f"{AGENT_NAME} | Executing tool "
                    f"| name={tool_name}"
                )

                result = await _execute_tool(
                    tool_call,
                    AGENT_TOOLS
                )

                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )

        # Prevent incomplete investigations from being
        # incorrectly saved as no_signal
        if not analysis_completed:
            raise RuntimeError(
                f"{AGENT_NAME} reached maximum LLM iterations "
                "without completing analysis."
            )

        # Extract unique, valid NCT IDs from signals
        signal_nct_ids = []

        for signal in signals_found:

            possible_ids = []

            signal_nct_id = signal.get("nct_id", "")

            if signal_nct_id:
                possible_ids.append(str(signal_nct_id))

            evidence = signal.get("evidence", [])

            if isinstance(evidence, list):
                possible_ids.extend(
                    str(item) for item in evidence
                )

            for value in possible_ids:

                extracted_ids = re.findall(
                    r"\bNCT\d{8}\b",
                    value.upper()
                )

                signal_nct_ids.extend(extracted_ids)

        signal_nct_ids = list(dict.fromkeys(signal_nct_ids))

        # Save episodic memory separately for each actual NCT ID
        if signal_nct_ids:

            for nct_id in signal_nct_ids:

                related_signals = [
                    signal
                    for signal in signals_found
                    if nct_id in (
                        [signal.get("nct_id", "").upper()]
                        + [
                            str(item).upper()
                            for item in signal.get("evidence", [])
                        ]
                    )
                ]

                episode_content = (
                    f"Task: {task}. "
                    f"Found {len(related_signals)} low credibility "
                    f"signal(s) associated with {nct_id}. "
                    f"Sponsors/signals: "
                    f"{[signal.get('summary') for signal in related_signals]}"
                )

                await _episodic.save_episode(
                    agent_name=AGENT_NAME,
                    content=episode_content,
                    nct_id=nct_id,
                    outcome="signal_generated",
                )

        else:

            episode_content = (
                f"Task: {task}. "
                f"Found {len(signals_found)} low credibility signals. "
                f"Sponsors/signals: "
                f"{[signal.get('summary') for signal in signals_found]}. "
                f"No trial-specific NCT ID was available."
            )

            await _episodic.save_episode(
                agent_name=AGENT_NAME,
                content=episode_content,
                outcome=(
                    "signal_generated"
                    if signals_found
                    else "no_signal"
                ),
            )

        logger.info(
            f"{AGENT_NAME} | Complete "
            f"| signals_found={len(signals_found)}"
        )

        # Update graph state
        return {
            "signals": (
                state.get("signals", []) + signals_found
            ),
            "agents_activated": (
                state.get("agents_activated", [])
                + [AGENT_NAME]
            ),
        }

    except Exception as e:

        logger.error(
            f"{AGENT_NAME} | Error | {e}"
        )

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
    """Extract low credibility signals from LLM output."""

    if not response_text:
        return []

    if "NO_SIGNALS_FOUND" in response_text:
        return []

    pattern = re.compile(
        r'\{[^{}]*"signal_type"[^{}]*\}',
        re.DOTALL
    )

    signals = []

    for match in pattern.findall(response_text):

        try:
            data = json.loads(match)

            # Accept only this agent's signal type
            if data.get("signal_type") != SIGNAL_TYPE:
                continue

            confidence = float(
                data.get("confidence", 0.5)
            )

            confidence = max(
                0.0,
                min(1.0, confidence)
            )

            signal: SignalOutput = {
                "agent": agent_name,
                "signal_type": SIGNAL_TYPE,
                "nct_id": data.get("nct_id") or None,
                "summary": data.get("summary", ""),
                "evidence": data.get("evidence", []),
                "confidence": confidence,
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
    """Execute an asynchronous LangChain tool."""

    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})

    tool_func = next(
        (
            tool
            for tool in available_tools
            if tool.name == tool_name
        ),
        None
    )

    if tool_func is None:

        logger.error(
            f"Tool not found | name={tool_name}"
        )

        return f"Error: Tool '{tool_name}' not found."

    try:

        result = await tool_func.ainvoke(tool_args)

        if isinstance(result, str):
            return result

        return json.dumps(
            result,
            default=str
        )

    except Exception as e:

        logger.error(
            f"Tool execution failed "
            f"| tool={tool_name} | error={e}"
        )

        return json.dumps({
            "error": str(e),
            "tool": tool_name,
        })
