import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

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
    get_study_amendments,
)

from config.settings import settings
from config.logging_config import setup_logging


logger = setup_logging(__name__)

AGENT_NAME = "broken_promises_agent"
SIGNAL_TYPE = "broken_promise"

MAX_ITERATIONS = 10

AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
    fetch_study_details,
    get_study_amendments,
]

_procedural = ProceduralStore()
_episodic = EpisodicStore()

_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0.1,
    api_key=settings.openai_api_key,
).bind_tools(AGENT_TOOLS)


async def broken_promises_node(state: MosaicState) -> dict:
    """Detect outcome switching in clinical trials."""

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        # --------------------------------------------------
        # 1. LOAD PROCEDURAL MEMORY
        # --------------------------------------------------

        procedures = await _procedural.get_procedures(
            AGENT_NAME
        )

        procedures_text = "\n".join(
            f"- {rule}" for rule in procedures
        )

        # --------------------------------------------------
        # 2. SYSTEM PROMPT
        # --------------------------------------------------

        system_prompt = f"""
You are the Broken Promises Agent for MOSAIC.

MISSION:
Detect outcome switching — cases where a clinical trial
changed its primary outcome measure AFTER the study began
enrolling participants.

REASONING RULES:
{procedures_text}

WORKFLOW:
1. Search past episodes for similar cases.
2. Search for studies with protocol amendments.
3. Fetch amendment history for suspicious studies.
4. Compare original and updated primary outcomes.
5. Check whether the change happened after enrollment began.
6. Generate signals for confirmed cases.

IMPORTANT:
- Do not assume an outcome changed just because an amendment exists.
- Compare the original and amended primary outcome measures.
- Establish the timing of the amendment relative to enrollment.
- Do not invent missing dates, outcomes, or evidence.
- If amendment history is unavailable, report insufficient evidence.
- Distinguish confirmed outcome switching from an unexplained
  or ambiguous protocol change.

CONFIDENCE SCORING:
- 0.9+ : Primary outcome changed after >50% enrollment,
         with no justification.
- 0.8  : Primary outcome changed mid-study, with weak justification.
- 0.7  : Secondary outcome changed to primary after enrollment.
- 0.6  : Measurement method changed.
- Below 0.6: Ambiguous — send to human review.

OUTPUT FORMAT:
Return valid JSON for each confirmed signal:

{{
    "nct_id": "NCT_ID",
    "signal_type": "broken_promise",
    "summary": "What changed, when, and why it matters",
    "evidence": [
        "original outcome",
        "new outcome",
        "timing"
    ],
    "confidence": 0.85
}}

If no violations are found, return:
NO_SIGNALS_FOUND

If evidence is insufficient, do not classify the study
as a confirmed violation.
"""

        # --------------------------------------------------
        # 3. USER TASK
        # --------------------------------------------------

        task = state.get(
            "task",
            "Find trials with outcome switching"
        )

        nct_ids = state.get("nct_ids", [])

        human_message = f"""
ANALYSIS TASK: {task}

SPECIFIC STUDIES:
{nct_ids if nct_ids else "Search broadly"}

Investigate now.

Focus on protocol amendments and primary outcome changes.
Use the available tools to collect evidence before
generating any signal.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]

        signals_found = []
        analysis_completed = False

        # --------------------------------------------------
        # 4. ASYNC TOOL-CALLING LOOP
        # --------------------------------------------------

        for iteration in range(MAX_ITERATIONS):

            logger.info(
                f"{AGENT_NAME} | "
                f"LLM iteration {iteration + 1}/{MAX_ITERATIONS}"
            )

            response = await _llm.ainvoke(messages)

            # Preserve the complete AI response, including
            # its tool calls and tool-call IDs.
            messages.append(response)

            # --------------------------------------------------
            # 5. CHECK WHETHER THE AGENT HAS FINISHED
            # --------------------------------------------------

            if not response.tool_calls:

                logger.info(
                    f"{AGENT_NAME} | "
                    f"Analysis completed | "
                    f"iteration={iteration + 1}"
                )

                signals_found = _parse_signals(
                    response.content,
                    AGENT_NAME
                )

                analysis_completed = True
                break

            # --------------------------------------------------
            # 6. EXECUTE REQUESTED TOOLS
            # --------------------------------------------------

            for tool_call in response.tool_calls:

                result = await _execute_tool(
                    tool_call,
                    AGENT_TOOLS
                )

                # ToolMessage must contain the corresponding
                # tool_call_id from the assistant response.
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    )
                )

        # --------------------------------------------------
        # 7. HANDLE ITERATION LIMIT
        # --------------------------------------------------

        if not analysis_completed:

            logger.warning(
                f"{AGENT_NAME} | "
                f"Maximum iterations reached without completion"
            )

            return {
                "error_log": state.get("error_log", []) + [
                    (
                        f"{AGENT_NAME}: Maximum iterations "
                        f"reached before analysis completed."
                    )
                ],
                "agents_activated": (
                    state.get("agents_activated", [])
                    + [AGENT_NAME]
                ),
            }

        # --------------------------------------------------
        # 8. SAVE INVESTIGATION TO EPISODIC MEMORY
        # --------------------------------------------------

        # Get unique NCT IDs from generated signals.
        signal_nct_ids = list(dict.fromkeys(
            signal.get("nct_id")
            for signal in signals_found
            if signal.get("nct_id")
        ))

        if signal_nct_ids:

            # Save one episode for each identified trial.
            for nct_id in signal_nct_ids:

                trial_signals = [
                    signal
                    for signal in signals_found
                    if signal.get("nct_id") == nct_id
                ]

                await _episodic.save_episode(
                    agent_name=AGENT_NAME,
                    content=(
                        f"Task: {task}. "
                        f"Trial: {nct_id}. "
                        f"Found {len(trial_signals)} "
                        f"broken promise signals."
                    ),
                    nct_id=nct_id,
                    outcome="signal_generated",
                )

        else:

            # No trial-specific signal exists.
            # Save a general investigation episode.
            await _episodic.save_episode(
                agent_name=AGENT_NAME,
                content=(
                    f"Task: {task}. "
                    f"Found {len(signals_found)} "
                    f"broken promise signals."
                ),
                outcome=(
                    "signal_generated"
                    if signals_found
                    else "no_signal"
                ),
            )

        # --------------------------------------------------
        # 9. LOG RESULTS
        # --------------------------------------------------

        logger.info(
            f"{AGENT_NAME} | Complete | "
            f"signals_found={len(signals_found)}"
        )

        # --------------------------------------------------
        # 10. RETURN GRAPH STATE
        # --------------------------------------------------

        return {
            "signals": (
                state.get("signals", [])
                + signals_found
            ),
            "agents_activated": (
                state.get("agents_activated", [])
                + [AGENT_NAME]
            ),
        }

    except Exception as e:

        logger.exception(
            f"{AGENT_NAME} | Analysis failed"
        )

        return {
            "error_log": (
                state.get("error_log", [])
                + [f"{AGENT_NAME}: {str(e)}"]
            ),
            "agents_activated": (
                state.get("agents_activated", [])
                + [AGENT_NAME]
            ),
        }


# ==========================================================
# SIGNAL PARSER
# ==========================================================

def _parse_signals(
    response_text: str,
    agent_name: str
) -> list[SignalOutput]:

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

            confidence = float(
                data.get("confidence", 0.5)
            )

            signals.append({
                "agent": agent_name,
                "signal_type": data.get(
                    "signal_type",
                    SIGNAL_TYPE
                ),
                "nct_id": data.get("nct_id", ""),
                "summary": data.get("summary", ""),
                "evidence": data.get("evidence", []),
                "confidence": confidence,
            })

        except (json.JSONDecodeError, ValueError) as e:

            logger.warning(
                f"{AGENT_NAME} | "
                f"Could not parse signal JSON | {e}"
            )

            continue

    return signals


# ==========================================================
# TOOL EXECUTION
# ==========================================================

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

        logger.error(
            f"{AGENT_NAME} | "
            f"Tool not found | name={tool_name}"
        )

        return f"Error: Tool '{tool_name}' not found."

    try:

        logger.info(
            f"{AGENT_NAME} | "
            f"Executing tool | name={tool_name}"
        )

        result = await tool_func.ainvoke(tool_args)

        return str(result)

    except Exception as e:

        logger.exception(
            f"{AGENT_NAME} | "
            f"Tool execution failed | name={tool_name}"
        )

        return f"Error executing '{tool_name}': {str(e)}"