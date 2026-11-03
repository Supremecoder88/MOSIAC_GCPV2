
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
MAX_ITERATIONS = 10


AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
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

    unverified_history_ids = set()

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        # Load procedural rules
        procedures = await _procedural.get_procedures(AGENT_NAME)

        procedures_text = "\n".join(
            f"- {rule}" for rule in procedures
        )

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
3. Fetch current study details.
4. Check amendment history for an official extension.
5. Flag delays greater than 180 days only when evidence supports it.
6. Check the sponsor profile for repeated delays.
7. Update sponsor profiles when appropriate.

SILENCE THRESHOLD:
Only flag delays greater than 180 days past the completion date
when the absence of a relevant amendment has been verified.

EVIDENCE REQUIREMENTS:
- Use the actual completion date from the study record.
- Distinguish estimated completion dates from actual completion dates.
- Use the current date when calculating delays.
- Check amendment history before concluding that no amendment was filed.
- If amendment history is unavailable, report insufficient evidence.
- A failed API request is NOT proof that no amendment exists.
- Do not invent completion dates, extensions, or sponsor history.
- Consider whether the study has posted results.
- Distinguish an ongoing trial from a completed trial.
- Consider possible explanations such as COVID-related delays
  only when supporting evidence exists.
- Never generate a timeline signal for a study whose amendment
  history could not be retrieved and verified.

CONFIDENCE SCORING:
- 0.9+ : 2+ years delayed, verified absence of relevant amendment,
         no results, repeat offender.
- 0.8  : 1-2 years delayed, verified absence of relevant amendment,
         prior sponsor delays.
- 0.7  : 6-12 months delayed, verified absence of relevant amendment.
- 0.6  : Borderline timing — send to human review.
- Do not assign confidence when evidence is insufficient.

OUTPUT FORMAT:
Return valid JSON objects for detected timeline delays:

{{
    "nct_id": "NCT_ID",
    "signal_type": "timeline_delay",
    "summary": "How long delayed and whether an amendment was filed",
    "evidence": ["completion date", "current date", "amendment status"],
    "confidence": 0.80
}}

If no verified timeline violations are found, return:
NO_SIGNALS_FOUND

Return the final result only after completing the investigation.
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
                tool_args = tool_call.get("args", {})

                logger.info(
                    f"{AGENT_NAME} | Executing tool "
                    f"| name={tool_name}"
                )

                result = await _execute_tool(
                    tool_call,
                    AGENT_TOOLS
                )

                # Deterministically detect unavailable history
                if tool_name == "get_study_amendments":

                    try:
                        history_data = (
                            json.loads(result)
                            if isinstance(result, str)
                            else result
                        )

                        if isinstance(history_data, dict):

                            history_unavailable = (
                                history_data.get(
                                    "verification_status"
                                ) == "unavailable"
                                or history_data.get(
                                    "history_available"
                                ) is False
                            )

                            if history_unavailable:

                                nct_id = (
                                    history_data.get("nct_id")
                                    or tool_args.get("nct_id")
                                )

                                if nct_id:
                                    unverified_history_ids.add(nct_id)

                                    logger.warning(
                                        f"{AGENT_NAME} | Amendment history "
                                        f"unavailable | nct_id={nct_id}. "
                                        f"Signals for this study will be suppressed."
                                    )

                    except (json.JSONDecodeError, TypeError, AttributeError) as e:
                        logger.warning(
                            f"{AGENT_NAME} | Could not inspect amendment "
                            f"history result | {e}"
                        )

                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )

        # Do not save incomplete investigations as no_signal
        if not analysis_completed:
            raise RuntimeError(
                f"{AGENT_NAME} reached maximum LLM iterations "
                "without completing analysis."
            )

        # Remove signals for studies with unavailable history
        if unverified_history_ids:

            original_count = len(signals_found)

            signals_found = [
                signal
                for signal in signals_found
                if signal.get("nct_id")
                not in unverified_history_ids
            ]

            suppressed_count = original_count - len(signals_found)

            if suppressed_count:
                logger.warning(
                    f"{AGENT_NAME} | Suppressed {suppressed_count} "
                    f"unverified timeline signal(s)"
                )

        # Save episodic memory separately for each NCT ID
        signal_nct_ids = list(dict.fromkeys(
            signal.get("nct_id")
            for signal in signals_found
            if signal.get("nct_id")
        ))

        if signal_nct_ids:

            for nct_id in signal_nct_ids:

                trial_signals = [
                    signal
                    for signal in signals_found
                    if signal.get("nct_id") == nct_id
                ]

                episode_content = (
                    f"Task: {task}. "
                    f"Found {len(trial_signals)} timeline delay "
                    f"signal(s) for {nct_id}. "
                    f"Details: "
                    f"{[signal.get('summary') for signal in trial_signals]}"
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
                f"Found {len(signals_found)} timeline delay signals. "
                f"No trial-specific NCT ID was available."
            )

            await _episodic.save_episode(
                agent_name=AGENT_NAME,
                content=episode_content,
                outcome="no_signal",
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
    """Extract timeline delay signals from LLM output."""

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

            evidence = data.get("evidence", [])

            if not isinstance(evidence, list):
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
                "nct_id": data.get("nct_id", ""),
                "summary": data.get("summary", ""),
                "evidence": evidence,
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
