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
)

from tools.clinical_tools import (
    check_results_posted,
    fetch_study_details,
)

from config.settings import settings
from config.logging_config import setup_logging


logger = setup_logging(__name__)

AGENT_NAME = "missing_results_agent"
SIGNAL_TYPE = "missing_results"
MAX_ITERATIONS = 10


AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
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


async def _verify_missing_results(
    nct_ids: list
) -> list[SignalOutput]:
    """Deterministically verify supplied trials using live registry data."""

    signals = []

    for nct_id in nct_ids:

        logger.info(
            f"{AGENT_NAME} | Mandatory results verification "
            f"| nct_id={nct_id}"
        )

        try:
            raw_result = await check_results_posted.ainvoke({
                "nct_id": nct_id
            })

            result = (
                json.loads(raw_result)
                if isinstance(raw_result, str)
                else raw_result
            )

            if not isinstance(result, dict):
                logger.warning(
                    f"Invalid verification response | nct_id={nct_id}"
                )
                continue

            # Require verified live registry evidence.
            if (
                result.get("found") is not True
                or result.get("status") != "COMPLETED"
                or result.get("results_posted") is not False
                or result.get("is_violation") is not True
            ):
                logger.info(
                    f"No verified missing-results signal "
                    f"| nct_id={nct_id}"
                )
                continue

            primary_date = result.get("primary_completion_date")
            months_overdue = result.get("months_overdue")

            evidence = [
                "Source: live ClinicalTrials.gov study record",
                "Study status: COMPLETED",
                "Results posted: false",
                f"Primary completion date: {primary_date}",
                (
                    "Reporting deadline basis: "
                    f"{result.get('reporting_deadline_basis')}"
                ),
                (
                    "Months beyond 12-month screening threshold: "
                    f"{months_overdue}"
                ),
            ]

            if result.get("study_completion_date"):
                evidence.append(
                    "Overall study completion date: "
                    f"{result['study_completion_date']}"
                )

            if result.get("sponsor"):
                evidence.append(
                    f"Sponsor: {result['sponsor']}"
                )

            signals.append({
                "agent": AGENT_NAME,
                "signal_type": SIGNAL_TYPE,
                "nct_id": nct_id,
                "summary": (
                    f"Completed clinical trial {nct_id} has not posted "
                    f"results and is {months_overdue} months beyond "
                    "the 12-month screening threshold."
                ),
                "evidence": evidence,
                "confidence": 0.95,
            })

            logger.info(
                f"{AGENT_NAME} | Verified signal generated "
                f"| nct_id={nct_id}"
            )

        except Exception as e:
            logger.exception(
                f"{AGENT_NAME} | Verification failed "
                f"| nct_id={nct_id} | error={e}"
            )

    return signals


async def missing_results_node(state: MosaicState) -> dict:
    """Find completed clinical trials that have not posted results."""

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        # Load agent-specific procedural rules.
        procedures = await _procedural.get_procedures(AGENT_NAME)

        procedures_text = "\n".join(
            f"- {rule}" for rule in procedures
        )

        task = state.get(
            "task",
            "Find completed trials with missing results"
        )

        nct_ids = state.get("nct_ids", [])

        # --------------------------------------------------
        # MANDATORY VERIFICATION FOR SUPPLIED TRIAL IDS
        # --------------------------------------------------

        if nct_ids:

            signals_found = await _verify_missing_results(nct_ids)

            # Save one episode per supplied trial ID.
            for nct_id in nct_ids:
                trial_signals = [
                    signal
                    for signal in signals_found
                    if signal.get("nct_id") == nct_id
                ]

                await _episodic.save_episode(
                    agent_name=AGENT_NAME,
                    content=(
                        f"Task: {task}. "
                        f"Verified trial: {nct_id}. "
                        f"Signal generated: {bool(trial_signals)}. "
                        f"Signals: "
                        f"{[signal.get('summary') for signal in trial_signals]}"
                    ),
                    nct_id=nct_id,
                    outcome=(
                        "signal_generated"
                        if trial_signals
                        else "no_signal"
                    ),
                )

            logger.info(
                f"{AGENT_NAME} | Positive-control verification complete "
                f"| signals_found={len(signals_found)}"
            )

            return {
                "signals": state.get("signals", []) + signals_found,
                "agents_activated": (
                    state.get("agents_activated", [])
                    + [AGENT_NAME]
                ),
            }

        # --------------------------------------------------
        # EXISTING LLM WORKFLOW FOR BROAD INVESTIGATIONS
        # --------------------------------------------------

        system_prompt = f"""
You are the Missing Results Agent for MOSAIC.

MISSION:
Find completed clinical trials that have never posted their results.

REASONING RULES:
{procedures_text}

WORKFLOW:
1. Search past episodes for relevant investigations.
2. Search for completed studies with missing results.
3. Verify suspicious studies using the live ClinicalTrials.gov API.
4. Check the sponsor profile.
5. Generate signals with confidence scores.
6. Update the sponsor profile when appropriate.
7. Save the session as an episodic memory.

VERIFICATION REQUIREMENTS:
- Only investigate completed clinical trials.
- Verify the current results-posted status using the live API.
- Do not assume that a completed trial has missing results.
- Distinguish between results not posted and results unavailable.
- Use the actual study completion date and results-posting status.
- Do not invent dates, evidence, or sponsor information.
- Do not generate a signal when evidence is insufficient.
- If the live API cannot verify a study, report insufficient evidence.
- Do not treat API failures as proof of missing results.
- Always distinguish primary completion date from overall study completion date.
- The results reporting deadline calculation must use the actual
  primary completion date.
- Never describe the overall study completion date as the date
  from which the reporting deadline was calculated.
- When both dates are available, include both in the evidence.
- State the overdue duration as time beyond the 12-month screening
  threshold, not simply time since overall study completion.
- Describe this as a screening flag, not a confirmed legal violation.

OUTPUT FORMAT:
Return valid JSON objects for detected signals:

{{
    "nct_id": "NCT_ID",
    "signal_type": "missing_results",
    "summary": "Description",
    "evidence": ["fact 1", "fact 2"],
    "confidence": 0.85
}}

If no verified signals are found, return:
NO_SIGNALS_FOUND

Return the final result only after completing the investigation.
"""

        human_message = f"""
ANALYSIS TASK: {task}

SPECIFIC STUDIES:
Search broadly

Begin the investigation.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]

        signals_found = []
        analysis_completed = False

        # LLM tool-calling loop.
        for iteration in range(MAX_ITERATIONS):

            logger.info(
                f"{AGENT_NAME} | LLM iteration "
                f"{iteration + 1}/{MAX_ITERATIONS}"
            )

            response = await _llm.ainvoke(messages)

            messages.append(response)

            # LLM has completed its investigation.
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

            # Execute requested tools.
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

        if not analysis_completed:
            raise RuntimeError(
                f"{AGENT_NAME} reached maximum LLM iterations "
                "without completing analysis."
            )

        # Save episodic memory.
        episode_content = (
            f"Task: {task}. "
            f"Found {len(signals_found)} signals. "
            f"Signals: "
            f"{[signal.get('nct_id') for signal in signals_found]}"
        )

        # Link each signal-bearing episode to its trial.
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

                await _episodic.save_episode(
                    agent_name=AGENT_NAME,
                    content=(
                        f"Task: {task}. "
                        f"Trial: {nct_id}. "
                        f"Signals: "
                        f"{[signal.get('summary') for signal in trial_signals]}"
                    ),
                    nct_id=nct_id,
                    outcome="signal_generated",
                )
        else:
            await _episodic.save_episode(
                agent_name=AGENT_NAME,
                content=episode_content,
                outcome=(
                    "signal_generated"
                    if signals_found
                    else "no_signal"
                ),
            )

        current_signals = state.get("signals", [])
        current_activated = state.get("agents_activated", [])

        logger.info(
            f"{AGENT_NAME} | Complete "
            f"| signals_found={len(signals_found)}"
        )

        return {
            "signals": current_signals + signals_found,
            "agents_activated": current_activated + [
                AGENT_NAME
            ],
        }

    except Exception as e:

        logger.exception(
            f"{AGENT_NAME} | Error | {e}"
        )

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
    """Extract signal JSON objects from the LLM response."""

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
                "nct_id": data.get("nct_id", ""),
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