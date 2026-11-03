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
MAX_ITERATIONS = 10


AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
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
        # Load procedural rules
        procedures = await _procedural.get_procedures(AGENT_NAME)

        procedures_text = "\n".join(
            f"- {rule}" for rule in procedures
        )

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
3. Retrieve official clinical trial details.
4. Retrieve publications associated with the trial.
5. Use compare_filing_vs_papers to compare the evidence.
6. Identify meaningful differences in reported safety outcomes.
7. Generate a signal only when an actual discrepancy is supported
   by comparable evidence from both sources.
8. If official safety data is unavailable, classify the comparison
   as insufficient evidence.

LOOK FOR:
- Official filing explicitly reports zero serious adverse events,
  while a publication reports serious adverse events for a
  comparable population and reporting period.
- Official filing reports a lower event severity than a publication,
  with evidence supporting the comparison.
- Official filing and publication report conflicting safety counts
  for the same population, treatment arm, and reporting period.
- Published safety findings that materially contradict explicitly
  reported official safety findings.

EVIDENCE REQUIREMENTS:
- Compare information from the same clinical trial.
- Verify that the publication actually concerns the trial.
- Distinguish serious adverse events from ordinary adverse events.
- Verify reporting periods, participant populations, treatment arms,
  and denominators whenever available.
- Do not assume that different terminology means contradictory findings.
- Do not interpret missing registry data as zero adverse events.
- Do not interpret results_posted=false as evidence of zero events.
- Do not interpret an absent adverse-event count as a negative finding.
- A publication reporting adverse events does not automatically
  establish a discrepancy with the official registry.
- Require explicit official safety information and publication
  safety information before generating a discrepancy signal.
- If official adverse-event counts or relevant safety findings
  are unavailable, report insufficient evidence.
- Do not invent adverse events, numbers, dates, or publication findings.
- If either source is unavailable, report insufficient evidence.
- Do not generate a signal based only on missing official results.
- Do not generate a signal without comparable evidence from both sources.
- Treat screening findings as potential discrepancies, not proven
  misconduct or confirmed reporting violations.

CONFIDENCE SCORING:
- 0.90+ : Explicit, directly comparable contradiction in both sources.
- 0.80  : Strong discrepancy with minor differences in reporting scope.
- 0.70  : Potential discrepancy with some comparable evidence.
- 0.60  : Limited evidence requiring human review.
- No signal: Official safety data is missing or insufficient.

OUTPUT FORMAT:
Return valid JSON objects for detected safety gaps:

{{
    "nct_id": "NCT_ID",
    "signal_type": "safety_gap",
    "summary": "What the filing said vs what papers reported",
    "evidence": ["filing claim", "paper contradiction"],
    "confidence": 0.80
}}

If no verified gaps are found, return:
NO_SIGNALS_FOUND

Return the final result only after completing the investigation.
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

        # Do not save incomplete investigations as no_signal
        if not analysis_completed:
            raise RuntimeError(
                f"{AGENT_NAME} reached maximum LLM iterations "
                "without completing the investigation."
            )

        # --------------------------------------------------
        # SAVE INVESTIGATION TO EPISODIC MEMORY
        # --------------------------------------------------

        # Collect unique NCT IDs from generated signals.
        signal_nct_ids = list(dict.fromkeys(
            signal.get("nct_id")
            for signal in signals_found
            if signal.get("nct_id")
        ))

        if signal_nct_ids:

            # Save one episode per identified trial.
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
                        f"Found {len(trial_signals)} safety gap signals. "
                        f"Signals: "
                        f"{[signal.get('summary') for signal in trial_signals]}"
                    ),
                    nct_id=nct_id,
                    outcome="signal_generated",
                )

        else:

            # Save a general episode if no trial-specific
            # signal was generated.
            episode_content = (
                f"Task: {task}. "
                f"Found {len(signals_found)} safety gap signals. "
                f"Signals: "
                f"{[signal.get('nct_id') for signal in signals_found]}"
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
    """Extract safety gap signals from LLM output."""

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
                f"Could not parse safety gap JSON: {e}"
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