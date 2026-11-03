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
MAX_ITERATIONS = 12


AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
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
    temperature=0.1,
    api_key=settings.openai_api_key,
).bind_tools(AGENT_TOOLS)


async def pattern_finder_node(state: MosaicState) -> dict:
    """Detect patterns across multiple clinical trials."""

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        # Load procedural rules
        procedures = await _procedural.get_procedures(AGENT_NAME)

        procedures_text = "\n".join(
            f"- {rule}" for rule in procedures
        )

        system_prompt = f"""
You are the Pattern Finder Agent for MOSAIC.

MISSION:
Detect genuine research patterns spanning multiple clinical trials.

REASONING RULES:
{procedures_text}

WORKFLOW:
1. Search past episodes for previously identified patterns.
2. Search studies by condition, drug, or sponsor.
3. Look for the same drug across multiple sponsors.
4. Look for failed drugs being retried in another area.
5. Look for sponsors with repeated compliance issues.
6. Use PubMed to find papers connecting multiple trials.
7. A meaningful pattern requires at least 3 distinct studies.

STRICT PATTERN VALIDATION:
- All studies in a pattern MUST share one specific, identifiable factor.
- Valid shared factors include the same named drug, intervention,
  medical condition, sponsor, or clearly defined outcome.
- Every study must independently support the same shared factor.
- Similar words such as "adverse events", "treatment", "cancer",
  or "clinical trial" are NOT sufficient shared factors.
- Do not combine unrelated diseases or treatments just to reach
  three studies.
- Do not create a connection based only on broad terminology.
- Do not infer a shared sponsor, drug, or condition without evidence.
- Each study must be identifiable by its actual NCT ID.
- Every NCT ID must appear in the retrieved tool results.
- If three studies do not share the same specific factor,
  return NO_SIGNALS_FOUND.

CONFIDENCE SCORING:
- 0.9+ : 5+ studies, same sponsor, clear systematic pattern.
- 0.8  : 3-5 studies, specific shared factor and strong evidence.
- 0.7  : 3 studies, specific shared factor and credible evidence.
- Below 0.65: Do not generate a signal.

EVIDENCE REQUIREMENTS:
- Identify at least 3 distinct NCT IDs.
- Provide one evidence entry for each study.
- Each evidence entry MUST contain its NCT ID.
- Each evidence entry MUST explicitly support the exact shared factor.
- The shared_factor must be specific, not a broad generic term.
- Distinguish correlation from causation.
- Do not invent study IDs, findings, or sponsor relationships.
- Use retrieved study and publication evidence only.

OUTPUT FORMAT:
Return valid JSON objects for detected patterns:

{{
    "nct_id": "PATTERN",
    "signal_type": "cross_study_pattern",
    "summary": "Explain the genuine connection across studies",
    "shared_factor": "One specific shared drug, condition, sponsor, intervention, or outcome",
    "study_ids": ["NCT00000001", "NCT00000002", "NCT00000003"],
    "evidence": [
        "NCT00000001: Evidence supporting the shared factor",
        "NCT00000002: Evidence supporting the shared factor",
        "NCT00000003: Evidence supporting the shared factor"
    ],
    "confidence": 0.80
}}

If the evidence does not meet EVERY requirement, return:
NO_SIGNALS_FOUND

Return the final result only after completing the investigation.
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
Do not generate a pattern unless all strict validation requirements
are satisfied.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]

        signals_found = []
        retrieved_study_ids = set()
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
                    AGENT_NAME,
                    retrieved_study_ids
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

                # Record study IDs actually returned by tools
                if isinstance(result, str):
                    found_ids = re.findall(
                        r"\bNCT\d{8}\b",
                        result.upper()
                    )

                    retrieved_study_ids.update(found_ids)

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

        # Pattern signals use nct_id="PATTERN".
        # Extract actual trial IDs from their validated evidence.
        pattern_nct_ids = list(dict.fromkeys(
            nct_id
            for signal in signals_found
            for item in signal.get("evidence", [])
            if isinstance(item, str)
            for nct_id in re.findall(
                r"\bNCT\d{8}\b",
                item.upper()
            )
        ))

        if pattern_nct_ids:

            # Save one episode for every trial involved in a pattern.
            for nct_id in pattern_nct_ids:

                related_patterns = [
                    signal
                    for signal in signals_found
                    if any(
                        isinstance(item, str)
                        and nct_id in item.upper()
                        for item in signal.get("evidence", [])
                    )
                ]

                await _episodic.save_episode(
                    agent_name=AGENT_NAME,
                    content=(
                        f"Task: {task}. "
                        f"Cross-study pattern involving {nct_id}. "
                        f"Patterns: "
                        f"{[signal.get('summary') for signal in related_patterns]}"
                    ),
                    nct_id=nct_id,
                    outcome="signal_generated",
                )

        else:

            # No trial-specific evidence available.
            await _episodic.save_episode(
                agent_name=AGENT_NAME,
                content=(
                    f"Task: {task}. "
                    f"Found {len(signals_found)} cross-study pattern signals. "
                    f"Patterns: "
                    f"{[signal.get('summary') for signal in signals_found]}"
                ),
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
    agent_name: str,
    retrieved_study_ids: set | None = None
) -> list[SignalOutput]:
    """
    Extract and validate cross-study pattern signals.

    A signal is accepted only when:
    - At least 3 distinct study IDs are provided.
    - All IDs appeared in retrieved tool results.
    - A specific shared factor is provided.
    - Every study has an evidence entry containing its ID
      and the stated shared factor.
    """

    if not response_text:
        return []

    if "NO_SIGNALS_FOUND" in response_text:
        return []

    retrieved_study_ids = retrieved_study_ids or set()

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
            study_ids = data.get("study_ids", [])
            shared_factor = data.get("shared_factor", "")

            if not isinstance(evidence, list):
                continue

            if not isinstance(study_ids, list):
                continue

            if not isinstance(shared_factor, str):
                continue

            shared_factor = shared_factor.strip()

            if not shared_factor:
                logger.warning(
                    "Rejected pattern: missing shared factor."
                )
                continue

            # Require at least 3 distinct NCT IDs
            normalized_ids = {
                str(nct_id).strip().upper()
                for nct_id in study_ids
                if isinstance(nct_id, str)
                and re.fullmatch(
                    r"NCT\d{8}",
                    nct_id.strip().upper()
                )
            }

            if len(normalized_ids) < 3:
                logger.warning(
                    "Rejected pattern: fewer than 3 distinct "
                    "valid study IDs."
                )
                continue

            # All IDs must have appeared in actual tool results
            if not normalized_ids.issubset(retrieved_study_ids):
                logger.warning(
                    "Rejected pattern: one or more study IDs "
                    "were not present in retrieved tool results."
                )
                continue

            # Every study needs its own evidence entry
            valid_evidence = True

            for nct_id in normalized_ids:

                matching_evidence = [
                    item for item in evidence
                    if isinstance(item, str)
                    and nct_id in item.upper()
                    and shared_factor.casefold() in item.casefold()
                ]

                if not matching_evidence:
                    valid_evidence = False
                    logger.warning(
                        f"Rejected pattern: no evidence explicitly "
                        f"supports shared factor for {nct_id}."
                    )
                    break

            if not valid_evidence:
                continue

            confidence = float(
                data.get("confidence", 0.5)
            )

            confidence = max(
                0.0,
                min(1.0, confidence)
            )

            if confidence < 0.65:
                logger.warning(
                    "Rejected pattern: confidence below 0.65."
                )
                continue

            signal: SignalOutput = {
                "agent": agent_name,
                "signal_type": SIGNAL_TYPE,
                "nct_id": "PATTERN",
                "summary": data.get("summary", ""),
                "evidence": evidence,
                "confidence": confidence,
            }

            signals.append(signal)

        except (json.JSONDecodeError, ValueError, TypeError) as e:

            logger.warning(
                f"Could not parse pattern JSON: {e}"
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