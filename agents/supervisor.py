import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import MosaicState, SignalOutput
from config.settings import settings
from config.logging_config import setup_logging


logger = setup_logging(__name__)

llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0.1,
    api_key=settings.openai_api_key,
)


async def supervisor_route(state: MosaicState) -> dict:
    """Initialize the analysis run."""

    run_id = str(uuid.uuid4())

    logger.info(
        f"Supervisor routing | "
        f"run_id={run_id} | "
        f"task='{state.get('task', '')[:80]}'"
    )

    return {
        "run_id": run_id,
        "agents_activated": [],
        "signals": [],
        "run_complete": False,
        "error_log": [],
    }


async def supervisor_compile(state: MosaicState) -> dict:
    """Compile all agent signals into the final intelligence brief."""

    signals = state.get("signals", [])
    agents_activated = state.get("agents_activated", [])
    task = state.get("task", "")

    logger.info(
        f"Supervisor compiling brief | "
        f"signals={len(signals)} | "
        f"agents_activated={len(agents_activated)}"
    )

    if not signals:
        return {
            "final_brief": (
                "**EXECUTIVE SUMMARY:** Analysis complete. "
                "No significant research integrity signals were detected "
                "for the specified task and study set."
            ),
            "run_complete": True,
            "agents_activated": agents_activated,
        }

    high_confidence_signals = [
        signal
        for signal in signals
        if signal.get("confidence", 0) >= 0.6
    ]

    review_signals = [
        signal
        for signal in signals
        if signal.get("confidence", 0) < 0.6
    ]

    signals_text = _format_signals_for_llm(signals)

    system_prompt = """
You are the Chief Intelligence Officer of MOSAIC.

Compile a professional intelligence brief from signals generated
by specialist AI agents.

BRIEF FORMAT:
1. EXECUTIVE SUMMARY
2. SIGNALS BY PRIORITY
3. SIGNALS REQUIRING HUMAN REVIEW
4. PIPELINE HEALTH

For each signal include:
- What was found
- Why it matters
- What action to take

TONE:
Professional, factual, and actionable.
Include NCT IDs, sponsor names, and relevant timeframes.
"""

    human_prompt = f"""
ANALYSIS TASK:
{task}

SIGNALS FOUND:
{signals_text}

HIGH CONFIDENCE SIGNALS: {len(high_confidence_signals)}
SIGNALS REQUIRING REVIEW: {len(review_signals)}
AGENTS ACTIVATED: {', '.join(agents_activated)}

Compile the final intelligence brief.
"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])

        final_brief = response.content

        logger.info(
            f"Brief compiled successfully | "
            f"signals_included={len(signals)}"
        )

    except Exception as e:
        logger.error(
            f"LLM brief compilation failed | error={e}"
        )

        final_brief = _fallback_brief(
            signals,
            agents_activated,
            task,
        )

    return {
        "final_brief": final_brief,
        "run_complete": True,
        "total_signals": len(signals),
        "signals_requiring_review": len(review_signals),
        "agents_activated": agents_activated,
    }


def _format_signals_for_llm(
    signals: list[SignalOutput]
) -> str:

    if not signals:
        return "No signals generated."

    lines = []

    for i, signal in enumerate(signals, start=1):
        lines.append(f"SIGNAL {i}:")
        lines.append(
            f"  Agent: {signal.get('agent', 'unknown')}"
        )
        lines.append(
            f"  Type: {signal.get('signal_type', 'unknown')}"
        )
        lines.append(
            f"  NCT ID: {signal.get('nct_id', 'N/A')}"
        )
        lines.append(
            f"  Confidence: "
            f"{signal.get('confidence', 0.0):.2f}"
        )
        lines.append(
            f"  Summary: {signal.get('summary', '')}"
        )
        lines.append("")

    return "\n".join(lines)


def _fallback_brief(
    signals: list,
    agents_activated: list,
    task: str,
) -> str:

    lines = [
        "**EXECUTIVE SUMMARY:**",
        f"Analysis complete. {len(signals)} signal(s) detected.",
        "",
        "**SIGNALS BY PRIORITY:**",
        "",
    ]

    for i, signal in enumerate(signals, start=1):
        lines.append(
            f"{i}. **{signal.get('nct_id', 'Unknown')} - "
            f"{signal.get('signal_type', 'Unknown')}:**"
        )

        lines.append(
            f"   {signal.get('summary', 'No summary available.')}"
        )

        lines.append(
            f"   Confidence: "
            f"{signal.get('confidence', 0.0):.2f} | "
            f"Agent: {signal.get('agent', 'unknown')}"
        )

        lines.append("")

    lines.append(
        f"**AGENTS ACTIVATED:** "
        f"{', '.join(agents_activated)}"
    )

    lines.append(
        "\n*Brief generated without LLM assistance. "
        "Review the raw signals directly.*"
    )

    return "\n".join(lines)