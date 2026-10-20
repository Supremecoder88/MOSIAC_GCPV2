import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from api.schemas import AnalysisRequest, AnalysisResponse
from api.dependencies import get_hitl_gate, get_graph
from graph.hitl import HITLGate
from config.logging_config import setup_logging


logger = setup_logging(__name__)

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Trigger a full MOSAIC analysis run",
    description=(
        "Runs all 6 specialist agents in parallel and returns "
        "a compiled intelligence brief with all signals found."
    ),
    tags=["Analysis"],
)
async def run_analysis(
    request: AnalysisRequest,
    hitl_gate: HITLGate = Depends(get_hitl_gate),
    graph=Depends(get_graph),
):
    """Trigger a complete MOSAIC analysis run."""

    run_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"Analysis run started | "
        f"run_id={run_id} | "
        f"task='{request.task[:80]}'"
    )

    try:
        initial_state = {
            "task": request.task,
            "nct_ids": request.nct_ids,
            "max_studies": request.max_studies,
            "messages": [],
            "signals": [],
            "agents_activated": [],
            "final_brief": "",
            "run_complete": False,
            "run_id": run_id,
            "error_log": [],
        }

        result = await graph.ainvoke(initial_state)

        signals = result.get("signals", [])
        processed_signals = []

        for signal in signals:
            hitl_result = await hitl_gate.process_signal(signal)

            processed_signals.append({
                **signal,
                "hitl_action": hitl_result.get("action"),
            })

            logger.info(
                f"Signal processed | "
                f"agent={signal.get('agent')} | "
                f"confidence={signal.get('confidence', 0):.2f}"
            )

        duration = round(time.time() - start_time, 2)

        signals_requiring_review = sum(
            1
            for signal in processed_signals
            if signal.get("hitl_action") == "sent_to_review"
        )

        logger.info(
            f"Analysis run complete | "
            f"run_id={run_id} | "
            f"signals={len(signals)} | "
            f"duration={duration}s"
        )

        return AnalysisResponse(
            run_id=run_id,
            task=request.task,
            final_brief=result.get(
                "final_brief",
                "No brief generated."
            ),
            total_signals=len(signals),
            signals_requiring_review=signals_requiring_review,
            agents_activated=result.get(
                "agents_activated",
                []
            ),
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error(
            f"Analysis run failed | "
            f"run_id={run_id} | "
            f"error={e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Analysis run failed: {str(e)}",
        )