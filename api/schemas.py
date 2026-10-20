from typing import Any

from pydantic import BaseModel, Field


# Request schemas


class AnalysisRequest(BaseModel):
    """Request body for POST /api/v1/analyze."""

    task: str = Field(
        default="Find completed clinical trials with research integrity issues",
        description=(
            "The analysis task in plain English. "
            "Agents read this and decide what to investigate."
        ),
        example="Find completed trials where sponsor never posted results",
    )

    nct_ids: list[str] = Field(
        default=[],
        description=(
            "Specific NCT IDs to analyse. "
            "Empty list means analyse broadly across all studies."
        ),
        example=["NCT04788680", "NCT02208921"],
    )

    max_studies: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum studies to analyse per agent. Default 10.",
    )


class ReviewDecisionRequest(BaseModel):
    """Request body for PATCH /api/v1/review/{queue_id}."""

    decision: str = Field(
        ...,
        description="The reviewer's decision: approve, reject, or edit",
        example="reject",
    )

    reviewer: str = Field(
        default="analyst",
        description="Name or ID of the human reviewer.",
        example="chirantan@tarkaupskilling.com",
    )

    rejection_reason: str = Field(
        default="",
        description=(
            "Why this signal was rejected. "
            "This is written to procedural memory."
        ),
        example=(
            "This trial was terminated early due to COVID — "
            "terminated trials are exempt from result posting requirements."
        ),
    )

    edit_summary: str = Field(
        default="",
        description=(
            "Corrected signal summary if decision is 'edit'. "
            "Replaces the agent's original summary."
        ),
    )


# Response schemas


class SignalResponse(BaseModel):
    """One signal in an API response."""

    signal_id: str
    nct_id: str
    agent: str
    signal_type: str
    summary: str
    confidence: float
    status: str
    created_at: str

    class Config:
        from_attributes = True


class AnalysisResponse(BaseModel):
    """Response from POST /api/v1/analyze."""

    run_id: str
    task: str
    final_brief: str
    total_signals: int
    signals_requiring_review: int
    agents_activated: list[str]
    duration_seconds: float


class ReviewQueueItem(BaseModel):
    """One item in the human review queue."""

    review_id: str
    signal_id: str
    agent: str
    signal_type: str
    summary: str
    confidence: float
    nct_id: str
    decision: str


class ReviewQueueResponse(BaseModel):
    """Response from GET /api/v1/review/queue."""

    queue: list[ReviewQueueItem]
    total_pending: int
    total_approved: int
    total_rejected: int


class ReviewDecisionResponse(BaseModel):
    """Response from PATCH /api/v1/review/{queue_id}."""

    success: bool
    decision: str
    signal_id: str
    queue_id: str
    memory_updated: bool
    message: str


class EpisodeResponse(BaseModel):
    """One episode from episodic memory."""

    episode_id: str
    agent_name: str
    nct_id: str | None
    content: str
    outcome: str | None
    similarity: float | None
    created_at: str


class ProcedureResponse(BaseModel):
    """One reasoning rule from procedural memory."""

    procedure_id: str
    agent_name: str
    rule_text: str
    rule_type: str
    source: str
    created_at: str


class SponsorProfileResponse(BaseModel):
    """Full sponsor credibility profile."""

    sponsor: str
    credibility_score: float
    total_studies: int
    results_posted: int
    results_missing: int
    broken_promises: int
    avg_delay_days: float
    last_updated: str


class HealthResponse(BaseModel):
    """System health check response."""

    status: str
    app: str
    version: str
    database: str
    details: dict[str, Any]