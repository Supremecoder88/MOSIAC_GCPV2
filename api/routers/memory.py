from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import (
    get_episodic_store,
    get_procedural_store,
    get_semantic_store,
)
from memory.episodic_store import EpisodicStore
from memory.procedural_store import ProceduralStore
from memory.semantic_store import SemanticStore
from config.logging_config import setup_logging


logger = setup_logging(__name__)
router = APIRouter()


@router.get(
    "/memory/episodes",
    summary="Search past agent reasoning sessions",
    description="Search episodic memory by meaning or list recent episodes.",
    tags=["Memory"],
)
async def get_episodes(
    query: str = Query(
        default=None,
        description="Search episodes by meaning. Leave empty for recent episodes.",
    ),
    agent_name: str = Query(
        default=None,
        description="Filter to one agent's episodes.",
    ),
    limit: int = Query(default=10, ge=1, le=50),
    store: EpisodicStore = Depends(get_episodic_store),
):
    """Search or retrieve recent episodic memories."""

    try:
        if query:
            episodes = await store.search_episodes(
                query=query,
                agent_name=agent_name,
                top_k=limit,
            )
        else:
            episodes = await store.get_recent_episodes(
                agent_name=agent_name,
                limit=limit,
            )

        return {
            "episodes": episodes,
            "count": len(episodes),
            "query": query,
        }

    except Exception as e:
        logger.error(f"get_episodes failed | error={e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/memory/procedures/{agent_name}",
    summary="Get an agent's current reasoning rules",
    description=(
        "Returns all reasoning rules for the specified agent, "
        "including default and learned rules."
    ),
    tags=["Memory"],
)
async def get_procedures(
    agent_name: str,
    store: ProceduralStore = Depends(get_procedural_store),
):
    """Return current reasoning rules for an agent."""

    try:
        procedures = await store.get_all_procedures_for_api(
            agent_name=agent_name
        )

        return {
            "agent_name": agent_name,
            "procedures": procedures,
            "total_rules": len(procedures),
            "default_rules": sum(
                1 for p in procedures if p["rule_type"] == "default"
            ),
            "learned_rules": sum(
                1 for p in procedures if p["rule_type"] == "learned"
            ),
        }

    except Exception as e:
        logger.error(f"get_procedures failed | error={e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/sponsors/{sponsor_name:path}",
    summary="Get sponsor credibility profile",
    tags=["Sponsors"],
)
async def get_sponsor_profile(
    sponsor_name: str,
    store: SemanticStore = Depends(get_semantic_store),
):
    """Return the credibility profile for a sponsor."""

    try:
        profile = await store.get_sponsor_profile(sponsor=sponsor_name)

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No profile found for sponsor '{sponsor_name}'. "
                    "They may not have been analysed yet."
                ),
            )

        return profile

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_sponsor_profile failed | error={e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/sponsors",
    summary="List all sponsor profiles",
    description=(
        "Returns all sponsor profiles ordered by credibility ascending "
        "(worst sponsors first)."
    ),
    tags=["Sponsors"],
)
async def list_sponsors(
    limit: int = Query(default=50, ge=1, le=200),
    store: SemanticStore = Depends(get_semantic_store),
):
    """Return sponsor profiles ordered by credibility."""

    try:
        sponsors = await store.get_all_sponsor_profiles(limit=limit)

        return {
            "sponsors": sponsors,
            "count": len(sponsors),
        }

    except Exception as e:
        logger.error(f"list_sponsors failed | error={e}")
        raise HTTPException(status_code=500, detail=str(e))