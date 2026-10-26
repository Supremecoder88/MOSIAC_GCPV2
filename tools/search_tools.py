import asyncio
import json

from langchain_core.tools import tool

from processing.vector_store import VectorStore
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from config.logging_config import setup_logging


logger = setup_logging(__name__)

# Shared store instances
_vector_store = VectorStore()
_episodic_store = EpisodicStore()
_semantic_store = SemanticStore()


def _run_async(coroutine):
    """Run an async database operation synchronously."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


@tool
def search_studies_by_meaning(
    query: str,
    top_k: int = 5,
    source_filter: str = "study",
) -> str:
    """Search clinical trial studies by semantic similarity."""
    logger.info(
        f"Tool called: search_studies_by_meaning | "
        f"query='{query[:60]}' | top_k={top_k}"
    )

    try:
        results = _run_async(
            _vector_store.search(
                query_text=query,
                top_k=top_k,
                source_filter=source_filter,
            )
        )

        if not results:
            return json.dumps({
                "results": [],
                "message": "No relevant studies found for this query.",
                "query": query,
            })

        return json.dumps({
            "results": results,
            "count": len(results),
            "query": query,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"search_studies_by_meaning failed | error={e}"
        )
        return json.dumps({"error": str(e), "results": []})


@tool
def search_past_episodes(
    query: str,
    agent_name: str,
    top_k: int = 3,
) -> str:
    """Search past agent sessions for relevant findings."""
    logger.info(
        f"Tool called: search_past_episodes | "
        f"agent={agent_name} | query='{query[:60]}'"
    )

    try:
        episodes = _run_async(
            _episodic_store.search_episodes(
                query=query,
                agent_name=agent_name,
                top_k=top_k,
            )
        )

        if not episodes:
            return json.dumps({
                "episodes": [],
                "message": (
                    "No relevant past episodes found. "
                    "This appears to be a new type of investigation."
                ),
                "query": query,
            })

        return json.dumps({
            "episodes": episodes,
            "count": len(episodes),
            "query": query,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"search_past_episodes failed | error={e}")
        return json.dumps({"error": str(e), "episodes": []})


@tool
def save_episode(
    agent_name: str,
    content: str,
    nct_id: str = "",
    outcome: str = "completed",
) -> str:
    """Save an agent's investigation to episodic memory."""
    logger.info(
        f"Tool called: save_episode | "
        f"agent={agent_name} | nct_id={nct_id} | outcome={outcome}"
    )

    try:
        episode_id = _run_async(
            _episodic_store.save_episode(
                agent_name=agent_name,
                content=content,
                nct_id=nct_id if nct_id else None,
                outcome=outcome,
            )
        )

        return json.dumps({
            "success": True,
            "episode_id": episode_id,
            "message": "Episode saved to long-term memory successfully.",
            "agent": agent_name,
        }, indent=2)

    except Exception as e:
        logger.error(f"save_episode failed | error={e}")
        return json.dumps({"success": False, "error": str(e)})


@tool
def get_sponsor_profile(sponsor_name: str) -> str:
    """Retrieve historical information about a research sponsor."""
    logger.info(
        f"Tool called: get_sponsor_profile | sponsor={sponsor_name}"
    )

    try:
        profile = _run_async(
            _semantic_store.get_sponsor_profile(sponsor=sponsor_name)
        )

        if profile is None:
            return json.dumps({
                "sponsor": sponsor_name,
                "found": False,
                "message": (
                    f"No historical data for '{sponsor_name}'. "
                    "This sponsor has not been analysed before. "
                    "Proceed with lower confidence."
                ),
            }, indent=2)

        return json.dumps({
            "found": True,
            "profile": profile,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"get_sponsor_profile failed | error={e}")
        return json.dumps({"error": str(e), "found": False})


@tool
def update_sponsor_profile(
    sponsor_name: str,
    results_posted: bool = False,
    had_broken_promise: bool = False,
    delay_days: int = 0,
) -> str:
    """Update a sponsor's profile with findings from a study."""
    logger.info(
        f"Tool called: update_sponsor_profile | "
        f"sponsor={sponsor_name} | "
        f"results_posted={results_posted} | "
        f"broken_promise={had_broken_promise} | "
        f"delay_days={delay_days}"
    )

    try:
        _run_async(
            _semantic_store.update_sponsor_knowledge(
                sponsor=sponsor_name,
                results_posted=results_posted,
                had_broken_promise=had_broken_promise,
                delay_days=delay_days,
            )
        )

        return json.dumps({
            "success": True,
            "sponsor": sponsor_name,
            "message": "Sponsor profile updated successfully.",
            "results_posted": results_posted,
            "had_broken_promise": had_broken_promise,
            "delay_days": delay_days,
        }, indent=2)

    except Exception as e:
        logger.error(f"update_sponsor_profile failed | error={e}")
        return json.dumps({"success": False, "error": str(e)})


@tool
def get_low_credibility_sponsors(
    threshold: float = 0.6,
    min_studies: int = 3,
) -> str:
    """Find sponsors whose credibility scores are below a threshold."""
    logger.info(
        f"Tool called: get_low_credibility_sponsors | "
        f"threshold={threshold} | min_studies={min_studies}"
    )

    try:
        sponsors = _run_async(
            _semantic_store.get_low_credibility_sponsors(
                threshold=threshold,
                min_studies=min_studies,
            )
        )

        if not sponsors:
            return json.dumps({
                "sponsors": [],
                "message": (
                    f"No sponsors found below credibility {threshold} "
                    f"with at least {min_studies} studies."
                ),
                "count": 0,
            }, indent=2)

        return json.dumps({
            "sponsors": sponsors,
            "count": len(sponsors),
            "threshold": threshold,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"get_low_credibility_sponsors failed | error={e}"
        )
        return json.dumps({"error": str(e), "sponsors": []})


@tool
def search_study_chunks_by_nct_id(
    nct_id: str,
    query: str = "",
) -> str:
    """Retrieve chunks for a specific clinical trial."""
    logger.info(
        f"Tool called: search_study_chunks_by_nct_id | "
        f"nct_id={nct_id}"
    )

    try:
        if query:
            # Search relevant chunks within this study.
            results = _run_async(
                _vector_store.search(
                    query_text=query,
                    top_k=5,
                    nct_id_filter=nct_id,
                )
            )
        else:
            # Retrieve every chunk for the study.
            results = _run_async(
                _vector_store.get_chunks_for_study(nct_id=nct_id)
            )

        if not results:
            return json.dumps({
                "nct_id": nct_id,
                "chunks": [],
                "message": (
                    f"No chunks found for study {nct_id}. "
                    "The study may not have been processed yet."
                ),
            })

        return json.dumps({
            "nct_id": nct_id,
            "chunks": results,
            "count": len(results),
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"search_study_chunks_by_nct_id failed | "
            f"nct_id={nct_id} | error={e}"
        )
        return json.dumps({"error": str(e), "chunks": []})


@tool
def search_papers_by_meaning(
    query: str,
    top_k: int = 5,
) -> str:
    """Search published PubMed papers by semantic similarity."""
    logger.info(
        f"Tool called: search_papers_by_meaning | "
        f"query='{query[:60]}' | top_k={top_k}"
    )

    try:
        results = _run_async(
            _vector_store.search(
                query_text=query,
                top_k=top_k,
                source_filter="paper",
            )
        )

        if not results:
            return json.dumps({
                "results": [],
                "message": "No relevant papers found for this query.",
                "query": query,
            })

        return json.dumps({
            "results": results,
            "count": len(results),
            "query": query,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"search_papers_by_meaning failed | error={e}")
        return json.dumps({"error": str(e), "results": []})


# Tools exported for agents to use.
ALL_SEARCH_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
    get_low_credibility_sponsors,
    search_study_chunks_by_nct_id,
    search_papers_by_meaning,
]