# tools/pubmed_tools.py

import asyncio
import json

from langchain_core.tools import tool

from ingestion.pubmed_client import PubMedClient
from ingestion.document_parser import DocumentParser
from config.logging_config import setup_logging


logger = setup_logging(__name__)
_parser = DocumentParser()


def _run_async(coroutine):
    """Run an async operation synchronously."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


@tool
def fetch_papers_for_trial(
    nct_id: str,
    max_papers: int = 10,
) -> str:
    """Fetch published papers referencing a clinical trial."""
    logger.info(
        f"Tool called: fetch_papers_for_trial | "
        f"nct_id={nct_id} | max_papers={max_papers}"
    )

    async def _fetch():
        async with PubMedClient() as client:
            return await client.fetch_papers_for_trial(
                nct_id=nct_id,
                max_results=max_papers,
            )

    try:
        raw_papers = _run_async(_fetch())

        if not raw_papers:
            return json.dumps({
                "nct_id": nct_id,
                "papers": [],
                "count": 0,
                "message": (
                    f"No published papers found on PubMed that reference "
                    f"trial {nct_id}. Results may exist in other sources."
                ),
            }, indent=2)

        parsed_papers = _parser.parse_papers(raw_papers=raw_papers)
        papers_list = []

        for paper in parsed_papers:
            paper_dict = paper.model_dump()
            papers_list.append({
                "pmid": paper_dict["pmid"],
                "title": paper_dict["title"],
                "abstract": paper_dict["abstract"],
                "journal": paper_dict["journal"],
                "pub_date": paper_dict["pub_date"],
                "authors": paper_dict["authors"][:5],
                "word_count": paper_dict["word_count"],
                "nct_ids_referenced": paper_dict["nct_ids_referenced"],
            })

        return json.dumps({
            "nct_id": nct_id,
            "papers": papers_list,
            "count": len(papers_list),
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"fetch_papers_for_trial failed | "
            f"nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error": str(e),
            "papers": [],
            "count": 0,
        })


@tool
def search_pubmed_by_query(
    query: str,
    max_papers: int = 5,
) -> str:
    """Search PubMed for papers matching a topic or query."""
    logger.info(
        f"Tool called: search_pubmed_by_query | "
        f"query='{query[:60]}' | max_papers={max_papers}"
    )

    async def _search():
        async with PubMedClient() as client:
            paper_ids = await client._search_paper_ids(
                nct_id=query,
                max_results=max_papers,
            )

            if not paper_ids:
                return []

            return await client._fetch_paper_details(
                paper_ids=paper_ids
            )

    try:
        raw_papers = _run_async(_search())

        if not raw_papers:
            return json.dumps({
                "query": query,
                "papers": [],
                "count": 0,
                "message": (
                    f"No papers found on PubMed for query: '{query}'. "
                    "Try broader keywords."
                ),
            }, indent=2)

        parsed_papers = _parser.parse_papers(raw_papers=raw_papers)
        papers_list = []

        for paper in parsed_papers:
            paper_dict = paper.model_dump()
            papers_list.append({
                "pmid": paper_dict["pmid"],
                "title": paper_dict["title"],
                "abstract": paper_dict["abstract"],
                "journal": paper_dict["journal"],
                "pub_date": paper_dict["pub_date"],
                "authors": paper_dict["authors"][:5],
                "word_count": paper_dict["word_count"],
                "nct_ids_referenced": paper_dict["nct_ids_referenced"],
            })

        return json.dumps({
            "query": query,
            "papers": papers_list,
            "count": len(papers_list),
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"search_pubmed_by_query failed | "
            f"query={query} | error={e}"
        )
        return json.dumps({
            "query": query,
            "error": str(e),
            "papers": [],
            "count": 0,
        })


@tool
def compare_filing_vs_papers(
    nct_id: str,
    filing_summary: str,
) -> str:
    """Compare a trial's official filing with published papers."""
    logger.info(
        f"Tool called: compare_filing_vs_papers | nct_id={nct_id}"
    )

    async def _fetch():
        async with PubMedClient() as client:
            return await client.fetch_papers_for_trial(
                nct_id=nct_id,
                max_results=15,
            )

    try:
        raw_papers = _run_async(_fetch())
        parsed_papers = _parser.parse_papers(
            raw_papers=raw_papers or []
        )

        papers_list = []

        for paper in parsed_papers:
            paper_dict = paper.model_dump()
            papers_list.append({
                "pmid": paper_dict["pmid"],
                "title": paper_dict["title"],
                "abstract": paper_dict["abstract"],
                "journal": paper_dict["journal"],
                "pub_date": paper_dict["pub_date"],
                "authors": paper_dict["authors"][:3],
            })

        comparison_note = (
            "Compare the filing_summary against each paper's abstract. "
            "Check for adverse events absent from the filing, differences "
            "in event severity, contradictory outcome results, and "
            "published results when results_posted=False."
        )

        return json.dumps({
            "nct_id": nct_id,
            "filing_summary": filing_summary,
            "papers": papers_list,
            "papers_count": len(papers_list),
            "comparison_note": comparison_note,
            "has_papers": len(papers_list) > 0,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"compare_filing_vs_papers failed | "
            f"nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error": str(e),
            "papers": [],
        })


# Tools exported for agent assignment.
ALL_PUBMED_TOOLS = [
    fetch_papers_for_trial,
    search_pubmed_by_query,
    compare_filing_vs_papers,
]