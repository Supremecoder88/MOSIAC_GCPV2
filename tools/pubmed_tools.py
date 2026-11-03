
import json

from langchain_core.tools import tool

from ingestion.pubmed_client import PubMedClient
from ingestion.document_parser import DocumentParser
from config.logging_config import setup_logging


logger = setup_logging(__name__)
_parser = DocumentParser()


@tool
async def fetch_papers_for_trial(
    nct_id: str,
    max_papers: int = 10,
) -> str:
    """Fetch published papers referencing a clinical trial."""

    logger.info(
        f"Tool called: fetch_papers_for_trial | "
        f"nct_id={nct_id} | max_papers={max_papers}"
    )

    try:
        async with PubMedClient() as client:
            raw_papers = await client.fetch_papers_for_trial(
                nct_id=nct_id,
                max_results=max_papers,
            )

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

        parsed_papers = _parser.parse_papers(
            raw_papers=raw_papers
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
async def search_pubmed_by_query(
    query: str,
    max_papers: int = 5,
) -> str:
    """Search PubMed for papers matching a topic or query."""

    logger.info(
        f"Tool called: search_pubmed_by_query | "
        f"query='{query[:60]}' | max_papers={max_papers}"
    )

    try:
        async with PubMedClient() as client:
            raw_papers = await client.search_papers_by_query(
                query=query,
                max_results=max_papers,
            )

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

        parsed_papers = _parser.parse_papers(
            raw_papers=raw_papers
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
async def compare_filing_vs_papers(
    nct_id: str,
    filing_summary: str,
) -> str:
    """
    Compare published papers with trial filing context.

    The filing_summary is agent-provided context, not independently
    verified official registry safety data.
    """

    logger.info(
        f"Tool called: compare_filing_vs_papers | nct_id={nct_id}"
    )

    try:
        async with PubMedClient() as client:
            raw_papers = await client.fetch_papers_for_trial(
                nct_id=nct_id,
                max_results=15,
            )

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

        return json.dumps({
            "nct_id": nct_id,

            # Agent-provided context; not verified registry evidence.
            "filing_summary": filing_summary,
            "filing_summary_verified": False,

            # Retrieved publication evidence.
            "papers": papers_list,
            "papers_count": len(papers_list),
            "has_papers": len(papers_list) > 0,

            # Conservative safety validation.
            "official_safety_data_verified": False,
            "official_adverse_event_counts_available": False,
            "safety_comparison_status": "INSUFFICIENT_OFFICIAL_DATA",
            "safety_gap_signal_allowed": False,

            "validation_rules": [
                "The filing_summary is agent-provided context, not verified registry evidence.",
                "Missing adverse-event counts must be treated as unknown, never as zero.",
                "results_posted=false does not establish that zero adverse events occurred.",
                "Published adverse events alone do not establish an official reporting discrepancy.",
                "A safety reporting gap must not be generated without verified official "
                "adverse-event counts and comparable publication evidence.",
                "Compare matching populations, treatment arms, reporting periods, "
                "event definitions, severity, and denominators.",
            ],

            "comparison_note": (
                "Published papers were retrieved, but official adverse-event "
                "counts have not been independently verified by this tool. "
                "Therefore, this result cannot establish a discrepancy in "
                "safety reporting. Treat the safety comparison as insufficient "
                "evidence and do not generate a safety-gap signal."
            ),

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
            "official_safety_data_verified": False,
            "safety_comparison_status": "INSUFFICIENT_OFFICIAL_DATA",
            "safety_gap_signal_allowed": False,
        })


# Tools exported for agent assignment.
ALL_PUBMED_TOOLS = [
    fetch_papers_for_trial,
    search_pubmed_by_query,
    compare_filing_vs_papers,
]
