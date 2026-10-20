
import asyncio

from ingestion.clinical_trials_client import ClinicalTrialsClient
from ingestion.pubmed_client import PubMedClient
from ingestion.document_parser import DocumentParser
from ingestion.gcs_store import GCSStore
from config.logging_config import setup_logging


logger = setup_logging(__name__)

SEARCH_CONDITIONS = [
    "diabetes",
    "cancer",
    "cardiovascular disease",
]

MAX_STUDIES_PER_CONDITION = 50
MAX_PAPERS_PER_STUDY = 10


async def run_ingestion():
    """Run the complete clinical trial and paper ingestion pipeline."""

    logger.info("=" * 60)
    logger.info("Starting MOSAIC ingestion pipeline")
    logger.info(f"Conditions to search: {', '.join(SEARCH_CONDITIONS)}")
    logger.info(
        f"Max studies per condition: {MAX_STUDIES_PER_CONDITION}"
    )
    logger.info("=" * 60)

    parser = DocumentParser()
    store = GCSStore()

    total_studies = 0
    total_papers = 0

    async with ClinicalTrialsClient() as ct_client:
        async with PubMedClient() as pubmed_client:
            for condition in SEARCH_CONDITIONS:
                logger.info(f"Fetching studies | condition={condition}")

                raw_studies = await ct_client.search_studies(
                    condition=condition,
                    max_results=MAX_STUDIES_PER_CONDITION,
                )

                logger.info(
                    f"Fetched {len(raw_studies)} studies | "
                    f"condition={condition}"
                )

                parsed_studies = parser.parse_studies(raw_studies)

                for study in parsed_studies:
                    raw_match = next(
                        (
                            raw
                            for raw in raw_studies
                            if raw.get("protocolSection", {})
                            .get("identificationModule", {})
                            .get("nctId") == study.nct_id
                        ),
                        None,
                    )

                    if raw_match:
                        await store.save_raw_study(
                            nct_id=study.nct_id,
                            data=raw_match,
                        )

                    await store.save_parsed_study(study)
                    total_studies += 1

                    papers = await pubmed_client.fetch_papers_for_trial(
                        nct_id=study.nct_id,
                        max_results=MAX_PAPERS_PER_STUDY,
                    )

                    parsed_papers = parser.parse_papers(papers)

                    for paper in parsed_papers:
                        await store.save_raw_paper(
                            pmid=paper.pmid,
                            data=paper.model_dump(),
                        )

                        await store.save_parsed_paper(paper)
                        total_papers += 1

    logger.info("=" * 60)
    logger.info("Ingestion complete")
    logger.info(f"Studies saved: {total_studies}")
    logger.info(f"Papers saved: {total_papers}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_ingestion())
