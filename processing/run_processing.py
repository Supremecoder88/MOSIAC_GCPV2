
import asyncio

from config.logging_config import setup_logging
from ingestion.document_parser import ParsedPaper, ParsedStudy
from ingestion.gcs_store import GCSStore
from processing.chunker import Chunker
from processing.embedder import Embedder
from processing.vector_store import VectorStore


logger = setup_logging(__name__)


async def run_processing() -> None:
    """
    Load parsed studies and papers from GCS,
    save metadata, generate only required embeddings,
    and store vectors in Cloud SQL.
    """

    gcs_store = GCSStore()
    chunker = Chunker()
    embedder = Embedder()

    logger.info("=" * 60)
    logger.info("MOSAIC PROCESSING PIPELINE STARTED")
    logger.info("=" * 60)

    # --------------------------------------------------
    # STEP 1: LOAD STUDIES FROM GCS
    # --------------------------------------------------

    logger.info("Loading parsed studies from GCS...")

    nct_ids = await gcs_store.list_processed_studies()

    logger.info(f"Found {len(nct_ids)} studies in GCS")

    studies: list[ParsedStudy] = []

    for nct_id in nct_ids:
        study = await gcs_store.load_parsed_study(nct_id)

        if study:
            studies.append(study)

    logger.info(
        f"Studies loaded | "
        f"loaded={len(studies)} | "
        f"listed={len(nct_ids)}"
    )

    if len(studies) != len(nct_ids):
        raise RuntimeError(
            "Study loading incomplete. "
            "Stopping before database processing to avoid "
            "processing an incomplete GCS dataset."
        )

    # --------------------------------------------------
    # STEP 2: LOAD PAPERS FROM GCS
    # --------------------------------------------------

    logger.info("Loading parsed PubMed papers from GCS...")

    pmids = await gcs_store.list_processed_papers()

    logger.info(f"Found {len(pmids)} papers in GCS")

    papers: list[ParsedPaper] = []

    for pmid in pmids:
        paper = await gcs_store.load_parsed_paper(pmid)

        if paper:
            papers.append(paper)

    logger.info(
        f"Papers loaded | "
        f"loaded={len(papers)} | "
        f"listed={len(pmids)}"
    )

    if len(papers) != len(pmids):
        raise RuntimeError(
            "Paper loading incomplete. "
            "Stopping before database processing to avoid "
            "processing an incomplete GCS dataset."
        )

    # --------------------------------------------------
    # STEP 3: SAVE STUDY AND PAPER METADATA
    # --------------------------------------------------

    async with VectorStore() as vector_store:

        logger.info("Saving study metadata to Cloud SQL...")

        studies_saved = 0

        for study in studies:
            await vector_store.save_study(
                study_data={
                    "nct_id": study.nct_id,
                    "title": study.title,
                    "sponsor": study.sponsor,
                    "phase": study.phase,
                    "status": study.status,
                    "conditions": study.conditions,
                    "interventions": study.interventions,
                    "primary_outcome": study.primary_outcome,
                    "secondary_outcomes": study.secondary_outcomes,
                    "start_date": study.start_date,
                    "completion_date": study.completion_date,
                    "results_posted": study.results_posted,
                    "enrollment": study.enrollment,
                    "gcs_path": (
                        f"processed/studies/{study.nct_id}.json"
                    ),
                }
            )

            studies_saved += 1

        logger.info(
            f"Study metadata saved | saved={studies_saved}"
        )

        # Save studies before paper relationships because
        # paper_studies references studies.nct_id.

        logger.info("Saving PubMed paper metadata to Cloud SQL...")

        papers_saved = 0
        paper_links_created = 0

        for paper in papers:
            links = await vector_store.save_paper(
                paper_data={
                    "pmid": paper.pmid,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "journal": paper.journal,
                    "pub_date": paper.pub_date,
                    "authors": paper.authors,
                    "nct_ids_referenced": paper.nct_ids_referenced,
                    "source": paper.source,
                    "word_count": paper.word_count,
                    "parsed_at": paper.parsed_at,
                }
            )

            papers_saved += 1
            paper_links_created += links

        logger.info(
            f"Paper metadata saved | "
            f"saved={papers_saved} | "
            f"new_trial_links={paper_links_created}"
        )

        # --------------------------------------------------
        # STEP 4: CHUNK STUDIES AND PAPERS
        # --------------------------------------------------

        logger.info("Chunking clinical studies...")

        study_chunks = chunker.chunk_studies(studies)

        logger.info(
            f"Study chunks created: {len(study_chunks)}"
        )

        logger.info("Chunking PubMed papers...")

        paper_chunks = chunker.chunk_papers(papers)

        logger.info(
            f"Paper chunks created: {len(paper_chunks)}"
        )

        all_chunks = study_chunks + paper_chunks

        logger.info(
            f"Total chunks generated: {len(all_chunks)}"
        )

        # --------------------------------------------------
        # STEP 5: IDENTIFY NEW OR MODIFIED CHUNKS
        # --------------------------------------------------

        logger.info(
            "Checking existing database chunks..."
        )

        chunks_to_embed = (
            await vector_store.get_chunks_needing_embedding(
                all_chunks
            )
        )

        unchanged_chunks = len(all_chunks) - len(chunks_to_embed)

        logger.info(
            f"Incremental comparison complete | "
            f"total={len(all_chunks)} | "
            f"unchanged={unchanged_chunks} | "
            f"requiring_embedding={len(chunks_to_embed)}"
        )

        # --------------------------------------------------
        # STEP 6: GENERATE ONLY REQUIRED EMBEDDINGS
        # --------------------------------------------------

        logger.info(
            "Generating embeddings for new or modified chunks..."
        )

        embedded_chunks = await embedder.embed_chunks(
            chunks_to_embed
        )

        logger.info(
            f"Embedding results | "
            f"input={len(chunks_to_embed)} | "
            f"embedded={len(embedded_chunks)}"
        )

        if len(embedded_chunks) != len(chunks_to_embed):
            failed = len(chunks_to_embed) - len(embedded_chunks)

            raise RuntimeError(
                f"Embedding incomplete: {failed} chunks failed. "
                f"Expected {len(chunks_to_embed)}, "
                f"received {len(embedded_chunks)}. "
                f"No embeddings from this stage were submitted "
                f"to Cloud SQL."
            )

        # --------------------------------------------------
        # STEP 7: SAVE NEW OR UPDATED EMBEDDINGS
        # --------------------------------------------------

        logger.info(
            "Saving new or updated embeddings to Cloud SQL..."
        )

        chunks_stored = await vector_store.save_embedded_chunks(
            embedded_chunks
        )

        if chunks_stored != len(embedded_chunks):
            raise RuntimeError(
                f"Chunk storage incomplete: "
                f"expected {len(embedded_chunks)}, "
                f"processed {chunks_stored}."
            )

        # --------------------------------------------------
        # STEP 8: FINAL SUMMARY
        # --------------------------------------------------

        total_chunks = await vector_store.get_chunk_count()

        logger.info("=" * 60)
        logger.info("MOSAIC PROCESSING COMPLETE")
        logger.info("=" * 60)

        logger.info(f"Studies processed       : {len(studies)}")
        logger.info(f"Papers processed        : {len(papers)}")
        logger.info(f"Study metadata saved    : {studies_saved}")
        logger.info(f"Paper metadata saved    : {papers_saved}")
        logger.info(f"New paper-trial links   : {paper_links_created}")
        logger.info(f"Study chunks generated  : {len(study_chunks)}")
        logger.info(f"Paper chunks generated  : {len(paper_chunks)}")
        logger.info(f"Total chunks generated  : {len(all_chunks)}")
        logger.info(f"Unchanged chunks skipped: {unchanged_chunks}")
        logger.info(f"Chunks requiring embed  : {len(chunks_to_embed)}")
        logger.info(f"Chunks embedded         : {len(embedded_chunks)}")
        logger.info(f"Chunks stored/updated   : {chunks_stored}")
        logger.info(f"Total database chunks   : {total_chunks}")

        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_processing())
