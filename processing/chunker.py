
from dataclasses import dataclass

from ingestion.document_parser import ParsedPaper, ParsedStudy
from config.logging_config import setup_logging


logger = setup_logging(__name__)

CHUNK_SIZE = 500
OVERLAP_SIZE = 50


@dataclass
class TextChunk:
    """A text chunk with its source and metadata."""

    chunk_id: str
    nct_id: str
    chunk_text: str
    chunk_index: int
    source: str
    word_count: int


class Chunker:
    """Split study and paper documents into overlapping text chunks."""

    def chunk_study(self, study: ParsedStudy) -> list[TextChunk]:
        """Chunk a parsed clinical study."""

        full_text = self._build_study_text(study)
        chunks = self._split_into_chunks(
            text=full_text,
            nct_id=study.nct_id,
            source="study",
        )

        logger.info(
            f"Chunked study | nct_id={study.nct_id} | "
            f"chunks_produced={len(chunks)}"
        )
        return chunks

    def chunk_paper(self, paper: ParsedPaper) -> list[TextChunk]:
        """Chunk a parsed research paper."""

        full_text = self._build_paper_text(paper)
        chunks = self._split_into_chunks(
            text=full_text,
            nct_id=paper.pmid,
            source="paper",
        )

        logger.info(
            f"Chunked paper | pmid={paper.pmid} | "
            f"chunks_produced={len(chunks)}"
        )
        return chunks

    def chunk_studies(self, studies: list[ParsedStudy]) -> list[TextChunk]:
        """Chunk multiple studies into one list."""

        all_chunks = []

        for study in studies:
            all_chunks.extend(self.chunk_study(study))

        logger.info(
            f"Chunked all studies | studies={len(studies)} | "
            f"total_chunks={len(all_chunks)}"
        )
        return all_chunks

    def chunk_papers(self, papers: list[ParsedPaper]) -> list[TextChunk]:
        """Chunk multiple papers into one list."""

        all_chunks = []

        for paper in papers:
            all_chunks.extend(self.chunk_paper(paper))

        logger.info(
            f"Chunked all papers | papers={len(papers)} | "
            f"total_chunks={len(all_chunks)}"
        )
        return all_chunks

    def _build_study_text(self, study: ParsedStudy) -> str:
        """Build labelled text from a study's fields."""

        sections = [
            f"NCT ID: {study.nct_id}",
            f"TITLE: {study.title}",
            f"SPONSOR: {study.sponsor}",
            f"PHASE: {study.phase}",
            f"STATUS: {study.status}",
        ]

        if study.conditions:
            sections.append(f"CONDITIONS: {', '.join(study.conditions)}")

        if study.interventions:
            sections.append(
                f"INTERVENTIONS: {', '.join(study.interventions)}"
            )

        if study.primary_outcome:
            sections.append(f"PRIMARY OUTCOME: {study.primary_outcome}")

        if study.secondary_outcomes:
            sections.append(
                f"SECONDARY OUTCOMES: {'; '.join(study.secondary_outcomes)}"
            )

        if study.start_date:
            sections.append(f"START DATE: {study.start_date}")

        if study.completion_date:
            sections.append(f"COMPLETION DATE: {study.completion_date}")

        sections.append(
            f"RESULTS POSTED: {'YES' if study.results_posted else 'NO'}"
        )

        if study.enrollment:
            sections.append(f"ENROLLMENT: {study.enrollment} participants")

        if study.protocol_amendments:
            sections.append(
                f"PROTOCOL AMENDMENTS: "
                f"{len(study.protocol_amendments)} amendment(s) filed"
            )

        return "\n".join(sections)

    def _build_paper_text(self, paper: ParsedPaper) -> str:
        """Build labelled text from a paper's fields."""

        sections = [
            f"PMID: {paper.pmid}",
            f"TITLE: {paper.title}",
        ]

        if paper.journal:
            sections.append(f"JOURNAL: {paper.journal}")

        if paper.pub_date:
            sections.append(f"PUBLICATION DATE: {paper.pub_date}")

        if paper.authors:
            sections.append(f"AUTHORS: {', '.join(paper.authors[:5])}")

        if paper.nct_ids_referenced:
            sections.append(
                f"CLINICAL TRIALS REFERENCED: "
                f"{', '.join(paper.nct_ids_referenced)}"
            )

        if paper.abstract:
            sections.append(f"ABSTRACT: {paper.abstract}")

        return "\n".join(sections)

    def _split_into_chunks(
        self,
        text: str,
        nct_id: str,
        source: str,
    ) -> list[TextChunk]:
        """Split text into overlapping chunks."""

        words = text.split()

        if not words:
            logger.warning(
                f"Empty text — no chunks produced | "
                f"nct_id={nct_id} | source={source}"
            )
            return []

        chunks: list[TextChunk] = []
        step = CHUNK_SIZE - OVERLAP_SIZE
        chunk_index = 0

        for start in range(0, len(words), step):
            chunk_words = words[start : start + CHUNK_SIZE]

            if not chunk_words:
                break

            chunks.append(
                TextChunk(
                    chunk_id=f"{nct_id}_chunk_{chunk_index}",
                    nct_id=nct_id,
                    chunk_text=" ".join(chunk_words),
                    chunk_index=chunk_index,
                    source=source,
                    word_count=len(chunk_words),
                )
            )
            chunk_index += 1

        logger.info(
            f"Split complete | nct_id={nct_id} | "
            f"total_words={len(words)} | chunks={len(chunks)} | "
            f"chunk_size={CHUNK_SIZE} | overlap={OVERLAP_SIZE}"
        )

        return chunks
