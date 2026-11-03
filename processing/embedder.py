
import asyncio
from dataclasses import dataclass

from openai import AsyncOpenAI

from config.logging_config import setup_logging
from config.settings import settings
from processing.chunker import TextChunk


logger = setup_logging(__name__)

BATCH_SIZE = 50
RETRY_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 2


@dataclass
class EmbeddedChunk:
    """A text chunk with its vector embedding and document metadata."""

    chunk_id: str
    nct_id: str | None
    pmid: str | None
    chunk_text: str
    chunk_index: int
    source: str
    word_count: int
    embedding: list[float]


class Embedder:
    """Generate vector embeddings for text chunks using OpenAI."""

    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key
        )
        self._model = settings.openai_embedding_model

        logger.info(
            f"Embedder initialised | model={self._model}"
        )

    async def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single search query."""

        if not query or not query.strip():
            raise ValueError("Embedding query cannot be empty.")

        response = await self._client.embeddings.create(
            model=self._model,
            input=query.strip(),
        )

        embedding = response.data[0].embedding

        logger.info(
            f"Query embedded successfully | "
            f"dimensions={len(embedding)}"
        )

        return embedding

    async def embed_chunks(
        self,
        chunks: list[TextChunk],
    ) -> list[EmbeddedChunk]:
        """Embed chunks in batches, skipping batches that fail after retries."""

        if not chunks:
            logger.warning(
                "embed_chunks called with empty list"
            )
            return []

        logger.info(
            f"Starting embedding | "
            f"total_chunks={len(chunks)} | "
            f"batch_size={BATCH_SIZE} | "
            f"model={self._model}"
        )

        all_embedded: list[EmbeddedChunk] = []
        batches = self._create_batches(chunks)

        for batch_num, batch in enumerate(batches):
            logger.info(
                f"Embedding batch {batch_num + 1}/{len(batches)} | "
                f"chunks_in_batch={len(batch)}"
            )

            embedded_batch = await self._embed_batch_with_retry(
                batch=batch,
                batch_num=batch_num,
            )

            all_embedded.extend(embedded_batch)

            if batch_num < len(batches) - 1:
                await asyncio.sleep(0.5)

        logger.info(
            f"Embedding complete | "
            f"total_embedded={len(all_embedded)} | "
            f"total_input={len(chunks)} | "
            f"skipped={len(chunks) - len(all_embedded)}"
        )

        return all_embedded

    def _create_batches(
        self,
        chunks: list[TextChunk],
    ) -> list[list[TextChunk]]:
        """Split chunks into batches of BATCH_SIZE."""

        return [
            chunks[i : i + BATCH_SIZE]
            for i in range(0, len(chunks), BATCH_SIZE)
        ]

    async def _embed_batch_with_retry(
        self,
        batch: list[TextChunk],
        batch_num: int,
    ) -> list[EmbeddedChunk]:
        """Retry a failed batch and return an empty list if all attempts fail."""

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                return await self._embed_batch(batch=batch)

            except Exception as e:
                if attempt < RETRY_ATTEMPTS:
                    logger.warning(
                        f"Embedding failed | "
                        f"batch={batch_num + 1} | "
                        f"attempt={attempt}/{RETRY_ATTEMPTS} | "
                        f"error={e} | "
                        f"retrying in {RETRY_SLEEP_SECONDS}s..."
                    )

                    await asyncio.sleep(RETRY_SLEEP_SECONDS)

                else:
                    logger.error(
                        f"Embedding failed after "
                        f"{RETRY_ATTEMPTS} attempts | "
                        f"batch={batch_num + 1} | "
                        f"error={e} | "
                        f"skipping this batch"
                    )

                    return []

        return []

    async def _embed_batch(
        self,
        batch: list[TextChunk],
    ) -> list[EmbeddedChunk]:
        """Generate embeddings for one batch of chunks."""

        texts = [
            chunk.chunk_text
            for chunk in batch
        ]

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )

        embedded_chunks: list[EmbeddedChunk] = []

        for i, chunk in enumerate(batch):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    nct_id=chunk.nct_id,
                    pmid=chunk.pmid,
                    chunk_text=chunk.chunk_text,
                    chunk_index=chunk.chunk_index,
                    source=chunk.source,
                    word_count=chunk.word_count,
                    embedding=response.data[i].embedding,
                )
            )

        logger.info(
            f"Batch embedded successfully | "
            f"chunks={len(embedded_chunks)} | "
            f"embedding_dims="
            f"{len(embedded_chunks[0].embedding) if embedded_chunks else 0}"
        )

        return embedded_chunks
