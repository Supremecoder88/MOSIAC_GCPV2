
import asyncio

from processing.embedder import Embedder
from processing.vector_store import VectorStore


async def main():
    embedder = Embedder()
    vector_store = VectorStore()

    try:
        query = "COVID-19 clinical trials involving remdesivir and treatment outcomes"

        print("\nGenerating query embedding...")

        query_embedding = await embedder.embed_query(query)

        print("Searching Cloud SQL without distance filtering...\n")

        results = await vector_store.search(
            query_embedding=query_embedding,
            top_k=10,
            source_filter="study",
            max_distance=None,
        )

        if not results:
            print("No results found.")
            return

        for index, result in enumerate(results, start=1):
            print(f"\n{'=' * 70}")
            print(f"RESULT {index}")
            print(f"NCT ID: {result['nct_id']}")
            print(f"DISTANCE: {result['distance']:.4f}")
            print(f"SOURCE: {result['source']}")
            print(f"TEXT: {result['chunk_text'][:350]}")

    finally:
        await vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())
