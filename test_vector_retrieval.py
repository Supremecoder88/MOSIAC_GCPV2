
import asyncio

from openai import AsyncOpenAI

from config.settings import settings
from processing.vector_store import VectorStore


async def main():
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    store = VectorStore()

    test_queries = [
    "Survival outcomes after neoadjuvant chemotherapy with trastuzumab or lapatinib in HER2-positive breast cancer patients",
    ]

    try:
        for query_text in test_queries:
            print("\n" + "=" * 75)
            print(f"QUERY: {query_text}")
            print("=" * 75)

            response = await client.embeddings.create(
                model=settings.openai_embedding_model,
                input=query_text,
            )

            query_embedding = response.data[0].embedding

            # Search both studies and papers together.
            results = await store.search(
                query_embedding=query_embedding,
                top_k=10,
                max_distance=None,
            )

            print(f"\nCombined results: {len(results)}")

            for result in results:
                print("\n-----------------------------------")
                print(f"Source:     {result['source']}")
                print(f"NCT ID:     {result['nct_id']}")
                print(f"PMID:       {result['pmid']}")
                print(f"Distance:   {result['distance']:.4f}")
                print(f"Text:       {result['chunk_text'][:350]}")

            # Verify each source independently.
            for source in ["study", "paper"]:
                filtered = await store.search(
                    query_embedding=query_embedding,
                    top_k=5,
                    source_filter=source,
                    max_distance=None,
                )

                print(
                    f"\n{source.upper()}-ONLY RESULTS: "
                    f"{len(filtered)}"
                )

                for result in filtered:
                    print(
                        f"  Distance={result['distance']:.4f} | "
                        f"NCT={result['nct_id']} | "
                        f"PMID={result['pmid']}"
                    )

            # Compare results using the current 0.40 threshold.
            filtered_results = await store.search(
                query_embedding=query_embedding,
                top_k=10,
                max_distance=0.40,
            )

            print(
                f"\nResults with distance <= 0.40: "
                f"{len(filtered_results)}"
            )

    finally:
        await store.close()
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
