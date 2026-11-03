
import asyncio
import json

from tools.search_tools import (
    search_studies_by_meaning,
    search_papers_by_meaning,
)


async def main():
    query = (
        "Survival outcomes after neoadjuvant chemotherapy "
        "with trastuzumab or lapatinib in HER2-positive "
        "breast cancer patients"
    )

    try:
        print("\n========== PAPER SEARCH ==========")

        paper_result = await search_papers_by_meaning.ainvoke({
            "query": query,
            "top_k": 5,
        })

        print(json.dumps(json.loads(paper_result), indent=2))

        print("\n========== STUDY SEARCH ==========")

        study_result = await search_studies_by_meaning.ainvoke({
            "query": query,
            "top_k": 5,
        })

        print(json.dumps(json.loads(study_result), indent=2))

    finally:
        from tools.search_tools import _vector_store
        await _vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())
