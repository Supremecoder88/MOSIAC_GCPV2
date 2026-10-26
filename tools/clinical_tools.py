# tools/clinical_tools.py

import asyncio
import json

from langchain_core.tools import tool

from ingestion.clinical_trials_client import ClinicalTrialsClient
from ingestion.document_parser import DocumentParser
from config.logging_config import setup_logging


logger = setup_logging(__name__)
_parser = DocumentParser()


def _run_async(coroutine):
    """Run an async operation synchronously."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


@tool
def fetch_study_details(nct_id: str) -> str:
    """Fetch the latest details for a clinical trial."""
    logger.info(f"Tool called: fetch_study_details | nct_id={nct_id}")

    async def _fetch():
        async with ClinicalTrialsClient() as client:
            return await client.fetch_study(nct_id=nct_id)

    try:
        raw_study = _run_async(_fetch())

        if raw_study is None:
            return json.dumps({
                "found": False,
                "nct_id": nct_id,
                "message": (
                    f"Study {nct_id} was not found on ClinicalTrials.gov. "
                    "Check the NCT ID or whether the study is available."
                ),
            }, indent=2)

        parsed_study = _parser.parse_study(raw=raw_study)

        if parsed_study is None:
            return json.dumps({
                "found": False,
                "nct_id": nct_id,
                "message": "Study was found but could not be parsed.",
            }, indent=2)

        study_dict = parsed_study.model_dump()
        study_dict.pop("raw_data", None)

        return json.dumps({
            "found": True,
            "nct_id": nct_id,
            "study": study_dict,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"fetch_study_details failed | nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "found": False,
            "error": str(e),
            "nct_id": nct_id,
        })


@tool
def search_studies_by_condition(
    condition: str,
    max_results: int = 10,
    status_filter: str = "COMPLETED",
) -> str:
    """Search live clinical trials by medical condition."""
    logger.info(
        f"Tool called: search_studies_by_condition | "
        f"condition={condition} | max_results={max_results}"
    )

    async def _search():
        async with ClinicalTrialsClient() as client:
            return await client.search_studies(
                condition=condition,
                status=[status_filter] if status_filter else None,
                max_results=min(max_results, 50),
            )

    try:
        raw_studies = _run_async(_search())

        if not raw_studies:
            return json.dumps({
                "studies": [],
                "count": 0,
                "message": (
                    f"No studies found for condition '{condition}' "
                    f"with status '{status_filter}'."
                ),
            }, indent=2)

        parsed_studies = _parser.parse_studies(
            raw_studies=raw_studies
        )
        studies_list = []

        for study in parsed_studies:
            study_dict = study.model_dump()
            study_dict.pop("raw_data", None)
            study_dict.pop("protocol_amendments", None)
            studies_list.append(study_dict)

        return json.dumps({
            "studies": studies_list,
            "count": len(studies_list),
            "condition": condition,
            "status_filter": status_filter,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"search_studies_by_condition failed | "
            f"condition={condition} | error={e}"
        )
        return json.dumps({
            "error": str(e),
            "studies": [],
            "count": 0,
        })


@tool
def check_results_posted(nct_id: str) -> str:
    """Check the current results-posting status of a trial."""
    logger.info(f"Tool called: check_results_posted | nct_id={nct_id}")

    async def _check():
        async with ClinicalTrialsClient() as client:
            return await client.fetch_study(nct_id=nct_id)

    try:
        raw_study = _run_async(_check())

        if raw_study is None:
            return json.dumps({
                "nct_id": nct_id,
                "found": False,
                "message": f"Study {nct_id} not found on ClinicalTrials.gov.",
            }, indent=2)

        parsed = _parser.parse_study(raw=raw_study)

        if parsed is None:
            return json.dumps({
                "nct_id": nct_id,
                "found": False,
                "message": "Could not parse study response.",
            }, indent=2)

        result = {
            "nct_id": parsed.nct_id,
            "found": True,
            "results_posted": parsed.results_posted,
            "status": parsed.status,
            "completion_date": parsed.completion_date,
            "sponsor": parsed.sponsor,
        }

        # Check whether a completed study is overdue.
        if (
            not parsed.results_posted
            and parsed.status == "COMPLETED"
            and parsed.completion_date
        ):
            try:
                from datetime import datetime

                completion = datetime.strptime(
                    parsed.completion_date[:7],
                    "%Y-%m",
                )
                now = datetime.utcnow()

                months_since_completion = (
                    (now.year - completion.year) * 12
                    + now.month - completion.month
                )
                months_overdue = months_since_completion - 12

                if months_overdue > 0:
                    result["months_overdue"] = months_overdue
                    result["years_overdue"] = round(
                        months_overdue / 12, 1
                    )
                    result["is_violation"] = True

            except ValueError:
                pass

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"check_results_posted failed | nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error": str(e),
            "found": False,
        })


@tool
def get_study_amendments(nct_id: str) -> str:
    """Fetch protocol amendments for a clinical trial."""
    logger.info(f"Tool called: get_study_amendments | nct_id={nct_id}")

    async def _fetch():
        async with ClinicalTrialsClient() as client:
            return await client.fetch_study(nct_id=nct_id)

    try:
        raw_study = _run_async(_fetch())

        if raw_study is None:
            return json.dumps({
                "nct_id": nct_id,
                "found": False,
                "amendments": [],
            }, indent=2)

        parsed = _parser.parse_study(raw=raw_study)

        if parsed is None:
            return json.dumps({
                "nct_id": nct_id,
                "found": False,
                "amendments": [],
                "message": "Could not parse study.",
            }, indent=2)

        return json.dumps({
            "nct_id": parsed.nct_id,
            "found": True,
            "title": parsed.title,
            "sponsor": parsed.sponsor,
            "start_date": parsed.start_date,
            "completion_date": parsed.completion_date,
            "primary_outcome": parsed.primary_outcome,
            "amendments": parsed.protocol_amendments,
            "amendment_count": len(parsed.protocol_amendments),
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"get_study_amendments failed | nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error": str(e),
            "found": False,
        })


# Export tools for agent assignment.
ALL_CLINICAL_TOOLS = [
    fetch_study_details,
    search_studies_by_condition,
    check_results_posted,
    get_study_amendments,
]