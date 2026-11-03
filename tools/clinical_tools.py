# tools/clinical_tools.py

import json
from datetime import datetime

from langchain_core.tools import tool

from ingestion.clinical_trials_client import ClinicalTrialsClient
from ingestion.document_parser import DocumentParser
from config.logging_config import setup_logging


logger = setup_logging(__name__)
_parser = DocumentParser()


# ==========================================================
# COMPLETION DATE EXTRACTION
# ==========================================================

def _extract_trial_dates(raw_study: dict) -> dict:
    """
    Extract primary completion and overall study completion
    dates separately from the ClinicalTrials.gov response.
    """

    if not isinstance(raw_study, dict):
        return {}

    if "protocolSection" not in raw_study:
        raw_study = raw_study.get("study", raw_study)

    status_module = (
        raw_study.get("protocolSection", {})
        .get("statusModule", {})
    )

    primary = status_module.get(
        "primaryCompletionDateStruct", {}
    ) or {}

    overall = status_module.get(
        "completionDateStruct", {}
    ) or {}

    return {
        "primary_completion_date": primary.get("date"),
        "primary_completion_date_type": primary.get("type"),
        "study_completion_date": overall.get("date"),
        "study_completion_date_type": overall.get("type"),
    }


# ==========================================================
# FETCH LATEST STUDY DETAILS
# ==========================================================

@tool
async def fetch_study_details(nct_id: str) -> str:
    """Fetch the latest details for a clinical trial."""

    logger.info(
        f"Tool called: fetch_study_details | nct_id={nct_id}"
    )

    try:
        async with ClinicalTrialsClient() as client:
            raw_study = await client.fetch_study(nct_id=nct_id)

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

        # Preserve both completion dates separately.
        study_dict.update(_extract_trial_dates(raw_study))

        return json.dumps({
            "found": True,
            "nct_id": nct_id,
            "study": study_dict,
        }, indent=2, default=str)

    except Exception as e:
        logger.exception(
            f"fetch_study_details failed | "
            f"nct_id={nct_id} | error={e}"
        )

        return json.dumps({
            "found": False,
            "error": str(e),
            "nct_id": nct_id,
        })


# ==========================================================
# SEARCH STUDIES BY CONDITION
# ==========================================================

@tool
async def search_studies_by_condition(
    condition: str,
    max_results: int = 10,
    status_filter: str = "COMPLETED",
) -> str:
    """Search live clinical trials by medical condition."""

    logger.info(
        f"Tool called: search_studies_by_condition | "
        f"condition={condition} | max_results={max_results}"
    )

    try:
        async with ClinicalTrialsClient() as client:
            raw_studies = await client.search_studies(
                condition=condition,
                status=[status_filter] if status_filter else None,
                max_results=min(max_results, 50),
            )

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
        logger.exception(
            f"search_studies_by_condition failed | "
            f"condition={condition} | error={e}"
        )

        return json.dumps({
            "error": str(e),
            "studies": [],
            "count": 0,
        })


# ==========================================================
# CHECK RESULTS POSTING STATUS
# ==========================================================

@tool
async def check_results_posted(nct_id: str) -> str:
    """Check current results status and reporting deadline."""

    logger.info(
        f"Tool called: check_results_posted | nct_id={nct_id}"
    )

    try:
        async with ClinicalTrialsClient() as client:
            raw_study = await client.fetch_study(nct_id=nct_id)

        if raw_study is None:
            return json.dumps({
                "nct_id": nct_id,
                "found": False,
                "message": "Study not found on ClinicalTrials.gov.",
            }, indent=2)

        parsed = _parser.parse_study(raw=raw_study)

        if parsed is None:
            return json.dumps({
                "nct_id": nct_id,
                "found": False,
                "message": "Could not parse study response.",
            }, indent=2)

        dates = _extract_trial_dates(raw_study)

        primary_date = dates.get("primary_completion_date")
        primary_type = dates.get("primary_completion_date_type")

        study_date = dates.get("study_completion_date")
        study_type = dates.get("study_completion_date_type")

        result = {
            "nct_id": parsed.nct_id,
            "found": True,
            "results_posted": parsed.results_posted,
            "status": parsed.status,

            # Keep both dates distinct.
            "primary_completion_date": primary_date,
            "primary_completion_date_type": primary_type,
            "study_completion_date": study_date,
            "study_completion_date_type": study_type,

            # Reporting deadline uses primary completion.
            "reporting_deadline_basis": "primary_completion_date",

            "sponsor": parsed.sponsor,
        }

        # Calculate the reporting deadline only when the
        # primary completion date is explicitly ACTUAL.
        if (
            not parsed.results_posted
            and parsed.status == "COMPLETED"
            and primary_date
            and primary_type == "ACTUAL"
        ):
            try:
                date_formats = [
                    "%Y-%m-%d",
                    "%Y-%m",
                    "%B %Y",
                    "%b %Y",
                ]

                completion = None

                for date_format in date_formats:
                    try:
                        completion = datetime.strptime(
                            primary_date,
                            date_format
                        )
                        break
                    except ValueError:
                        continue

                if completion is not None:
                    now = datetime.utcnow()

                    months_since_completion = (
                        (now.year - completion.year) * 12
                        + now.month - completion.month
                    )

                    months_overdue = months_since_completion - 12

                    result["months_since_primary_completion"] = (
                        months_since_completion
                    )

                    if months_overdue > 0:
                        result["months_overdue"] = months_overdue

                        result["years_overdue"] = round(
                            months_overdue / 12, 1
                        )

                        result["is_violation"] = True
                    else:
                        result["is_violation"] = False

            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not calculate reporting deadline "
                    f"| nct_id={nct_id} | error={e}"
                )

        return json.dumps(
            result,
            indent=2,
            default=str
        )

    except Exception as e:
        logger.exception(
            f"check_results_posted failed | "
            f"nct_id={nct_id} | error={e}"
        )

        return json.dumps({
            "nct_id": nct_id,
            "error": str(e),
            "found": False,
        })


# ==========================================================
# COMPARE HISTORICAL STUDY VERSIONS
# ==========================================================

@tool
async def get_study_amendments(nct_id: str) -> str:
    """
    Compare historical ClinicalTrials.gov versions.

    Detect changes in primary and secondary outcomes.

    Clearly report when historical verification is unavailable.
    """

    logger.info(
        f"Tool called: get_study_amendments | nct_id={nct_id}"
    )

    try:
        async with ClinicalTrialsClient() as client:
            history = await client.fetch_study_history(
                nct_id=nct_id
            )

        # --------------------------------------------------
        # 1. CHECK HISTORY AVAILABILITY
        # --------------------------------------------------

        if not history.get("history_available"):
            return json.dumps({
                "nct_id": nct_id,
                "found": None,
                "verification_status": "unavailable",
                "history_available": False,
                "outcome_changes_verified": False,
                "outcome_changes_detected": None,
                "versions_checked": 0,
                "amendments": [],
                "message": (
                    "Historical versions could not be retrieved. "
                    "No conclusion about historical outcome changes "
                    "can be made."
                ),
                "errors": history.get("errors", []),
            }, indent=2, default=str)

        # --------------------------------------------------
        # 2. GET HISTORICAL VERSIONS
        # --------------------------------------------------

        versions = history.get("versions", [])

        if not versions:
            return json.dumps({
                "nct_id": nct_id,
                "found": None,
                "verification_status": "insufficient_data",
                "history_available": True,
                "outcome_changes_verified": False,
                "outcome_changes_detected": None,
                "versions_checked": 0,
                "amendments": [],
                "message": (
                    "History metadata was available, but no "
                    "historical study snapshots were retrieved."
                ),
                "errors": history.get("errors", []),
            }, indent=2, default=str)

        # --------------------------------------------------
        # 3. EXTRACT OUTCOMES FROM A HISTORICAL VERSION
        # --------------------------------------------------

        def get_outcomes(study):
            protocol = study.get("protocolSection", {})

            outcomes = protocol.get(
                "outcomesModule", {}
            )

            return {
                "primary": outcomes.get(
                    "primaryOutcomes", []
                ),
                "secondary": outcomes.get(
                    "secondaryOutcomes", []
                ),
            }

        # --------------------------------------------------
        # 4. SORT HISTORICAL VERSIONS
        # --------------------------------------------------

        ordered_versions = sorted(
            versions,
            key=lambda item: item["version_number"]
        )

        amendments = []

        # --------------------------------------------------
        # 5. COMPARE CONSECUTIVE VERSIONS
        # --------------------------------------------------

        for previous, current in zip(
            ordered_versions,
            ordered_versions[1:]
        ):

            previous_outcomes = get_outcomes(
                previous["study"]
            )

            current_outcomes = get_outcomes(
                current["study"]
            )

            changed_fields = []

            if (
                previous_outcomes["primary"]
                != current_outcomes["primary"]
            ):
                changed_fields.append(
                    "primary_outcomes"
                )

            if (
                previous_outcomes["secondary"]
                != current_outcomes["secondary"]
            ):
                changed_fields.append(
                    "secondary_outcomes"
                )

            if changed_fields:
                amendments.append({
                    "from_version": previous["version_number"],
                    "to_version": current["version_number"],
                    "from_date": previous.get("version_date"),
                    "to_date": current.get("version_date"),
                    "changed_fields": changed_fields,
                    "previous_outcomes": previous_outcomes,
                    "current_outcomes": current_outcomes,
                })

        # --------------------------------------------------
        # 6. DETERMINE VERIFICATION STATUS
        # --------------------------------------------------

        comparison_complete = (
            len(ordered_versions) >= 2
            and not history.get("errors")
        )

        if comparison_complete:
            verification_status = "verified"
            outcome_changes_detected = bool(amendments)

        elif amendments:
            verification_status = "partial"
            outcome_changes_detected = True

        else:
            verification_status = "partial"
            outcome_changes_detected = None

        # --------------------------------------------------
        # 7. PARSE LATEST HISTORICAL VERSION
        # --------------------------------------------------

        latest_study = ordered_versions[-1]["study"]

        parsed_latest = _parser.parse_study(
            raw=latest_study
        )

        title = None
        sponsor = None

        if parsed_latest:
            title = parsed_latest.title
            sponsor = parsed_latest.sponsor

        # --------------------------------------------------
        # 8. BUILD RESPONSE
        # --------------------------------------------------

        if amendments:
            message = (
                "Outcome changes detected between historical versions."
            )

        elif comparison_complete:
            message = (
                "No primary or secondary outcome changes detected "
                "across the successfully compared versions."
            )

        else:
            message = (
                "Historical comparison is incomplete. "
                "No conclusion about unchanged outcomes can be made."
            )

        result = {
            "nct_id": nct_id,
            "found": True,
            "verification_status": verification_status,
            "history_available": True,
            "outcome_changes_verified": comparison_complete,
            "outcome_changes_detected": outcome_changes_detected,
            "versions_checked": len(ordered_versions),
            "title": title,
            "sponsor": sponsor,
            "amendments": amendments,
            "amendment_count": len(amendments),
            "errors": history.get("errors", []),
            "message": message,
        }

        return json.dumps(
            result,
            indent=2,
            default=str
        )

    except Exception as e:
        logger.exception(
            f"get_study_amendments failed | "
            f"nct_id={nct_id} | error={e}"
        )

        return json.dumps({
            "nct_id": nct_id,
            "found": False,
            "verification_status": "error",
            "outcome_changes_verified": False,
            "outcome_changes_detected": None,
            "amendments": [],
            "error": str(e),
        }, indent=2)


# ==========================================================
# EXPORT TOOLS FOR AGENT ASSIGNMENT
# ==========================================================

ALL_CLINICAL_TOOLS = [
    fetch_study_details,
    search_studies_by_condition,
    check_results_posted,
    get_study_amendments,
]