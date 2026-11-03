
import asyncio
import requests

from typing import Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import settings
from config.logging_config import setup_logging


logger = setup_logging(__name__)

BASE_URL = settings.clinical_trials_base_url
PAGE_SIZE = settings.clinical_trials_page_size
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "MOSAIC Clinical Research Integrity System",
}


class ClinicalTrialsClient:

    def __init__(self):
        self._session: requests.Session | None = None

    async def __aenter__(self) -> "ClinicalTrialsClient":
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        logger.info("ClinicalTrials client opened")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            self._session.close()
            logger.info("ClinicalTrials client closed")

    # ======================================================
    # SEARCH STUDIES
    # ======================================================

    async def search_studies(
        self,
        condition: str | None = None,
        intervention: str | None = None,
        sponsor: str | None = None,
        status: list[str] | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:

        all_studies: list[dict[str, Any]] = []
        next_page_token: str | None = None
        page_number = 0

        logger.info(
            f"Searching studies | "
            f"condition={condition} | "
            f"intervention={intervention} | "
            f"sponsor={sponsor} | "
            f"max_results={max_results}"
        )

        while len(all_studies) < max_results:

            page_number += 1

            params = self._build_search_params(
                condition=condition,
                intervention=intervention,
                sponsor=sponsor,
                status=status,
                page_token=next_page_token,
            )

            response_data = await self._fetch_page(params=params)

            if not response_data:
                break

            page_studies = response_data.get("studies", [])

            if not page_studies:
                logger.info("No more studies available")
                break

            all_studies.extend(page_studies)

            logger.info(
                f"Page {page_number} | "
                f"fetched={len(page_studies)} | "
                f"total={len(all_studies)}"
            )

            next_page_token = response_data.get("nextPageToken")

            if not next_page_token:
                break

        all_studies = all_studies[:max_results]

        logger.info(
            f"Search complete | "
            f"total studies={len(all_studies)}"
        )

        return all_studies

    # ======================================================
    # FETCH LATEST STUDY
    # ======================================================

    async def fetch_study(
        self,
        nct_id: str
    ) -> dict[str, Any] | None:

        logger.info(
            f"Fetching latest study | nct_id={nct_id}"
        )

        def _get_study():
            return self._session.get(
                f"{BASE_URL}/studies/{nct_id}",
                timeout=REQUEST_TIMEOUT,
            )

        try:
            response = await asyncio.to_thread(_get_study)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.warning(
                f"Study HTTP error | "
                f"nct_id={nct_id} | "
                f"status={e.response.status_code}"
            )
            return None

        except Exception as e:
            logger.exception(
                f"Failed to fetch study | "
                f"nct_id={nct_id} | error={e}"
            )
            return None

    # ======================================================
    # FETCH HISTORICAL STUDY VERSIONS
    # ======================================================

    async def fetch_study_history(
        self,
        nct_id: str
    ) -> dict[str, Any]:

        logger.info(
            f"Fetching study history | nct_id={nct_id}"
        )

        result = {
            "nct_id": nct_id,
            "history_available": False,
            "versions": [],
            "errors": [],
        }

        # --------------------------------------------------
        # 1. FETCH HISTORY OVERVIEW
        # --------------------------------------------------

        def _get_history_overview():
            return self._session.get(
                f"https://clinicaltrials.gov/api/int/studies/"
                f"{nct_id}",
                params={"history": "true"},
                timeout=REQUEST_TIMEOUT,
            )

        try:
            response = await asyncio.to_thread(
                _get_history_overview
            )

            response.raise_for_status()

            history_data = response.json()

        except Exception as e:

            logger.warning(
                f"History overview unavailable | "
                f"nct_id={nct_id} | error={e}"
            )

            result["errors"].append(
                f"History overview unavailable: {str(e)}"
            )

            return result

        # --------------------------------------------------
        # 2. EXTRACT VERSION METADATA
        # --------------------------------------------------

        history = history_data.get("history", {})

        changes = history.get("changes", [])

        if not changes:

            logger.info(
                f"No historical versions returned | "
                f"nct_id={nct_id}"
            )

            result["history_available"] = True

            return result

        result["history_available"] = True

        # --------------------------------------------------
        # 3. FETCH EACH HISTORICAL VERSION
        # --------------------------------------------------

        for change in changes:

            version_number = change.get("version")

            version_date = change.get("date")

            if version_number is None:
                continue

            def _get_version(version=version_number):
                return self._session.get(
                    f"https://clinicaltrials.gov/api/int/"
                    f"studies/{nct_id}/history/{version}",
                    timeout=REQUEST_TIMEOUT,
                )

            try:

                version_response = await asyncio.to_thread(
                    _get_version
                )

                version_response.raise_for_status()

                version_data = version_response.json()

                study = version_data.get("study")

                if not isinstance(study, dict):
                    raise ValueError(
                        "Historical response did not contain "
                        "a valid study object."
                    )

                actual_version = version_data.get(
                    "studyVersion",
                    version_number
                )

                result["versions"].append({
                    "version_number": actual_version,
                    "version_date": version_date,
                    "study": study,
                })

                logger.info(
                    f"Historical version fetched | "
                    f"nct_id={nct_id} | "
                    f"version={actual_version} | "
                    f"date={version_date}"
                )

            except Exception as e:

                logger.warning(
                    f"Historical version failed | "
                    f"nct_id={nct_id} | "
                    f"version={version_number} | "
                    f"error={e}"
                )

                result["errors"].append({
                    "version": version_number,
                    "error": str(e),
                })

        # --------------------------------------------------
        # 4. SORT VERSIONS
        # --------------------------------------------------

        result["versions"].sort(
            key=lambda item: item["version_number"]
        )

        logger.info(
            f"Study history complete | "
            f"nct_id={nct_id} | "
            f"versions_fetched={len(result['versions'])} | "
            f"errors={len(result['errors'])}"
        )

        return result

    # ======================================================
    # SEARCH PARAMETER BUILDER
    # ======================================================

    def _build_search_params(
        self,
        condition: str | None,
        intervention: str | None,
        sponsor: str | None,
        status: list[str] | None,
        page_token: str | None,
    ) -> dict[str, Any]:

        params: dict[str, Any] = {
            "pageSize": PAGE_SIZE,
            "format": "json",
        }

        if condition:
            params["query.cond"] = condition

        if intervention:
            params["query.intr"] = intervention

        if sponsor:
            params["query.spons"] = sponsor

        if status:
            params["filter.overallStatus"] = "|".join(status)

        if page_token:
            params["pageToken"] = page_token

        return params

    # ======================================================
    # FETCH PAGINATED SEARCH RESULTS
    # ======================================================

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=8
        ),
        retry=retry_if_exception_type(
            (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            )
        ),
    )
    async def _fetch_page(
        self,
        params: dict[str, Any]
    ) -> dict[str, Any] | None:

        def _get():
            return self._session.get(
                f"{BASE_URL}/studies",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

        try:

            response = await asyncio.to_thread(_get)

            response.raise_for_status()

            return response.json()

        except requests.exceptions.Timeout:

            logger.warning(
                f"Request timed out after {REQUEST_TIMEOUT}s"
            )

            raise

        except requests.exceptions.ConnectionError:

            logger.warning("Connection error — retrying")

            raise

        except requests.exceptions.HTTPError as e:

            logger.error(
                f"HTTP error | "
                f"status={e.response.status_code} | "
                f"url={e.response.url}"
            )

            return None

        except Exception as e:

            logger.exception(
                f"Unexpected error fetching page | {e}"
            )

            return None
