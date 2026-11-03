
import asyncio
import json
from typing import Any

from google.cloud import storage

from config.settings import settings
from config.logging_config import setup_logging
from ingestion.document_parser import ParsedPaper, ParsedStudy


logger = setup_logging(__name__)

PREFIX_RAW_STUDIES = "raw/studies"
PREFIX_RAW_PAPERS = "raw/papers"
PREFIX_PROCESSED_STUDIES = "processed/studies"
PREFIX_PROCESSED_PAPERS = "processed/papers"


class GCSStore:
    """Save and load MOSAIC data using Google Cloud Storage."""

    def __init__(self):
        self._client = storage.Client(
            project=settings.gcp_project_id
        )
        self._bucket = self._client.bucket(
            settings.gcs_bucket_name
        )

        logger.info(
            f"GCSStore initialized | "
            f"bucket={settings.gcs_bucket_name} | "
            f"project={settings.gcp_project_id}"
        )

    async def save_raw_study(
        self,
        nct_id: str,
        data: dict[str, Any],
    ) -> str:
        """Save a raw ClinicalTrials.gov study response."""

        gcs_path = f"{PREFIX_RAW_STUDIES}/{nct_id}.json"

        await self._upload_json(
            path=gcs_path,
            data=data,
        )

        logger.info(
            f"Saved raw study | nct_id={nct_id} | path={gcs_path}"
        )

        return gcs_path

    async def save_raw_paper(
        self,
        pmid: str,
        data: dict[str, Any],
    ) -> str:
        """Save a raw PubMed paper response."""

        gcs_path = f"{PREFIX_RAW_PAPERS}/{pmid}.json"

        await self._upload_json(
            path=gcs_path,
            data=data,
        )

        logger.info(
            f"Saved raw paper | pmid={pmid} | path={gcs_path}"
        )

        return gcs_path

    async def save_parsed_study(
        self,
        study: ParsedStudy,
    ) -> str:
        """Save a parsed clinical study."""

        gcs_path = f"{PREFIX_PROCESSED_STUDIES}/{study.nct_id}.json"

        await self._upload_json(
            path=gcs_path,
            data=study.model_dump(),
        )

        logger.info(
            f"Saved parsed study | "
            f"nct_id={study.nct_id} | path={gcs_path}"
        )

        return gcs_path

    async def save_parsed_paper(
        self,
        paper: ParsedPaper,
    ) -> str:
        """Save a parsed research paper."""

        gcs_path = f"{PREFIX_PROCESSED_PAPERS}/{paper.pmid}.json"

        await self._upload_json(
            path=gcs_path,
            data=paper.model_dump(),
        )

        logger.info(
            f"Saved parsed paper | "
            f"pmid={paper.pmid} | path={gcs_path}"
        )

        return gcs_path

    async def load_parsed_study(
        self,
        nct_id: str,
    ) -> ParsedStudy | None:
        """Load a parsed study from GCS."""

        gcs_path = f"{PREFIX_PROCESSED_STUDIES}/{nct_id}.json"

        data = await self._download_json(path=gcs_path)

        if not data:
            return None

        return ParsedStudy(**data)

    async def load_parsed_paper(
        self,
        pmid: str,
    ) -> ParsedPaper | None:
        """Load a parsed PubMed paper from GCS."""

        gcs_path = f"{PREFIX_PROCESSED_PAPERS}/{pmid}.json"

        data = await self._download_json(path=gcs_path)

        if not data:
            return None

        return ParsedPaper(**data)

    async def list_processed_studies(self) -> list[str]:
        """Return all processed clinical trial NCT IDs."""

        blob_names = await asyncio.to_thread(
            self._list_blob_names,
            PREFIX_PROCESSED_STUDIES,
        )

        nct_ids = [
            name.rsplit("/", 1)[-1][:-5]
            for name in blob_names
            if name.endswith(".json")
        ]

        logger.info(
            f"Listed processed studies | count={len(nct_ids)}"
        )

        return nct_ids

    async def list_processed_papers(self) -> list[str]:
        """Return all processed PubMed paper PMIDs."""

        blob_names = await asyncio.to_thread(
            self._list_blob_names,
            PREFIX_PROCESSED_PAPERS,
        )

        pmids = [
            name.rsplit("/", 1)[-1][:-5]
            for name in blob_names
            if name.endswith(".json")
        ]

        logger.info(
            f"Listed processed papers | count={len(pmids)}"
        )

        return pmids

    def _list_blob_names(self, prefix: str) -> list[str]:
        """List blob names synchronously for execution in a worker thread."""

        return [
            blob.name
            for blob in self._bucket.list_blobs(
                prefix=f"{prefix}/"
            )
        ]

    async def _upload_json(
        self,
        path: str,
        data: dict[str, Any],
    ) -> None:
        """Serialize a dictionary and upload it as JSON."""

        json_bytes = json.dumps(
            data,
            indent=2,
            default=str,
        ).encode("utf-8")

        blob = self._bucket.blob(path)

        await asyncio.to_thread(
            blob.upload_from_string,
            json_bytes,
            content_type="application/json",
        )

    async def _download_json(
        self,
        path: str,
    ) -> dict[str, Any] | None:
        """Download a JSON file and parse it into a dictionary."""

        try:
            blob = self._bucket.blob(path)

            json_bytes = await asyncio.to_thread(
                blob.download_as_bytes
            )

            return json.loads(
                json_bytes.decode("utf-8")
            )

        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                logger.warning(
                    f"File not found in GCS | path={path}"
                )
            else:
                logger.error(
                    f"Failed to download from GCS | "
                    f"path={path} | error={e}"
                )

            return None
