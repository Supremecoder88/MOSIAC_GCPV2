import asyncio
import httpx
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


BASE_URL = settings.pubmed_base_url
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
FETCH_BATCH_SIZE = 20
RATE_LIMIT_SLEEP = 0.4


class PubMedClient:

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "PubMedClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
            headers={"Accept": "application/json"},
        )
        logger.info("PubMed client opened")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            logger.info("PubMed client closed")

    async def fetch_papers_for_trial(
        self,
        nct_id: str,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:

        logger.info(f"Fetching PubMed papers | nct_id={nct_id}")

        paper_ids = await self._search_paper_ids(
            nct_id=nct_id,
            max_results=max_results,
        )

        if not paper_ids:
            logger.info(f"No PubMed papers found | nct_id={nct_id}")
            return []

        logger.info(
            f"Found {len(paper_ids)} paper IDs | nct_id={nct_id}"
        )

        papers = await self._fetch_paper_details(
            paper_ids=paper_ids
        )

        logger.info(
            f"PubMed fetch complete | "
            f"nct_id={nct_id} | "
            f"papers_returned={len(papers)}"
        )

        return papers

    
    async def search_papers_by_query(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search PubMed for papers matching a general topic or query."""

        logger.info(f"Searching PubMed by topic | query={query}")

        paper_ids = await self._search_paper_ids_by_query(
            query=query,
            max_results=max_results,
        )

        if not paper_ids:
            logger.info(f"No papers found | query={query}")
            return []

        papers = await self._fetch_paper_details(
            paper_ids=paper_ids
        )

        logger.info(
            f"PubMed topic search complete | "
            f"query={query} | papers_returned={len(papers)}"
        )

        return papers


    async def fetch_papers_for_trials(
        self,
        nct_ids: list[str],
        max_per_trial: int = 20,
    ) -> dict[str, list[dict[str, Any]]]:

        results: dict[str, list[dict[str, Any]]] = {}

        for i, nct_id in enumerate(nct_ids):

            logger.info(
                f"Processing trial {i + 1}/{len(nct_ids)} | "
                f"nct_id={nct_id}"
            )

            papers = await self.fetch_papers_for_trial(
                nct_id=nct_id,
                max_results=max_per_trial,
            )

            results[nct_id] = papers

            if i < len(nct_ids) - 1:
                await asyncio.sleep(RATE_LIMIT_SLEEP)

        return results

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=8,
        ),
        retry=retry_if_exception_type(
            (
                httpx.TimeoutException,
                httpx.ConnectError,
            )
        ),
    )
    async def _search_paper_ids(
        self,
        nct_id: str,
        max_results: int,
    ) -> list[str]:

        try:
            response = await self._client.get(
                f"{BASE_URL}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": f"{nct_id}[si]",
                    "retmax": max_results,
                    "retmode": "json",
                    "usehistory": "n",
                },
            )

            response.raise_for_status()

            data = response.json()

            id_list = data.get(
                "esearchresult",
                {}
            ).get(
                "idlist",
                []
            )

            await asyncio.sleep(RATE_LIMIT_SLEEP)

            return id_list

        except httpx.TimeoutException:
            logger.warning(
                f"Timeout searching PubMed | "
                f"nct_id={nct_id} — retrying..."
            )
            raise

        except httpx.ConnectError:
            logger.warning(
                f"Connection error searching PubMed | "
                f"nct_id={nct_id} — retrying..."
            )
            raise

        except Exception as e:
            logger.error(
                f"Failed to search PubMed | "
                f"nct_id={nct_id} | error={e}"
            )
            return []

    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=8,
        ),
        retry=retry_if_exception_type(
            (
                httpx.TimeoutException,
                httpx.ConnectError,
            )
        ),
    )
    async def _search_paper_ids_by_query(
        self,
        query: str,
        max_results: int,
    ) -> list[str]:
        """Search PubMed using a general query instead of an NCT ID."""

        try:
            response = await self._client.get(
                f"{BASE_URL}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                    "usehistory": "n",
                },
            )

            response.raise_for_status()

            data = response.json()

            id_list = data.get(
                "esearchresult", {}
            ).get(
                "idlist", []
            )

            await asyncio.sleep(RATE_LIMIT_SLEEP)

            return id_list

        except httpx.TimeoutException:
            logger.warning(
                f"Timeout searching PubMed by topic | "
                f"query={query} — retrying..."
            )
            raise

        except httpx.ConnectError:
            logger.warning(
                f"Connection error searching PubMed by topic | "
                f"query={query} — retrying..."
            )
            raise

        except Exception as e:
            logger.error(
                f"Failed to search PubMed by topic | "
                f"query={query} | error={e}"
            )
            return []


    async def _fetch_paper_details(
        self,
        paper_ids: list[str],
    ) -> list[dict[str, Any]]:

        all_papers: list[dict[str, Any]] = []

        batches = [
            paper_ids[i:i + FETCH_BATCH_SIZE]
            for i in range(0, len(paper_ids), FETCH_BATCH_SIZE)
        ]

        for batch_num, batch in enumerate(batches):

            logger.info(
                f"Fetching paper details | "
                f"batch={batch_num + 1}/{len(batches)} | "
                f"papers_in_batch={len(batch)}"
            )

            batch_papers = await self._fetch_batch(
                paper_ids=batch
            )

            all_papers.extend(batch_papers)

            if batch_num < len(batches) - 1:
                await asyncio.sleep(RATE_LIMIT_SLEEP)

        return all_papers

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=8,
        ),
        retry=retry_if_exception_type(
            (
                httpx.TimeoutException,
                httpx.ConnectError,
            )
        ),
    )
    async def _fetch_batch(
        self,
        paper_ids: list[str],
    ) -> list[dict[str, Any]]:

        try:
            response = await self._client.get(
                f"{BASE_URL}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(paper_ids),
                    "retmode": "xml",
                    "rettype": "abstract",
                },
            )

            response.raise_for_status()

            papers = self._parse_xml_response(
                xml_text=response.text
            )

            await asyncio.sleep(RATE_LIMIT_SLEEP)

            return papers

        except httpx.TimeoutException:
            logger.warning(
                "Timeout fetching paper batch — retrying..."
            )
            raise

        except httpx.ConnectError:
            logger.warning(
                "Connection error fetching paper batch — retrying..."
            )
            raise

        except Exception as e:
            logger.error(
                f"Failed to fetch paper batch | error={e}"
            )
            return []

    def _parse_xml_response(
        self,
        xml_text: str,
    ) -> list[dict[str, Any]]:

        import xml.etree.ElementTree as ET

        papers: list[dict[str, Any]] = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):

                paper = self._extract_paper_fields(
                    article
                )

                if paper:
                    papers.append(paper)

        except ET.ParseError as e:
            logger.error(
                f"Failed to parse PubMed XML response | error={e}"
            )

        return papers

    def _extract_paper_fields(
        self,
        article_element: Any,
    ) -> dict[str, Any] | None:

        import xml.etree.ElementTree as ET

        def get_text(
            element: Any,
            path: str,
            default: str = "",
        ) -> str:

            node = element.find(path)

            return (
                node.text.strip()
                if node is not None and node.text
                else default
            )

        try:
            pmid = get_text(
                article_element,
                ".//PMID"
            )

            title = get_text(
                article_element,
                ".//ArticleTitle"
            )

            abstract_texts = article_element.findall(
                ".//AbstractText"
            )

            abstract = " ".join(
                node.text.strip()
                for node in abstract_texts
                if node.text
            )

            pub_year = get_text(
                article_element,
                ".//PubDate/Year"
            )

            pub_month = get_text(
                article_element,
                ".//PubDate/Month",
                "01"
            )

            pub_date = (
                f"{pub_year}-{pub_month}"
                if pub_year
                else ""
            )

            journal = get_text(
                article_element,
                ".//Journal/Title"
            )

            author_elements = article_element.findall(
                ".//Author"
            )

            authors = []

            for author in author_elements:

                last = get_text(
                    author,
                    "LastName"
                )

                first = get_text(
                    author,
                    "ForeName"
                )

                if last:
                    authors.append(
                        f"{last}, {first}".strip(", ")
                    )

            nct_ids_referenced = [
                id_elem.text.strip()
                for id_elem in article_element.findall(
                    ".//DataBankList/"
                    "DataBank/"
                    "AccessionNumberList/"
                    "AccessionNumber"
                )
                if (
                    id_elem.text
                    and id_elem.text.strip().startswith("NCT")
                )
            ]

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "pub_date": pub_date,
                "authors": authors,
                "nct_ids_referenced": nct_ids_referenced,
                "source": "pubmed",
            }

        except Exception as e:
            logger.error(
                f"Failed to extract paper fields | "
                f"pmid=UNKNOWN | error={e}"
            )

            return None

