from __future__ import annotations

import html
import os
import re
import xml.etree.ElementTree as ET
from typing import Iterable

import httpx

from .schemas import PaperCandidate


def _clean_text(value: str | None) -> str | None:
    if not value:
        return None
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def dedupe_candidates(candidates: Iterable[PaperCandidate]) -> tuple[list[PaperCandidate], int]:
    seen: set[str] = set()
    unique: list[PaperCandidate] = []
    duplicates = 0
    for candidate in candidates:
        key = candidate.stable_id
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(candidate)
    return unique, duplicates


async def search_semantic_scholar(
    client: httpx.AsyncClient, query: str, limit: int
) -> list[PaperCandidate]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    params = {
        "query": query,
        "limit": limit,
        "fields": ",".join(
            [
                "title",
                "abstract",
                "authors",
                "year",
                "venue",
                "externalIds",
                "url",
                "openAccessPdf",
                "fieldsOfStudy",
                "citationCount",
            ]
        ),
    }
    response = await client.get(url, params=params, headers=headers)
    response.raise_for_status()
    payload = response.json()
    results: list[PaperCandidate] = []
    for item in payload.get("data", []):
        title = _clean_text(item.get("title"))
        if not title:
            continue
        external = item.get("externalIds") or {}
        pdf = item.get("openAccessPdf") or {}
        results.append(
            PaperCandidate(
                source="semantic_scholar",
                source_id=item.get("paperId"),
                title=title,
                abstract=_clean_text(item.get("abstract")),
                year=item.get("year"),
                authors=[a.get("name", "") for a in item.get("authors", []) if a.get("name")],
                journal=item.get("venue"),
                doi=external.get("DOI"),
                url=item.get("url"),
                pdf_url=pdf.get("url"),
                keywords=item.get("fieldsOfStudy") or [],
                citation_count=item.get("citationCount"),
                raw=item,
            )
        )
    return results


async def search_crossref(client: httpx.AsyncClient, query: str, limit: int) -> list[PaperCandidate]:
    params = {
        "query": query,
        "rows": limit,
        "filter": "type:journal-article",
    }
    mailto = os.getenv("NCBI_EMAIL")
    if mailto:
        params["mailto"] = mailto
    response = await client.get("https://api.crossref.org/works", params=params)
    response.raise_for_status()
    items = response.json().get("message", {}).get("items", [])
    results: list[PaperCandidate] = []
    for item in items:
        title = _clean_text((item.get("title") or [None])[0])
        if not title:
            continue
        published = item.get("published-print") or item.get("published-online") or {}
        date_parts = published.get("date-parts") or []
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        pdf_url = None
        for link in item.get("link") or []:
            if "pdf" in (link.get("content-type") or "").lower():
                pdf_url = link.get("URL")
                break
        authors = []
        for author in item.get("author") or []:
            name = " ".join([author.get("given", ""), author.get("family", "")]).strip()
            if name:
                authors.append(name)
        results.append(
            PaperCandidate(
                source="crossref",
                source_id=item.get("DOI"),
                title=title,
                abstract=_clean_text(item.get("abstract")),
                year=year,
                authors=authors,
                journal=_clean_text((item.get("container-title") or [None])[0]),
                doi=item.get("DOI"),
                url=item.get("URL"),
                pdf_url=pdf_url,
                keywords=item.get("subject") or [],
                raw=item,
            )
        )
    return results


async def search_pubmed(client: httpx.AsyncClient, query: str, limit: int) -> list[PaperCandidate]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": limit,
        "retmode": "json",
        "sort": "relevance",
    }
    email = os.getenv("NCBI_EMAIL")
    if email:
        params["email"] = email
    search_response = await client.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params
    )
    search_response.raise_for_status()
    ids = search_response.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    fetch_response = await client.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
    )
    fetch_response.raise_for_status()
    root = ET.fromstring(fetch_response.text)
    results: list[PaperCandidate] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = _clean_text(article.findtext(".//ArticleTitle"))
        if not title:
            continue
        abstract_parts = [
            _clean_text("".join(part.itertext()))
            for part in article.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(part for part in abstract_parts if part) or None
        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName") or ""
            fore = author.findtext("ForeName") or ""
            name = " ".join([fore, last]).strip()
            if name:
                authors.append(name)
        doi = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = article_id.text
                break
        year_text = article.findtext(".//JournalIssue/PubDate/Year")
        year = int(year_text) if year_text and year_text.isdigit() else None
        journal = _clean_text(article.findtext(".//Journal/Title"))
        results.append(
            PaperCandidate(
                source="pubmed",
                source_id=pmid,
                title=title,
                abstract=abstract,
                year=year,
                authors=authors,
                journal=journal,
                doi=doi,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                raw={"pmid": pmid},
            )
        )
    return results


async def search_database(
    client: httpx.AsyncClient, database: str, query: str, limit: int
) -> list[PaperCandidate]:
    if database == "semantic_scholar":
        return await search_semantic_scholar(client, query, limit)
    if database == "crossref":
        return await search_crossref(client, query, limit)
    if database == "pubmed":
        return await search_pubmed(client, query, limit)
    raise ValueError(f"Unsupported database: {database}")
