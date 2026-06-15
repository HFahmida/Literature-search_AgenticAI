from __future__ import annotations

import json
import os
import re
import ast
from typing import Any

import httpx

from .config import ReviewConfig
from .schemas import ManuscriptDraft, PaperCandidate, StudyExtraction


class OllamaAgent:
    def __init__(self, config: ReviewConfig):
        self.config = config
        self.base_url = os.getenv("OLLAMA_BASE_URL") or config.ollama_base_url
        self.base_url = self.base_url.rstrip("/")

    def check_ready(self) -> None:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "Ollama is not reachable. Install Ollama, start it, then run "
                f"'ollama pull {self.config.extract_model}'. Original error: {exc!r}"
            ) from exc
        models = response.json().get("models", [])
        names = {item.get("name") for item in models}
        if self.config.extract_model not in names:
            raise RuntimeError(
                f"Ollama is running, but model {self.config.extract_model!r} is not installed. "
                f"Run: ollama pull {self.config.extract_model}"
            )

    def extract_paper(self, candidate: PaperCandidate) -> StudyExtraction:
        data = {
            "review_question": self.config.review_question,
            "topic": self.config.topic,
            "inclusion_criteria": self.config.inclusion_criteria,
            "exclusion_criteria": self.config.exclusion_criteria,
            "minimum_relevance_score_for_inclusion": self.config.min_relevance_score,
            "extraction_detail_instructions": self.config.extraction_detail_instructions,
            "custom_extraction_questions": self.config.custom_extraction_questions,
            "paper_metadata": _candidate_for_prompt(candidate),
            "json_template": STUDY_EXTRACTION_TEMPLATE,
        }
        prompt = (
            EXTRACTION_SYSTEM_PROMPT
            + "\n\nReturn only one valid JSON object matching json_template. "
            + "Use double-quoted property names. Do not wrap it in markdown.\n\n"
            + json.dumps(data, ensure_ascii=False)
        )
        payload = self._generate_json(self.config.extract_model, prompt, max_tokens=5000)
        payload.setdefault("paper_id", candidate.stable_id)
        payload.setdefault("title", candidate.title)
        payload.setdefault("doi", candidate.doi)
        payload.setdefault("url", candidate.url)
        payload.setdefault("pdf_url", candidate.pdf_url)
        payload.setdefault("year", candidate.year)
        payload.setdefault("authors", candidate.authors)
        payload.setdefault("journal", candidate.journal)
        _normalize_study_extraction_payload(payload)
        return StudyExtraction.model_validate(payload)

    def draft_manuscript(
        self,
        extractions: list[StudyExtraction],
        run_stats: dict[str, Any],
        final: bool = False,
    ) -> ManuscriptDraft:
        data = {
            "topic": self.config.topic,
            "review_question": self.config.review_question,
            "inclusion_criteria": self.config.inclusion_criteria,
            "exclusion_criteria": self.config.exclusion_criteria,
            "run_stats": run_stats,
            "evidence": [_extraction_for_manuscript(item) for item in extractions],
            "is_final_draft": final,
            "json_template": MANUSCRIPT_TEMPLATE,
        }
        prompt = (
            MANUSCRIPT_SYSTEM_PROMPT
            + "\n\nReturn only one valid JSON object matching json_template. "
            + "Use double-quoted property names. Do not wrap it in markdown.\n\n"
            + json.dumps(data, ensure_ascii=False)
        )
        payload = self._generate_json(self.config.draft_model, prompt, max_tokens=5000)
        _normalize_manuscript_payload(payload)
        return ManuscriptDraft.model_validate(payload)

    def _generate_json(self, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "num_predict": max_tokens,
                    "num_ctx": 8192,
                },
            },
            timeout=900,
        )
        response.raise_for_status()
        text = response.json().get("response", "")
        try:
            return _loads_json_object(text)
        except Exception as exc:
            snippet = text[:700].replace("\n", "\\n")
            raise ValueError(f"Could not parse local model JSON response: {snippet}") from exc


def _loads_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        candidate = match.group(0)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            data = ast.literal_eval(candidate)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from the local model.")
    return data


def _candidate_for_prompt(candidate: PaperCandidate) -> dict[str, Any]:
    return {
        "paper_id": candidate.stable_id,
        "source": candidate.source,
        "source_id": candidate.source_id,
        "title": candidate.title,
        "abstract": candidate.abstract,
        "year": candidate.year,
        "authors": candidate.authors[:12],
        "journal": candidate.journal,
        "doi": candidate.doi,
        "url": candidate.url,
        "pdf_url": candidate.pdf_url,
        "keywords": candidate.keywords[:20],
        "citation_count": candidate.citation_count,
    }


def _normalize_study_extraction_payload(payload: dict[str, Any]) -> None:
    list_fields = [
        "authors",
        "methods",
        "outcomes_measured",
        "main_results",
        "conclusions",
        "limitations",
        "risk_of_bias_notes",
        "extracted_numeric_results",
        "evidence_quotes_or_locations",
        "missing_information",
        "key_takeaways",
    ]
    for field in list_fields:
        if payload.get(field) is None:
            payload[field] = []
        elif isinstance(payload.get(field), str):
            payload[field] = [payload[field]]
    screening = payload.get("screening")
    if isinstance(screening, dict):
        for field in ["matched_inclusion_criteria", "matched_exclusion_criteria"]:
            if screening.get(field) is None:
                screening[field] = []
            elif isinstance(screening.get(field), str):
                screening[field] = [screening[field]]
    for field in [
        "overall_concept_summary",
        "detailed_summary",
        "methods_summary",
        "results_summary",
        "conclusion_summary",
        "relevance_to_review",
    ]:
        if payload.get(field) is None:
            payload[field] = ""
    if not payload.get("overall_concept_summary"):
        payload["overall_concept_summary"] = payload.get("detailed_summary") or payload.get("title") or "No summary available from the supplied metadata."


def _normalize_manuscript_payload(payload: dict[str, Any]) -> None:
    for field in ["references", "notes_for_human_reviewer"]:
        if payload.get(field) is None:
            payload[field] = []
        elif isinstance(payload.get(field), str):
            payload[field] = [payload[field]]
    for field in [
        "title",
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "limitations",
        "conclusions",
        "prisma_flow_summary",
        "tables_markdown",
    ]:
        if payload.get(field) is None:
            payload[field] = ""


def _truncate(text: str | None, max_chars: int = 900) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _extraction_for_manuscript(extraction: StudyExtraction) -> dict[str, Any]:
    return {
        "title": extraction.title,
        "year": extraction.year,
        "authors": extraction.authors[:6],
        "journal": extraction.journal,
        "doi": extraction.doi,
        "screening": extraction.screening.model_dump(),
        "objective": _truncate(extraction.objective, 500),
        "study_design": extraction.study_design,
        "population_or_sample": _truncate(extraction.population_or_sample, 400),
        "setting": _truncate(extraction.setting, 300),
        "methods_summary": _truncate(extraction.methods_summary, 700),
        "outcomes_measured": extraction.outcomes_measured[:8],
        "results_summary": _truncate(extraction.results_summary, 900),
        "main_results": [_truncate(item, 350) for item in extraction.main_results[:8]],
        "conclusion_summary": _truncate(extraction.conclusion_summary, 650),
        "limitations": [_truncate(item, 250) for item in extraction.limitations[:5]],
        "relevance_to_review": _truncate(extraction.relevance_to_review, 600),
        "key_takeaways": [_truncate(item, 250) for item in extraction.key_takeaways[:6]],
        "extraction_confidence": extraction.extraction_confidence,
    }


EXTRACTION_SYSTEM_PROMPT = """You are a careful systematic-review extraction agent.
Screen the paper against the criteria, then extract structured evidence.
Do not invent missing facts. Use null or empty lists when information is unavailable.
For relevance, use 0 = unrelated, 1 = weak background relevance, 2 = tangential methods relevance, 3 = eligible but limited, 4 = clearly eligible, 5 = central paper.
Use decision values exactly: include, maybe, exclude.
Preserve units, sample sizes, effect directions, p-values, confidence intervals, and measurement windows when reported.
Write a detailed_summary of 6-10 sentences when enough abstract/full metadata exists.
If only abstract metadata is available, clearly say the extraction is abstract-limited.
Use methods_summary, results_summary, conclusion_summary, relevance_to_review, and key_takeaways to make the paper summary useful to a human reviewer."""


MANUSCRIPT_SYSTEM_PROMPT = """You are drafting a systematic review manuscript from extracted evidence.
Write cautiously and distinguish confirmed findings from preliminary synthesis.
Follow PRISMA 2020 concepts: eligibility criteria, information sources, search strategy, selection process, data extraction, synthesis methods, study characteristics, individual study results, synthesis, limitations, and transparent flow counts.
Do not overclaim. If evidence is incomplete, say what remains for human verification."""


STUDY_EXTRACTION_TEMPLATE = {
    "paper_id": "string",
    "title": "string",
    "citation": "string or null",
    "doi": "string or null",
    "url": "string or null",
    "pdf_url": "string or null",
    "year": "number or null",
    "authors": ["string"],
    "journal": "string or null",
    "screening": {
        "relevance_score": "integer 0-5",
        "decision": "include, maybe, or exclude",
        "reason": "string",
        "matched_inclusion_criteria": ["string"],
        "matched_exclusion_criteria": ["string"],
    },
    "overall_concept_summary": "string",
    "detailed_summary": "6-10 sentence detailed summary or null",
    "background_rationale": "string or null",
    "methods_summary": "string or null",
    "results_summary": "string or null",
    "conclusion_summary": "string or null",
    "relevance_to_review": "string or null",
    "key_takeaways": ["string"],
    "objective": "string or null",
    "study_design": "string or null",
    "population_or_sample": "string or null",
    "setting": "string or null",
    "intervention_or_exposure": "string or null",
    "comparator": "string or null",
    "methods": ["string"],
    "outcomes_measured": ["string"],
    "main_results": ["string"],
    "conclusions": ["string"],
    "limitations": ["string"],
    "risk_of_bias_notes": ["string"],
    "extracted_numeric_results": [],
    "evidence_quotes_or_locations": ["string"],
    "extraction_confidence": "high, medium, or low",
    "missing_information": ["string"],
}


MANUSCRIPT_TEMPLATE = {
    "title": "string",
    "abstract": "string",
    "introduction": "string",
    "methods": "string",
    "results": "string",
    "discussion": "string",
    "limitations": "string",
    "conclusions": "string",
    "prisma_flow_summary": "string",
    "tables_markdown": "string",
    "references": ["string"],
    "notes_for_human_reviewer": ["string"],
}
