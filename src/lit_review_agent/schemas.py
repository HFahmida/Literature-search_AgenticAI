from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class PaperCandidate(BaseModel):
    source: str
    source_id: str | None = None
    title: str
    abstract: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    doi: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    keywords: list[str] = Field(default_factory=list)
    citation_count: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def stable_id(self) -> str:
        if self.doi:
            return "doi:" + self.doi.strip().lower()
        normalized = " ".join(self.title.lower().split())
        return "title:" + normalized


class ScreeningDecision(BaseModel):
    relevance_score: int = Field(ge=0, le=5)
    decision: str = Field(description="include, maybe, or exclude")
    reason: str
    matched_inclusion_criteria: list[str] = Field(default_factory=list)
    matched_exclusion_criteria: list[str] = Field(default_factory=list)


class StudyExtraction(BaseModel):
    paper_id: str
    source: str | None = None
    source_id: str | None = None
    title: str
    abstract: str | None = None
    citation: str | None = None
    doi: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    screening: ScreeningDecision
    overall_concept_summary: str
    detailed_summary: str | None = None
    background_rationale: str | None = None
    methods_summary: str | None = None
    results_summary: str | None = None
    conclusion_summary: str | None = None
    relevance_to_review: str | None = None
    key_takeaways: list[str] = Field(default_factory=list)
    objective: str | None = None
    study_design: str | None = None
    population_or_sample: str | None = None
    setting: str | None = None
    intervention_or_exposure: str | None = None
    comparator: str | None = None
    methods: list[str] = Field(default_factory=list)
    outcomes_measured: list[str] = Field(default_factory=list)
    main_results: list[str] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    risk_of_bias_notes: list[str] = Field(default_factory=list)
    extracted_numeric_results: list[dict[str, Any]] = Field(default_factory=list)
    evidence_quotes_or_locations: list[str] = Field(default_factory=list)
    extraction_confidence: str = Field(description="high, medium, or low")
    missing_information: list[str] = Field(default_factory=list)


class RunStats(BaseModel):
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    records_identified: dict[str, int] = Field(default_factory=dict)
    duplicate_records_removed: int = 0
    unique_records_screened: int = 0
    reports_retrieved_or_abstract_only: int = 0
    reports_excluded: int = 0
    studies_included: int = 0
    extraction_errors: int = 0


class ManuscriptDraft(BaseModel):
    title: str
    abstract: str
    introduction: str
    methods: str
    results: str
    discussion: str
    limitations: str
    conclusions: str
    prisma_flow_summary: str
    tables_markdown: str
    references: list[str] = Field(default_factory=list)
    notes_for_human_reviewer: list[str] = Field(default_factory=list)
