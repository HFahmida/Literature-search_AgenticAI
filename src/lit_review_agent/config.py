from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


DatabaseName = Literal["semantic_scholar", "crossref", "pubmed"]
ProviderName = Literal["ollama"]


class ReviewConfig(BaseModel):
    topic: str
    review_question: str
    search_terms: list[str]
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    databases: list[DatabaseName] = Field(
        default_factory=lambda: ["semantic_scholar", "crossref", "pubmed"]
    )
    max_results_per_source: int = 25
    max_papers_to_extract: int = 30
    min_relevance_score: int = 3
    extract_full_text_when_pdf_available: bool = True
    output_dir: str = "runs"
    llm_provider: ProviderName = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    extract_model: str = "qwen2.5:7b"
    draft_model: str = "qwen2.5:7b"
    use_pubmedbert_ranking: bool = True
    pubmedbert_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    max_ranked_candidates: int = 30
    extraction_detail_instructions: list[str] = Field(default_factory=list)
    custom_extraction_questions: list[str] = Field(default_factory=list)
    draft_during_extraction: bool = False
    draft_manuscript: bool = True
    draft_interval_seconds: int = 120
    concurrency: int = 1

    @classmethod
    def load(cls, path: str | Path) -> "ReviewConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
