from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import numpy as np

from .config import ReviewConfig
from .schemas import PaperCandidate


@dataclass
class RankedCandidate:
    candidate: PaperCandidate
    score: float


class CandidateRanker:
    def __init__(self, config: ReviewConfig):
        self.config = config

    def rank(self, candidates: list[PaperCandidate]) -> list[RankedCandidate]:
        if not candidates:
            return []
        if not self.config.use_pubmedbert_ranking:
            return [RankedCandidate(candidate, 0.0) for candidate in candidates]
        try:
            return self._rank_with_pubmedbert(candidates)
        except Exception:
            return self._rank_with_lexical_fallback(candidates)

    def _rank_with_pubmedbert(self, candidates: list[PaperCandidate]) -> list[RankedCandidate]:
        import torch
        from transformers import AutoModel, AutoTokenizer, logging

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        logging.set_verbosity_error()

        tokenizer = AutoTokenizer.from_pretrained(self.config.pubmedbert_model)
        model = AutoModel.from_pretrained(self.config.pubmedbert_model)
        model.eval()

        query = " ".join(
            [
                self.config.topic,
                self.config.review_question,
                " ".join(self.config.inclusion_criteria),
            ]
        )
        texts = [query] + [_candidate_text(candidate) for candidate in candidates]
        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for text in texts:
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                )
                output = model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vector = pooled[0].cpu().numpy()
                vectors.append(vector / max(np.linalg.norm(vector), 1e-12))
        query_vector = vectors[0]
        query_terms = _terms(query)
        ranked = []
        for candidate, vector in zip(candidates, vectors[1:]):
            semantic_score = float(np.dot(query_vector, vector))
            lexical_score = _lexical_score(query_terms, _terms(_candidate_text(candidate)))
            ranked.append(RankedCandidate(candidate, 0.7 * semantic_score + 0.3 * lexical_score))
        return sorted(ranked, key=lambda item: item.score, reverse=True)

    def _rank_with_lexical_fallback(self, candidates: list[PaperCandidate]) -> list[RankedCandidate]:
        query_terms = _terms(
            " ".join([self.config.topic, self.config.review_question, *self.config.inclusion_criteria])
        )
        ranked = []
        for candidate in candidates:
            text_terms = _terms(_candidate_text(candidate))
            ranked.append(RankedCandidate(candidate, _lexical_score(query_terms, text_terms)))
        return sorted(ranked, key=lambda item: item.score, reverse=True)


def _candidate_text(candidate: PaperCandidate) -> str:
    return " ".join(
        part
        for part in [
            candidate.title,
            candidate.abstract or "",
            candidate.journal or "",
            " ".join(candidate.keywords),
        ]
        if part
    )


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())}


def _lexical_score(query_terms: set[str], text_terms: set[str]) -> float:
    overlap = len(query_terms & text_terms)
    denom = math.sqrt(max(len(query_terms), 1) * max(len(text_terms), 1))
    return overlap / denom
