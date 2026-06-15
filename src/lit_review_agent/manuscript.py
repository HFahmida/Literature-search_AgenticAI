from __future__ import annotations

from .schemas import ManuscriptDraft


def manuscript_to_markdown(draft: ManuscriptDraft) -> str:
    references = "\n".join(f"{i + 1}. {ref}" for i, ref in enumerate(draft.references))
    notes = "\n".join(f"- {note}" for note in draft.notes_for_human_reviewer)
    return f"""# {draft.title}

## Abstract

{draft.abstract}

## Introduction

{draft.introduction}

## Methods

{draft.methods}

## Results

{draft.results}

### Study Summary Table

{draft.tables_markdown}

### PRISMA Flow Summary

{draft.prisma_flow_summary}

## Discussion

{draft.discussion}

## Limitations

{draft.limitations}

## Conclusions

{draft.conclusions}

## References

{references}

## Notes For Human Reviewer

{notes}
"""
