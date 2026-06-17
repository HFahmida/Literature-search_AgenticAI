# CLEF TAR Evaluation Report

This document explains, step by step, how the literature-review agent was evaluated using open-access benchmark data, what files were used, how the statistics were computed, and how to interpret the results.

## 1. Evaluation Goal

The goal was to test whether the local agentic literature-review pipeline can:

1. Search for relevant biomedical literature.
2. Rank retrieved candidate papers.
3. Screen papers using the local Qwen/Ollama model.
4. Compare pipeline outputs against a public gold-standard benchmark.

This evaluation focused on one benchmark topic as a pipeline behavior test. It should not be interpreted as a full benchmark of the system.

## 2. Benchmark Dataset

The evaluation used the open-access **CLEF TAR 2019 Task 2** dataset.

CLEF TAR stands for:

```text
Conference and Labs of the Evaluation Forum: Technology Assisted Reviews
```

The dataset provides systematic-review topics and gold-standard relevance judgments. These gold labels are called **qrels**.

For each topic, CLEF TAR provides:

- a systematic-review topic/title
- a PubMed-style search query
- a pool of PubMed IDs
- qrels indicating whether each PubMed record is relevant or non-relevant

## 3. Topic Used

The evaluated topic was:

```text
CD000996: Inhaled corticosteroids for bronchiectasis
```

For this topic, the CLEF gold standard contained:

```text
6 gold relevant papers
```

The central evaluation question was:

```text
Can the pipeline retrieve, rank, and screen the known relevant PubMed papers for this review topic?
```

## 4. Data Location

The CLEF TAR data was downloaded to:

```text
D:\Literature_Search\tar
```

The qrels used for this evaluation were located at:

```text
D:\Literature_Search\tar\2019-TAR\Task2\Testing\Intervention\qrels
```

The evaluated pipeline run was:

```text
D:\Literature_Search\runs\20260616-192955-cd000996-inhaled-corticosteroids-for-bronchiectasis
```

## 5. Pipeline Outputs Used

The evaluator used these pipeline outputs:

```text
ranked_candidates.json
extractions/*.json
review_config.json
run_stats.json
```

The most important files were:

```text
ranked_candidates.json
```

This file contains all retrieved candidates ranked by PubMedBERT and lexical similarity.

```text
extractions/*.json
```

These files contain Qwen screening decisions, relevance scores, source IDs, abstracts, and extracted paper-level information.

## 6. Evaluation Command

The evaluation was run with:

```powershell
cd D:\Literature_Search

.\.venv\Scripts\python.exe scripts\evaluation\evaluate_clef_tar.py `
  --run-dir "D:\Literature_Search\runs\20260616-192955-cd000996-inhaled-corticosteroids-for-bronchiectasis" `
  --qrels "D:\Literature_Search\tar\2019-TAR\Task2\Testing\Intervention\qrels"
```

The evaluator matched papers using PubMed IDs from the pipeline outputs and the CLEF qrels.

## 7. Pipeline Run Summary

The evaluated run produced:

```text
197 ranked candidates
33 extracted papers
11 extracted papers matched CLEF qrels
6 gold relevant papers in CLEF
3 gold relevant papers retrieved by the pipeline
```

## 8. Evaluation Levels

The evaluation separates the system into three stages:

1. Search coverage
2. Ranking performance
3. Screening performance

This separation is important because downstream ranking and screening can only operate on papers that were first retrieved by search.

## 9. Search Coverage

Search coverage asks:

```text
Did the pipeline retrieve the known relevant papers at all?
```

The statistic used was **search recall**.

Formula:

```text
search recall = gold relevant papers retrieved / total gold relevant papers
```

For this run:

```text
search recall = 3 / 6 = 0.500
```

Interpretation:

The search step retrieved half of the known relevant papers for this CLEF topic.

This means search coverage is currently the main bottleneck. If a relevant paper is not retrieved, PubMedBERT ranking and Qwen screening cannot recover it later.

## 10. Ranking Performance

Ranking performance asks:

```text
Among retrieved candidates, did PubMedBERT place relevant papers near the top?
```

Ranking was evaluated using `ranked_candidates.json`.

The main ranking results were:

```text
Average precision: 0.2394
Ranking WSS@95: 0.7266
Recall@10: 0.6667
Recall@20: 0.6667
Recall@50: 1.000
Recall@100: 1.000
nDCG@10: 0.3836
nDCG@20: 0.3836
nDCG@50: 0.4691
nDCG@100: 0.4691
```

Metric meanings:

- **Average precision** measures how early relevant papers appear across the ranked list.
- **Recall@10** means the fraction of retrieved relevant papers found in the top 10 ranked candidates.
- **Recall@50** means the fraction of retrieved relevant papers found in the top 50 ranked candidates.
- **nDCG** rewards relevant papers appearing earlier in the ranking.
- **WSS@95** estimates work saved while maintaining high recall.

Interpretation:

The ranking stage performed reasonably after retrieval. All retrieved relevant papers appeared by the top 50. However, ranking performance is still limited by search coverage because the search stage retrieved only 3 of the 6 gold relevant papers.

## 11. Screening Performance

Screening performance asks:

```text
Did Qwen correctly classify extracted papers as relevant or not relevant?
```

The pipeline uses three screening decisions:

```text
include
maybe
exclude
```

For high-recall systematic-review screening, these were converted to binary labels:

```text
include = positive
maybe = positive
exclude = negative
```

This is intentional because false negatives are especially costly in systematic reviews.

The screening results were:

```text
Precision: 0.25
Recall: 1.00
F1: 0.40
Cohen's kappa: 0.1538
Screening WSS@95: 0.5864
False negatives: 0
```

Interpretation:

Qwen preserved recall among the extracted papers that matched CLEF qrels. It did not miss any relevant matched paper.

However, precision was low:

```text
Precision = 0.25
```

This means Qwen marked extra non-relevant papers as positive. In systematic-review workflows, this is often acceptable for early screening because the priority is not missing relevant studies.

## 12. Main Result

The most important result was:

```text
The pipeline retrieved 3 of 6 gold relevant papers.
```

Therefore:

```text
Search coverage is the main bottleneck.
```

The downstream models performed better once relevant papers were present in the candidate pool:

- PubMedBERT ranked all retrieved relevant papers by the top 50.
- Qwen screening had recall of 1.00 among matched extracted papers.
- Qwen produced no false negatives in the matched extracted subset.

But the system cannot screen or extract relevant papers that were never retrieved.

## 13. Limitations

This was a single-topic evaluation.

Important limitations:

- Only one CLEF topic was evaluated.
- The result should not be reported as a general benchmark.
- Search recall depends heavily on how the CLEF query is translated into PubMed search terms.
- Qwen screening was evaluated only on extracted papers that matched CLEF qrels.
- Full paper content was not used for all papers; many extractions are abstract-limited.

## 14. Recommended Improvements

Recommended next steps:

1. Improve PubMed query translation from CLEF/PICO text.
2. Add query expansion terms.
3. Use multiple PubMed search strings per topic.
4. Increase `max_results_per_source`.
5. Evaluate more CLEF topics.
6. Compare with and without PubMedBERT ranking.
7. Add full-text retrieval for open-access papers.
8. Report aggregate metrics across multiple topics before making benchmark claims.

## 15. PowerPoint Report

A PowerPoint report was created at:

```text
D:\Literature_Search\presentation\clef_evaluation_full_report.pptx
```

The slide deck includes:

- dataset description
- topic information
- pipeline run summary
- evaluation command and input files
- search coverage results
- ranking metrics
- Qwen screening metrics
- interpretation and next steps

## 16. One-Slide Run 2 Summary

A shorter Run 2-only slide deck was also created:

```text
D:\Literature_Search\presentation\clef_evaluation_run2_only.pptx
```

This file reports only the main Run 2 evaluation results and excludes the earlier tiny sample run.

