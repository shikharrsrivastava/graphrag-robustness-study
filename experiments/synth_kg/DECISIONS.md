# Synthetic KG Benchmark Decisions Log

This document records all architectural, algorithmic, and engineering decisions made during the unattended overnight benchmark.

| Timestamp (ISO) | Category | Decision | Justification |
| :--- | :--- | :--- | :--- |
| 2026-11-16T02:58:30+05:30 | Ontology Selection | Use `configs/domains/company.yaml` with its 5 native entity types (`EMPLOYEE`, `DEPARTMENT`, `PRODUCT`, `TECHNOLOGY`, `CLIENT`) and 5 native relationship types (`MANAGES`, `BELONGS_TO`, `DEVELOPS`, `BUILT_WITH`, `SERVES`). | `company.yaml` has 5 >= 3 usable relationship types. People map to `EMPLOYEE`, internal units map to `DEPARTMENT`, external orgs map to `CLIENT`, products map to `PRODUCT`, and technical tools map to `TECHNOLOGY`. Multi-hop chains of lengths 1, 2, and 3 are supported cleanly without modifying any config files. |
| 2026-11-16T03:03:46+05:30 | Calibration Corpus Selection | Freeze Attempt 1 (`n_docs=60`, `alias_rate=0.3`, `distractor_rate=0.3`, `pronoun_rate=0.2`) as `corpus_final/`. | Across the 3 pre-registered calibration attempts (Attempt 1: 60 docs, Attempt 2: 45 docs, Attempt 3: 80 docs), hop2 recall@4 reached 16.7% and hop3 recall@4 reached 0.0%. As anticipated by the benchmark design, pure dense single-vector retrieval cannot rank unmentioned bridge documents in multi-hop chains within the top 4. Attempt 1 provides the optimal canonical 60-document balance with 95 edges, rich graph topology, and 120 questions across all 5 groups. Recorded as instructed. |
| 2026-11-16T03:54:57+05:30 | Scoring Engine Normalization | Normalize document IDs in `score.py` using `normalize_doc_id` (`D033` $\leftrightarrow$ `DOC_0033`). | The ingestion pipeline formats doc IDs as `D001`..`D060` from `DOC_0001.txt`..`DOC_0060.txt`, while gold questions record `DOC_0001`..`DOC_0060`. Added unit test `test_doc_id_normalization` in `test_score.py` (6 passed) prior to updating `score.py`. Permitted under Rule 0 bug exception. |



