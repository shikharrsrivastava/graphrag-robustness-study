# HotpotQA Milestone 4 Progress Log

| Timestamp (ISO) | Step / Run | Output File / Details |
| :--- | :--- | :--- |
| 2026-11-16T00:03:20+05:30 | Part 1: Adjudication dump from M3 run files | `experiments/hotpotqa_m4/adjudication_dump.txt` |
| 2026-11-16T00:04:18+05:30 | Part 2(a): Sampled 200 questions (seed 43, 100 bridge, 100 comparison) | `experiments/hotpotqa_m4/pooled_questions.json` |


| 2026-11-16T00:06:37.881994+05:30 | Part 2(b,c): Built corpus (1981 paras, 19 dups) and generated embeddings (65.2s) | `experiments/hotpotqa_m4/embeddings/corpus_embeddings.npy` |
| 2026-11-16T00:06:37.944042+05:30 | Part 2(d,e): Evaluated pooled retrieval metrics (Overall Recall@3: 74.0% vs Own-10: 78.0%) | `experiments/hotpotqa_m4/pooled_recall_summary.json` |
| 2026-11-16T02:01:13.218197+05:30 | Part 3: Ingested 10 paragraphs (635.2s, 63.5s/para, 78 entities, 11 val rels, 52 rej rels) | `experiments/hotpotqa_m4/ingestion_stats.json` |
