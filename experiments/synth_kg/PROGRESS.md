# Synthetic KG Benchmark Progress Log

| Timestamp (ISO) | Step / Run | Output File / Details |
| :--- | :--- | :--- |
| 2026-11-16T02:58:30+05:30 | Step 1: Inspected company.yaml ontology and verified 5 entity & 5 relationship types | `configs/domains/company.yaml` |
| 2026-11-16T03:00:55+05:30 | Step 2: Implemented generator, validator, and test suite (5 passed) | `experiments/synth_kg/test_generate.py` |
| 2026-11-16T03:03:46+05:30 | Step 3: Calibrated difficulty across 3 attempts; froze Attempt 1 as corpus_final | `experiments/synth_kg/corpus_final/` |
| 2026-11-16T03:07:34+05:30 | Step 4: Wrote pre-registration protocol PROTOCOL.md | `experiments/synth_kg/PROTOCOL.md` |




| 2026-11-16T03:51:01.251287 | Stage 1: Ingested corpus_final and built knowledge base | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synth_kg/kb/synth_company_kb` |
| 2026-11-16T04:53:28.361345 | Stage 2: Completed query evaluation for mode vector_only | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synth_kg/runs/20261116_035101_vector_only/results.jsonl` |
| 2026-11-16T06:20:38.534409 | Stage 3: Completed query evaluation for mode hybrid | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synth_kg/runs/20261116_045328_hybrid/results.jsonl` |
| 2026-11-16T07:44:05.840802 | Stage 4: Completed query evaluation for mode graph_only | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synth_kg/runs/20261116_062038_graph_only/results.jsonl` |
| 2026-11-16T09:15:39.156944 | Stage 5: Completed query evaluation for mode community_aware | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synth_kg/runs/20261116_074405_community_aware/results.jsonl` |
| 2026-11-16T09:16:16.859048 | Stage 6: Completed offline scoring and hypothesis tests | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synth_kg/evaluation_results.json` |
| 2026-11-16T09:16:37.443201 | Stage 6: Completed offline scoring and hypothesis tests | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synth_kg/evaluation_results.json` |
| 2026-11-16T09:16:37.449795 | Stage 7: Generated benchmark report REPORT.md | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synth_kg/REPORT.md` |
