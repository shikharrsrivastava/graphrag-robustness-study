# Synthetic Academic Experiment Progress Log

| Timestamp (ISO) | Step / Run | Output File / Details |
| :--- | :--- | :--- |
| 2026-11-15T03:19:39+05:30 | Safety Backup | `~/graphrag_backup_20261115` (`tracked_changes.patch`, `test_milestone2_fixes.py`, `synthetic_academic`) |
| 2026-11-15T03:20:36+05:30 | Step 1: Created analyze_runs.py and verified on KB_01 Run A and KB_02 Run A | `experiments/synthetic_academic/runs/analysis_20261115_032036.txt` |
| 2026-11-15T03:25:57+05:30 | Step 3: Added graph_context_mode 'none' and test_fix_e_graph_context_mode_none_leaks_no_graph_text | `tests/test_milestone2_fixes.py` (184 passed, 1 skipped) |
| 2026-11-15T03:35:00+05:30 | Step 4: KB_02 Run B (hybrid, graph_context_mode off, limits 2/3) | `experiments/synthetic_academic/runs/20261115_032639_kb02_run_b_constrained_off/results.jsonl` |
| 2026-11-15T03:45:00+05:30 | Step 4: KB_02 Run C (vector_only, graph_context_mode off, limits 2/3) | `experiments/synthetic_academic/runs/20261115_033552_kb02_run_c_vector_only/results.jsonl` |
| 2026-11-15T03:56:00+05:30 | Step 4: KB_02 Run D (graph_only, graph_context_mode off, limits 2/3) | `experiments/synthetic_academic/runs/20261115_034513_kb02_run_d_graph_only/results.jsonl` |
| 2026-11-15T04:09:00+05:30 | Step 4: KB_02 Run B_E (hybrid, graph_context_mode fix_e, limits 2/3) | `experiments/synthetic_academic/runs/20261115_035649_kb02_run_b_e_fix_e/results.jsonl` |
| 2026-11-15T04:21:00+05:30 | Step 4: KB_02 Run B_N (hybrid, graph_context_mode none, limits 2/3) | `experiments/synthetic_academic/runs/20261115_040954_kb02_run_b_n_mode_none/results.jsonl` |
| 2026-11-15T04:33:00+05:30 | Step 4: KB_01 Run B (hybrid, graph_context_mode off, limits 2/3) | `experiments/synthetic_academic/runs/20261115_042132_kb01_run_b_constrained_off/results.jsonl` |
| 2026-11-15T04:41:00+05:30 | Step 4: KB_01 Run C (vector_only, graph_context_mode off, limits 2/3) | `experiments/synthetic_academic/runs/20261115_043249_kb01_run_c_vector_only/results.jsonl` |
| 2026-11-15T10:56:37+05:30 | Safety Backup (Session 3) | `~/graphrag_backup_20261115_b` (`tracked_changes.patch`, `test_milestone2_fixes.py`, `synthetic_academic`) |
| 2026-11-15T11:14:30+05:30 | Part 2 & 3: Implemented Items 1-4 & added 7 new regression tests | `tests/test_milestone2_fixes.py` (191 passed, 1 skipped) |
| 2026-11-15T11:24:24+05:30 | Part 4: Built KB synthetic_academic_03 (15 entities, 8 relationships, 8 community reports) | `knowledge_bases/synthetic_academic/synthetic_academic_03/` |
| 2026-11-15T13:06:00.151469 | Part 4(a): KB_02, hybrid, comparative_reasoning_mode=off | `experiments/synthetic_academic/runs/20261115_125807_kb02_run_a_hybrid_compoff/results.jsonl` (14/16 correct, 0.949 recall) |
| 2026-11-15T13:12:26.601980 | Part 4(b): KB_02, vector_only, comparative_reasoning_mode=off | `experiments/synthetic_academic/runs/20261115_130600_kb02_run_b_vector_compoff/results.jsonl` (14/16 correct, 0.833 recall) |
| 2026-11-15T13:21:08.277048 | Part 4(c): KB_02, graph_only, comparative_reasoning_mode=off | `experiments/synthetic_academic/runs/20261115_131226_kb02_run_c_graph_compoff/results.jsonl` (9/16 correct, 0.718 recall) |
| 2026-11-15T13:31:29.654601 | Part 4(d): KB_03, hybrid, comparative_reasoning_mode=off | `experiments/synthetic_academic/runs/20261115_132108_kb03_run_d_hybrid_compoff/results.jsonl` (12/16 correct, 0.949 recall) |
| 2026-11-15T13:40:14.978916 | Part 4(e): KB_03, graph_only, comparative_reasoning_mode=off | `experiments/synthetic_academic/runs/20261115_133129_kb03_run_e_graph_compoff/results.jsonl` (11/16 correct, 0.769 recall) |
| 2026-11-15T13:51:37.300824 | Part 4(f): KB_03, community_aware, comparative_reasoning_mode=off | `experiments/synthetic_academic/runs/20261115_134015_kb03_run_f_comm_compoff/results.jsonl` (12/16 correct, 0.949 recall) |
| 2026-11-15T16:50:04.538781 | Part 4(g): KB_02, hybrid, comparative_reasoning_mode=enforce | `experiments/synthetic_academic/runs/20261115_135137_kb02_run_g_hybrid_compenforce/results.jsonl` (12/16 correct, 0.949 recall) |
| 2026-11-15T16:58:28.033016 | Part 4(h): KB_02, vector_only, comparative_reasoning_mode=enforce | `experiments/synthetic_academic/runs/20261115_165004_kb02_run_h_vector_compenforce/results.jsonl` (11/16 correct, 0.833 recall) |
| 2026-11-15T17:09:30+05:30 | Safety Backup (Session 4) | `~/graphrag_backup_20261115_c` (`tracked_changes.patch`, `test_milestone2_fixes.py`, `prompts/`, `synthetic_academic`) |
| 2026-11-15T17:59:06.613900 | Part 4 Run 1: KB_03, hybrid, compoff | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synthetic_academic/runs/20261115_175148_kb03_final_run1_hybrid/results.jsonl` (13/16 correct, 0.949 recall) |
| 2026-11-15T18:05:40.609906 | Part 4 Run 2: KB_03, vector_only, compoff | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synthetic_academic/runs/20261115_175951_kb03_final_run2_vector/results.jsonl` (15/16 correct, 0.833 recall) |
| 2026-11-15T18:14:55.389415 | Part 4 Run 3: KB_03, graph_only, compoff | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synthetic_academic/runs/20261115_180541_kb03_final_run3_graph/results.jsonl` (12/16 correct, 0.769 recall) |
| 2026-11-15T18:26:52.653895 | Part 4 Run 4: KB_03, community_aware, compoff | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synthetic_academic/runs/20261115_181456_kb03_final_run4_comm/results.jsonl` (13/16 correct, 0.949 recall) |
| 2026-11-15T18:37:30.425018 | Part 4 Run 5: KB_02, hybrid, compoff | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synthetic_academic/runs/20261115_182653_kb02_final_run5_hybrid/results.jsonl` (14/16 correct, 0.949 recall) |
| 2026-11-15T18:44:41.377904 | Part 4 Run 6: KB_02, vector_only, compoff | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synthetic_academic/runs/20261115_183731_kb02_final_run6_vector/results.jsonl` (15/16 correct, 0.833 recall) |
| 2026-11-15T19:00:15.363151 | Part 4 Run 7: KB_03, hybrid (repeat determinism check), compoff | `/Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/synthetic_academic/runs/20261115_184442_kb03_final_run7_hybrid_repeat/results.jsonl` (13/16 correct, 0.949 recall) |
