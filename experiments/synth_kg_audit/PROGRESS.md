# Synthetic KG Audit Progress Log

| Timestamp (ISO) | Step | Output File / Details |
| :--- | :--- | :--- |
| 2026-11-16T09:46:30+05:30 | Step 0: Initialized experiments/synth_kg_audit/PROGRESS.md and verified git clean state | `experiments/synth_kg_audit/PROGRESS.md` |
| 2026-11-16T04:17:09Z | Wrote experiments/synth_kg_audit/PROTOCOL.md | experiments/synth_kg_audit/PROTOCOL.md |
| 2026-11-16T04:19:14Z | Completed Part 1(a) decisions, progress, calibration attempts, and dense recall audit | experiments/synth_kg_audit/audit_part1_a.py |
| 2026-11-16T04:20:02Z | Completed Part 1(b) scoring changes reconstruction and independent minimal scoring audit | experiments/synth_kg_audit/independent_scorer.py |
| 2026-11-16T04:21:56Z | Completed Part 1(c) question templates and relation token audit | experiments/synth_kg_audit/audit_part1_c.py |
| 2026-11-16T04:22:25Z | Completed Part 1(d) 15 sign tests and Holm-Bonferroni correction audit | experiments/synth_kg_audit/audit_part1_d.py |
| 2026-11-16T04:22:50Z | Completed Part 1(e) generation vs retrieval conditional accuracy audit | experiments/synth_kg_audit/audit_part1_e.py |
| 2026-11-16T04:23:37Z | Completed Part 1(f) unanswerable questions audit | experiments/synth_kg_audit/audit_part1_f.py |
| 2026-11-16T04:25:00Z | Completed Part 1(g) alias question audit | experiments/synth_kg_audit/audit_part1_g.py |
| 2026-11-16T04:26:41Z | Completed Part 1(h) hybrid mechanism on hop2 audit | experiments/synth_kg_audit/audit_part1_h.py |
| 2026-11-16T04:27:52Z | Completed Part 1(i) resolution drops audit | experiments/synth_kg_audit/audit_part1_i.py |
| 2026-11-16T04:28:35Z | Completed Part 1(j) extra canonical entities and split gold entities audit | experiments/synth_kg_audit/audit_part1_j.py |
| 2026-11-16T04:29:12Z | Completed Part 1(k) 3-hop validated path audit | experiments/synth_kg_audit/audit_part1_k.py |
| 2026-11-16T04:30:13Z | Completed Part 2 graph-free retrieval baselines evaluation | experiments/synth_kg_audit/audit_part2_baselines.py |
| 2026-11-16T04:33:01Z | Ran master audit runner and verified 1404 lines in audit_output.txt | experiments/synth_kg_audit/audit_output.txt |
