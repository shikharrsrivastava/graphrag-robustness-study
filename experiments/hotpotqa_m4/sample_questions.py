#!/usr/bin/env python3
"""Part 2(a): Sample 200 questions for pooled dense retrieval evaluation.

Uses random.Random(43) to sample 100 bridge and 100 comparison questions from
hotpot_dev_distractor_v1.json, excluding the 20 repository exclusion questions
and the 30 Milestone 3 selected questions.
Saves to experiments/hotpotqa_m4/pooled_questions.json BEFORE computing embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys


def main() -> None:
    data_path = Path("experiments/hotpotqa_m3/data/hotpot_dev_distractor_v1.json")
    exclusion_path = Path("experiments/hotpotqa_m3/exclusion_set.json")
    m3_selected_path = Path("experiments/hotpotqa_m3/selected_questions.json")
    output_path = Path("experiments/hotpotqa_m4/pooled_questions.json")

    print(f"Loading dev dataset from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    with open(exclusion_path, "r", encoding="utf-8") as f:
        excl_ids = set(json.load(f))

    with open(m3_selected_path, "r", encoding="utf-8") as f:
        m3_sel = json.load(f)
        m3_ids = set(q["_id"] for q in m3_sel)

    total_excluded = excl_ids | m3_ids
    print(f"Total excluded questions: {len(total_excluded)} ({len(excl_ids)} from exclusion set, {len(m3_ids)} from M3)")

    # Filter candidate questions preserving original dataset order
    bridge_cands = [q for q in dev_data if q["_id"] not in total_excluded and q.get("type") == "bridge"]
    comp_cands = [q for q in dev_data if q["_id"] not in total_excluded and q.get("type") == "comparison"]

    print(f"Eligible candidates: {len(bridge_cands)} bridge, {len(comp_cands)} comparison (Total: {len(bridge_cands) + len(comp_cands)})")

    # Sample with fixed seed 43
    rng = random.Random(43)
    sampled_bridge = rng.sample(bridge_cands, 100)
    sampled_comp = rng.sample(comp_cands, 100)

    pooled_sample = sampled_bridge + sampled_comp

    print(f"Sampled: {len(sampled_bridge)} bridge, {len(sampled_comp)} comparison (Total: {len(pooled_sample)})")

    # Verify no overlaps with exclusion set or M3
    sample_ids = set(q["_id"] for q in pooled_sample)
    assert len(sample_ids) == 200, "Duplicate IDs in sample!"
    assert not (sample_ids & total_excluded), "Sample overlaps with excluded IDs!"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pooled_sample, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved 200 sampled questions to {output_path}")

    # Also save compact metadata summary (IDs and types)
    meta_summary = [{"_id": q["_id"], "type": q["type"], "question": q["question"]} for q in pooled_sample]
    summary_path = Path("experiments/hotpotqa_m4/pooled_questions_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(meta_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
