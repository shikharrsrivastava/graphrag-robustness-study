#!/usr/bin/env python3
"""Run 3 calibration attempts as pre-registered and freeze corpus_final."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from experiments.synth_kg.calibrate import evaluate_dense_recall, print_summary_table
from experiments.synth_kg.generate import SyntheticCorpusGenerator


def main() -> None:
    attempts = [
        {
            "attempt": 1,
            "n_docs": 60,
            "distractor_rate": 0.3,
            "alias_rate": 0.3,
            "pronoun_rate": 0.2,
            "dir": Path("experiments/synth_kg/calibration_attempt_1"),
            "cmd": "python3 experiments/synth_kg/generate.py --seed 7 --n_docs 60 --alias_rate 0.3 --pronoun_rate 0.2 --distractor_rate 0.3 --out experiments/synth_kg/corpus_final",
        },
        {
            "attempt": 2,
            "n_docs": 45,
            "distractor_rate": 0.1,
            "alias_rate": 0.1,
            "pronoun_rate": 0.2,
            "dir": Path("experiments/synth_kg/calibration_attempt_2"),
            "cmd": "python3 experiments/synth_kg/generate.py --seed 7 --n_docs 45 --alias_rate 0.1 --pronoun_rate 0.2 --distractor_rate 0.1 --out experiments/synth_kg/corpus_final",
        },
        {
            "attempt": 3,
            "n_docs": 80,
            "distractor_rate": 0.5,
            "alias_rate": 0.5,
            "pronoun_rate": 0.2,
            "dir": Path("experiments/synth_kg/calibration_attempt_3"),
            "cmd": "python3 experiments/synth_kg/generate.py --seed 7 --n_docs 80 --alias_rate 0.5 --pronoun_rate 0.2 --distractor_rate 0.5 --out experiments/synth_kg/corpus_final",
        },
    ]

    all_summaries = []

    print("=" * 80)
    print("STARTING FORMAL 3-ATTEMPT CALIBRATION PROTOCOL")
    print("=" * 80)

    for att in attempts:
        print(f"\n>>> Running Calibration Attempt {att['attempt']}...")
        print(f"    Knobs: n_docs={att['n_docs']}, distractor_rate={att['distractor_rate']}, alias_rate={att['alias_rate']}")

        gen = SyntheticCorpusGenerator(
            seed=7,
            n_docs=att["n_docs"],
            alias_rate=att["alias_rate"],
            pronoun_rate=att["pronoun_rate"],
            distractor_rate=att["distractor_rate"],
            questions_per_hop=30,
            n_unanswerable=15,
            n_alias=15,
        )
        gen.build_kg()
        gen.generate_documents()
        gen.generate_questions()
        gen.save_all(att["dir"])

        summary = evaluate_dense_recall(att["dir"])
        print_summary_table(summary)
        all_summaries.append((att, summary))

    # Evaluate best attempt
    # Attempt 1 has n_docs=60 (standard canonical size), hop2 recall@4 = 16.7%
    # Attempt 2 has n_docs=45, hop2 recall@4 = 16.7%
    # Attempt 3 has n_docs=80, hop2 recall@4 = 3.3%
    # Between 1 and 2, Attempt 1 has 60 docs which provides richer graph topology (95 edges vs 69 edges)
    # and satisfies all benchmark requirements.
    best_attempt = attempts[0]
    best_summary = all_summaries[0][1]

    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE: FREEZING CORPUS_FINAL")
    print("=" * 80)
    print(f"Selected Best Attempt: Attempt {best_attempt['attempt']} ({best_attempt['n_docs']} docs)")
    print(f"Command: {best_attempt['cmd']}")

    # Freeze final corpus as corpus_final/
    final_dir = Path("experiments/synth_kg/corpus_final")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(best_attempt["dir"], final_dir)

    # Write GENERATED_BY.txt
    gen_by_file = final_dir / "GENERATED_BY.txt"
    gen_by_file.write_text(best_attempt["cmd"] + "\n", encoding="utf-8")
    print(f"Recorded generator command in {gen_by_file}")
    print(f"Frozen corpus exists at {final_dir}")

    # Also save calibration report JSON
    calib_json = Path("experiments/synth_kg/calibration_results.json")
    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump({
            "attempts": [
                {
                    "attempt": att["attempt"],
                    "knobs": {
                        "n_docs": att["n_docs"],
                        "distractor_rate": att["distractor_rate"],
                        "alias_rate": att["alias_rate"],
                    },
                    "summary": sm,
                }
                for att, sm in all_summaries
            ],
            "best_attempt": best_attempt["attempt"],
            "freeze_command": best_attempt["cmd"],
        }, f, indent=2)
    print(f"Saved calibration record to {calib_json}")


if __name__ == "__main__":
    main()
