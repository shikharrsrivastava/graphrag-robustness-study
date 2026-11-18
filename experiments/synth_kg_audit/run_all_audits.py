#!/usr/bin/env python3
"""Master script to run all audit components and generate a clean audit_output.txt."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

AUDIT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = AUDIT_DIR / "audit_output.txt"

SCRIPTS = [
    "audit_part1_a.py",
    "independent_scorer.py",
    "audit_part1_c.py",
    "audit_part1_d.py",
    "audit_part1_e.py",
    "audit_part1_f.py",
    "audit_part1_g.py",
    "audit_part1_h.py",
    "audit_part1_i.py",
    "audit_part1_j.py",
    "audit_part1_k.py",
    "audit_part2_baselines.py",
]


def main() -> None:
    # Clear output file with clean header
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("SYNTHETIC KG BENCHMARK AUDIT OUTPUT (CANONICAL COMPLETE RUN)\n")
        f.write("Workspace: /Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system\n")
        f.write("HEAD: 3877a4d02be92ca499fbb001e3a7afad971d4a4a\n")
        f.write("================================================================================\n\n")

    for s in SCRIPTS:
        script_path = AUDIT_DIR / s
        print(f"Running {s}...")
        res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error running {s}:\n{res.stderr}")
            sys.exit(res.returncode)
        print(f"Finished {s}")

    print(f"All audits completed successfully. Clean output saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
