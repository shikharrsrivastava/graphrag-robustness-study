import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TIMINGS_CSV, TIMINGS_PNG

PHASES = ["load_s", "chunk_s", "embed_store_s", "total_s"]
COLORS = {"full": "#d95f02", "incremental": "#1b9e77"}


def run() -> None:
    if not TIMINGS_CSV.exists():
        print(f"No {TIMINGS_CSV.name} found. Run some ingest commands first.")
        return
    with open(TIMINGS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{TIMINGS_CSV.name} has no data rows.")
        return

    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    modes = sorted(by_mode)

    fig, (ax_phases, ax_runs) = plt.subplots(1, 2, figsize=(13, 5.5))

    width = 0.8 / len(modes)
    for i, mode in enumerate(modes):
        runs = by_mode[mode]
        values = [sum(float(r[phase]) for r in runs) / len(runs) for phase in PHASES]
        xs = [x + i * width for x in range(len(PHASES))]
        bars = ax_phases.bar(
            xs,
            values,
            width=width,
            label=f"{mode} (n={len(runs)})",
            color=COLORS.get(mode),
        )
        ax_phases.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax_phases.set_xticks([x + width * (len(modes) - 1) / 2 for x in range(len(PHASES))])
    ax_phases.set_xticklabels(PHASES)
    ax_phases.set_ylabel("seconds (mean)")
    ax_phases.set_title("Mean phase timings by ingestion mode")
    ax_phases.legend()

    for i, row in enumerate(rows):
        total = float(row["total_s"])
        color = COLORS.get(row["mode"], "#7570b3")
        bar = ax_runs.bar(i, total, color=color)
        ax_runs.bar_label(bar, fmt="%.2f", padding=2, fontsize=8)
        ax_runs.text(
            i,
            total / 2,
            f"+{row['embedded']}\n-{row['deleted']}",
            ha="center",
            va="center",
            fontsize=7,
        )
    ax_runs.set_xticks(range(len(rows)))
    ax_runs.set_xticklabels(
        [row["label"] for row in rows], rotation=45, ha="right", fontsize=8
    )
    ax_runs.set_ylabel("total seconds")
    ax_runs.set_title("Total ingestion time per run (+embedded / -deleted)")

    fig.suptitle("RAG ingestion benchmark: incremental vs full")
    fig.tight_layout()
    fig.savefig(TIMINGS_PNG, dpi=150)
    summary = ", ".join(f"{mode} x{len(by_mode[mode])}" for mode in modes)
    print(f"Wrote {TIMINGS_PNG} from {len(rows)} runs ({summary}).")


if __name__ == "__main__":
    run()
