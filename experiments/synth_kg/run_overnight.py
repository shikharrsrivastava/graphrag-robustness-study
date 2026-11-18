#!/usr/bin/env python3
"""Overnight benchmark driver for Synthetic KG Benchmark.

Executes Stages 1-7 in order:
  Stage 1: Build Knowledge Base through public API (HybridGraphRAG) into experiments/synth_kg/kb/
  Stage 2: Query run - vector_only
  Stage 3: Query run - hybrid
  Stage 4: Query run - graph_only
  Stage 5: Query run - community_aware
  Stage 6: Offline scoring via score.py
  Stage 7: Report generation via report.py producing REPORT.md

Resumable: Each stage writes a marker file (*.done) and is skipped on restart.
Query runs append one JSONL record per question as it finishes and skip question IDs already present.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional
import yaml

from src.core.configuration import deep_merge_dicts
from hybrid_graphrag import HybridGraphRAG, QueryResult


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
EXP_DIR = BASE_DIR / "experiments" / "synth_kg"
CORPUS_DIR = EXP_DIR / "corpus_final"
DOCS_DIR = CORPUS_DIR / "docs"
QUESTIONS_FILE = CORPUS_DIR / "questions.jsonl"
KB_ROOT = EXP_DIR / "kb"
KB_ID = "synth_company_kb"
KB_DIR = KB_ROOT / KB_ID
RUNS_DIR = EXP_DIR / "runs"
PROGRESS_FILE = EXP_DIR / "PROGRESS.md"
START_TIME_FILE = EXP_DIR / "driver_start_time.txt"

MAX_TIME_SECONDS = 9 * 3600  # 9 hours


def log(msg: str) -> None:
    """Print log message with ISO timestamp and flush."""
    ts = datetime.now().isoformat()
    print(f"[{ts}] {msg}", flush=True)


def append_progress(step_desc: str, output_path: str) -> None:
    """Append one line to PROGRESS.md."""
    ts = datetime.now().isoformat()
    line = f"| {ts} | {step_desc} | `{output_path}` |\n"
    try:
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        log(f"Warning: failed to append to PROGRESS.md: {e}")


def check_and_ensure_ollama(timeout: int = 180) -> bool:
    """Check Ollama service health; attempt restart if unreachable."""
    start_time = time.time()
    url = "http://localhost:11434/api/tags"

    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass

        log("Ollama unreachable at http://localhost:11434. Attempting recovery...")
        try:
            subprocess.run(["open", "-a", "Ollama"], check=False)
        except Exception:
            try:
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

        time.sleep(5)

    log(f"ERROR: Ollama remained unreachable after {timeout} seconds.")
    return False


def get_benchmark_config() -> Dict[str, Any]:
    """Construct merged engine configuration dictionary."""
    with open(BASE_DIR / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(BASE_DIR / "configs" / "domains" / "company.yaml", "r", encoding="utf-8") as f:
        domain_cfg = yaml.safe_load(f)

    cfg = deep_merge_dicts(base_cfg, domain_cfg)
    cfg["storage_root"] = str(KB_ROOT)
    cfg["kb_id"] = KB_ID
    cfg.setdefault("ollama", {})["temperature"] = 0.0
    cfg.setdefault("retrieval", {})["vector_top_k"] = 4
    cfg.setdefault("retrieval", {})["final_context_chunks"] = 4
    cfg.setdefault("retrieval", {})["comparative_reasoning_mode"] = "off"
    cfg.setdefault("retrieval", {})["graph_context_mode"] = False
    cfg.setdefault("prompt_versions", {})["answer_generation"] = "v1.1"

    return cfg


def run_stage_build() -> bool:
    """Stage 1: Build knowledge base through public API."""
    marker = EXP_DIR / "stage1_build.done"
    if marker.exists():
        log("Stage 1 (Build) marker exists. Skipping build.")
        return True

    log("=" * 80)
    log(f"STAGE 1: Building Knowledge Base '{KB_ID}' from '{DOCS_DIR}'")
    log("=" * 80)

    if not check_and_ensure_ollama():
        log("Ollama unavailable. Aborting build.")
        return False

    cfg = get_benchmark_config()

    def do_build() -> Dict[str, Any]:
        start = time.perf_counter()
        rag = HybridGraphRAG.from_config(config=cfg, base_dir=BASE_DIR)
        ingest_stats = rag.ingest(input_path=DOCS_DIR, rebuild=True)
        kb_stats = rag.get_kb_stats()
        wall_time = time.perf_counter() - start
        rag.close()
        return {
            "ingest_stats": ingest_stats,
            "kb_stats": kb_stats,
            "build_wall_seconds": wall_time,
        }

    # Attempt build with retry
    stats_data = None
    for attempt in (1, 2):
        try:
            log(f"Starting build attempt {attempt}...")
            stats_data = do_build()
            log(f"Build attempt {attempt} succeeded in {stats_data['build_wall_seconds']:.2f}s.")
            break
        except Exception as e:
            log(f"Build attempt {attempt} failed: {e}")
            if KB_DIR.exists():
                log(f"Cleaning partial KB directory: {KB_DIR}")
                shutil.rmtree(KB_DIR, ignore_errors=True)
            if attempt == 2:
                log("All build attempts failed. Aborting.")
                return False

    # Save build stats
    KB_DIR.mkdir(parents=True, exist_ok=True)
    stats_file = KB_DIR / "build_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    marker.write_text(f"completed at {datetime.now().isoformat()}\n")
    append_progress("Stage 1: Ingested corpus_final and built knowledge base", str(KB_DIR))
    return True


def run_query_worker(mode: str, run_dir_path: str) -> None:
    """Worker process: queries the KB in a fresh Python process."""
    run_dir = Path(run_dir_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    config_snapshot_path = run_dir / "config_snapshot.yaml"

    log(f"Worker process started for mode '{mode}' in '{run_dir}'")

    # Load questions
    questions: List[Dict[str, Any]] = []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    # Determine already completed question IDs (resumability)
    completed_qids = set()
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed_qids.add(rec.get("question_id") or rec.get("id"))
                    except Exception:
                        pass
    log(f"Mode '{mode}': {len(completed_qids)}/{len(questions)} questions already completed.")

    if len(completed_qids) >= len(questions):
        log(f"All questions for mode '{mode}' already completed.")
        return

    # Check Ollama health
    if not check_and_ensure_ollama():
        raise RuntimeError("Ollama service unavailable.")

    # Reopen KB via public API
    log(f"Opening KB via HybridGraphRAG.from_existing('{KB_DIR}')")
    rag = HybridGraphRAG.from_existing(kb_dir=KB_DIR, base_dir=BASE_DIR)

    # Save config snapshot into run directory if not exists
    if not config_snapshot_path.exists():
        kb_snap = KB_DIR / "config_snapshot.yaml"
        if kb_snap.exists():
            shutil.copyfile(kb_snap, config_snapshot_path)

    log(f"Querying {len(questions) - len(completed_qids)} remaining questions...")

    # Run remaining questions
    for idx, q_item in enumerate(questions, start=1):
        qid = q_item["id"]
        if qid in completed_qids:
            continue

        q_text = q_item["question"]
        q_group = q_item.get("group", "")

        # Check Ollama health every 20 queries
        if idx % 20 == 0:
            if not check_and_ensure_ollama():
                log(f"Warning: Ollama health check failed at question {idx} ({qid}).")

        q_start = time.perf_counter()
        query_result: Optional[QueryResult] = None
        is_gen_err = False
        ans_text = ""

        try:
            query_result = rag.query(question=q_text, mode=mode)
            ans_text = query_result.answer
            is_gen_err = bool(query_result.generation_error)
        except Exception as exc:
            log(f"Query failed on question {qid}: {exc}")
            ans_text = f"ERROR: {exc}"
            is_gen_err = True

        q_duration = time.perf_counter() - q_start

        # Extract context chunks and doc IDs
        context_chunks: List[Dict[str, str]] = []
        context_doc_ids: List[str] = []
        context_chunk_ids: List[str] = []
        cited_chunk_ids: List[str] = []

        if query_result and query_result.raw_result:
            for ev in query_result.raw_result.evidence_list:
                context_chunks.append({"chunk_id": ev.chunk_id, "doc_id": ev.doc_id})
                context_doc_ids.append(ev.doc_id)
                context_chunk_ids.append(ev.chunk_id)
            cited_chunk_ids = list(query_result.cited_chunk_ids)

        record = {
            "question_id": qid,
            "group": q_group,
            "mode": mode,
            "question": q_text,
            "answer": ans_text,
            "abstained": query_result.abstained if query_result else False,
            "abstention_reason": query_result.abstention_reason if query_result else None,
            "generation_error": is_gen_err,
            "cited_chunk_ids": cited_chunk_ids,
            "context_chunks": context_chunks,
            "context_chunk_ids": context_chunk_ids,
            "context_doc_ids": context_doc_ids,
            "latency_seconds": q_duration,
        }

        # Append immediately to disk
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

        completed_qids.add(qid)
        if idx % 10 == 0 or idx == len(questions):
            log(f"[{idx}/{len(questions)}] {mode} | {qid} | latency: {q_duration:.2f}s | abstained: {record['abstained']}")

    rag.close()
    log(f"Worker for mode '{mode}' completed all queries.")


def run_stage_query(stage_num: int, mode: str, driver_start: float) -> bool:
    """Execute query stage in a fresh process."""
    marker = EXP_DIR / f"stage{stage_num}_{mode}.done"
    if marker.exists():
        log(f"Stage {stage_num} ({mode}) marker exists. Skipping.")
        return True

    # Check time cap
    elapsed = time.time() - driver_start
    if elapsed >= MAX_TIME_SECONDS:
        log(f"Time cap reached ({elapsed/3600:.2f}h >= 9h). Skipping {mode}.")
        return False

    log("=" * 80)
    log(f"STAGE {stage_num}: Query Evaluation - Mode '{mode}'")
    log("=" * 80)

    # Locate or create run directory
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing_runs = sorted(RUNS_DIR.glob(f"*_{mode}"))
    if existing_runs:
        run_dir = existing_runs[-1]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / f"{timestamp}_{mode}"
        run_dir.mkdir(parents=True, exist_ok=True)

    # Spawn fresh process
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker_mode",
        mode,
        "--run_dir",
        str(run_dir),
    ]

    log(f"Spawning subprocess: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        log(f"ERROR: Query worker for mode '{mode}' failed with return code {res.returncode}")
        return False

    marker.write_text(f"completed at {datetime.now().isoformat()}\n")
    append_progress(f"Stage {stage_num}: Completed query evaluation for mode {mode}", str(run_dir / "results.jsonl"))
    return True


def run_stage_scoring() -> bool:
    """Stage 6: Offline scoring."""
    marker = EXP_DIR / "stage6_scoring.done"
    log("=" * 80)
    log("STAGE 6: Offline Scoring via score.py")
    log("=" * 80)

    from experiments.synth_kg.score import score_runs_against_disk

    eval_json_path = EXP_DIR / "evaluation_results.json"
    eval_results = score_runs_against_disk(
        corpus_final_dir=CORPUS_DIR,
        kb_dir=KB_DIR,
        runs_dir=RUNS_DIR,
    )

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    marker.write_text(f"completed at {datetime.now().isoformat()}\n")
    append_progress("Stage 6: Completed offline scoring and hypothesis tests", str(eval_json_path))
    log(f"Offline scoring completed. Results saved to {eval_json_path}")
    return True


def run_stage_report(incomplete: bool = False, reason: str = "") -> bool:
    """Stage 7: Generate REPORT.md."""
    marker = EXP_DIR / "stage7_report.done"
    log("=" * 80)
    log("STAGE 7: Report Generation via report.py")
    log("=" * 80)

    from experiments.synth_kg.report import generate_report_markdown

    eval_json_path = EXP_DIR / "evaluation_results.json"
    if not eval_json_path.exists():
        run_stage_scoring()

    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_results = json.load(f)

    build_stats_path = KB_DIR / "build_stats.json"
    build_stats = None
    if build_stats_path.exists():
        with open(build_stats_path, "r", encoding="utf-8") as f:
            build_stats = json.load(f)

    report_md = generate_report_markdown(
        eval_results=eval_results,
        build_stats=build_stats,
        incomplete=incomplete,
        incomplete_reason=reason,
    )

    report_file = EXP_DIR / "REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    marker.write_text(f"completed at {datetime.now().isoformat()}\n")
    append_progress("Stage 7: Generated benchmark report REPORT.md", str(report_file))
    log(f"Report generated successfully at {report_file}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic KG Benchmark Overnight Driver")
    parser.add_argument("--worker_mode", default=None, help="Internal worker mode (fresh process)")
    parser.add_argument("--run_dir", default=None, help="Run directory for worker")
    args = parser.parse_args()

    # If worker invocation
    if args.worker_mode:
        if not args.run_dir:
            raise ValueError("--run_dir must be specified when --worker_mode is used.")
        run_query_worker(args.worker_mode, args.run_dir)
        return

    # Master driver invocation
    log("=" * 80)
    log("SYNTHETIC KG BENCHMARK OVERNIGHT DRIVER LAUNCHED")
    log(f"Base Directory: {BASE_DIR}")
    log(f"Experiment Directory: {EXP_DIR}")
    log("=" * 80)

    # Initialize start time
    if START_TIME_FILE.exists():
        try:
            driver_start = float(START_TIME_FILE.read_text().strip())
        except Exception:
            driver_start = time.time()
            START_TIME_FILE.write_text(str(driver_start))
    else:
        driver_start = time.time()
        START_TIME_FILE.write_text(str(driver_start))

    # Stage 1: Build
    build_ok = run_stage_build()
    if not build_ok:
        log("Build stage failed. Moving directly to reporting incomplete.")
        run_stage_report(incomplete=True, reason="Knowledge Base build failed.")
        return

    # Stages 2-5: Query modes in priority order
    query_modes = [
        (2, "vector_only"),
        (3, "hybrid"),
        (4, "graph_only"),
        (5, "community_aware"),
    ]

    time_cap_hit = False
    for stage_num, mode in query_modes:
        elapsed = time.time() - driver_start
        if elapsed >= MAX_TIME_SECONDS:
            log(f"Time cap of 9 hours exceeded ({elapsed/3600:.2f}h).")
            time_cap_hit = True
            break
        success = run_stage_query(stage_num, mode, driver_start)
        if not success:
            log(f"Stage {stage_num} ({mode}) did not complete successfully.")

    # Stage 6: Scoring
    run_stage_scoring()

    # Stage 7: Reporting
    run_stage_report(incomplete=time_cap_hit, reason="9-hour wall-clock time cap reached" if time_cap_hit else "")

    log("=" * 80)
    log("ALL DRIVER STAGES COMPLETED.")
    log("=" * 80)


if __name__ == "__main__":
    main()
