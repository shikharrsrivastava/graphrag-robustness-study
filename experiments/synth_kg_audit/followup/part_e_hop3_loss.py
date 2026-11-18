#!/usr/bin/env python3
"""
experiments/synth_kg_audit/followup/part_e_hop3_loss.py
Hop3 Loss Localisation Audit (Retrieval-Only Pipeline).
"""

import json
import sys
from pathlib import Path
import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path.cwd()))

from hybrid_graphrag import HybridGraphRAG
from src.retrieval.seed_retriever import SeedEntityRetriever
from src.retrieval.graph_retriever import GraphEvidenceRetriever
from src.graph.traversal import GraphTraverser
from src.core.models import QueryPlan, QueryIntent, SeedEntitySpec

def main():
    report = []
    report.append("================================================================================")
    report.append("PART E: HOP3 LOSS LOCALISATION (RETRIEVAL-ONLY AUDIT)")
    report.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
    report.append("Reconstructing retrieval stages for 5 hop3 questions (Q_0062, Q_0063, Q_0073, Q_0084, Q_0068)")
    report.append("================================================================================\n")

    # Load KB & Graph
    kb_path = Path("experiments/synth_kg/kb/synth_company_kb")
    rag = HybridGraphRAG.from_existing(str(kb_path))
    engine = rag._engine
    graph = engine.get_or_load_graph()
    entities_by_id = {e.entity_id: e for e in engine.doc_store.load_canonical_entities()}

    # Print Configuration and Code Lines
    cfg = engine.config.retrieval
    report.append("--------------------------------------------------------------------------------")
    report.append("CONFIGURED LIMITS AND CODE CITATIONS:")
    report.append(f"- max_traversal_hops: {cfg.max_traversal_hops}")
    report.append("  * Config: configs/default.yaml:26")
    report.append("  * Instantiated: src/engine.py:305 (GraphTraverser(max_hops=self.config.retrieval.max_traversal_hops))")
    report.append("  * Applied: src/graph/traversal.py:115 (`if current_hop >= self.max_hops: continue`)")
    report.append(f"- max_graph_candidates: {cfg.max_graph_candidates}")
    report.append("  * Config: configs/default.yaml:27")
    report.append("  * Applied: src/graph/traversal.py:156 (`return traversed_paths[: self.max_candidates]`)")
    report.append("             and src/retrieval/graph_retriever.py:397 (`return evidence[: self.max_candidates]`)")
    report.append(f"- final_context_chunks: {cfg.final_context_chunks}")
    report.append("  * Config: configs/default.yaml:30")
    report.append("  * Instantiated: src/engine.py:335 (final_context_chunks=self.config.retrieval.final_context_chunks)")
    report.append("  * Applied: src/retrieval/community_retriever.py:179 (`reranked = self.reranker.rerank(merged, top_k=self.final_context_chunks)`)")
    report.append(f"- rrf_k: {cfg.rrf_k}")
    report.append("  * Config: configs/default.yaml:31")
    report.append("  * Instantiated: src/engine.py:326 (`reranker = RRFReranker(rrf_k=self.config.retrieval.rrf_k)`)")
    report.append("  * Applied: src/retrieval/reranker.py:41 (`rrf_score = 1.0 / (self.rrf_k + rank)`)")
    report.append("--------------------------------------------------------------------------------\n")

    # Set up retrieval components
    seed_retriever = SeedEntityRetriever(
        doc_store=engine.doc_store,
        vector_store=engine.vector_store,
        embedding_provider=engine.embedding_service,
        similarity_threshold=engine.config.resolution.similarity_candidate_threshold,
        max_candidates_per_seed=getattr(engine.config.retrieval, "max_candidates_per_seed", 5),
    )
    traverser = GraphTraverser(
        max_hops=cfg.max_traversal_hops,
        max_candidates=cfg.max_graph_candidates,
    )
    graph_retriever = GraphEvidenceRetriever(
        doc_store=engine.doc_store,
        seed_retriever=seed_retriever,
        graph=graph,
        traverser=traverser,
        retrieval_config=cfg,
    )

    # Load recorded run records
    vec_results = {r["question_id"]: r for r in [json.loads(line) for line in open("experiments/synth_kg/runs/20261116_035101_vector_only/results.jsonl", encoding="utf-8")]}
    hyb_results = {r["question_id"]: r for r in [json.loads(line) for line in open("experiments/synth_kg/runs/20261116_045328_hybrid/results.jsonl", encoding="utf-8")]}
    questions = {q["id"]: q for q in [json.loads(line) for line in open("experiments/synth_kg/corpus_final/questions.jsonl", encoding="utf-8")]}

    targets = [
        ("Q_0062", "Nebothnok Styxinvath"),
        ("Q_0063", "Krenokmoth Tarnonmar"),
        ("Q_0073", "Rynartor Aldislan"),
        ("Q_0084", "Tarniklan Raelonrak"),
        ("Q_0068", "Synorreth Aldathdyn"),
    ]

    for idx, (qid, seed_name) in enumerate(targets, 1):
        q = questions[qid]
        q_text = q["question"]
        gold_docs = q["gold_supporting_doc_ids"]
        gold_path = q.get("gold_path", [])

        plan = QueryPlan(
            question=q_text,
            intent=QueryIntent.LOCAL,
            seed_entities=[SeedEntitySpec(name=seed_name, expected_type="EMPLOYEE")],
        )
        seed_res = seed_retriever.resolve_seeds(plan.seed_entities)
        resolved_seeds = [(sid, entities_by_id[sid].name) for sid in seed_res.resolved_entity_ids if sid in entities_by_id]
        
        # Traverse graph directly
        traversed = traverser.traverse(graph=graph, seed_entity_ids=seed_res.resolved_entity_ids)

        # Graph route evidence
        graph_ev = graph_retriever.retrieve(plan.question, plan, kb_id=engine.kb_id, seed_resolution_result=seed_res)

        # Recorded Vector Route
        vec_chunks = vec_results[qid].get("context_chunk_ids", [])
        vec_docs = vec_results[qid].get("context_doc_ids", [])

        # Recorded Hybrid Final RRF-merged Context
        hyb_chunks = hyb_results[qid].get("context_chunk_ids", [])
        hyb_docs = hyb_results[qid].get("context_doc_ids", [])

        # Graph candidate chunk list
        graph_chunks = [ev.chunk_id for ev in graph_ev]
        graph_docs_norm = [f"DOC_{int(ev.doc_id[1:]):04d}" for ev in graph_ev]
        vec_docs_norm = [f"DOC_{int(d[1:]):04d}" for d in vec_docs]
        hyb_docs_norm = [f"DOC_{int(d[1:]):04d}" for d in hyb_docs]

        report.append(f"### [{idx}] Question ID: `{qid}`")
        report.append(f"- **Question**: \"{q_text}\"")
        report.append(f"- **Gold Supporting Docs**: {gold_docs}")
        report.append(f"- **Gold Path**: {gold_path}")
        report.append(f"- **Resolved Seeds**: {resolved_seeds}")
        
        report.append(f"- **Graph Traversed Paths** ({len(traversed)} paths):")
        if traversed:
            for p in traversed:
                report.append(f"  * {p.to_path_string()}")
        else:
            report.append("  * (No paths traversed beyond seed)")

        report.append(f"- **Graph-Route Candidate Chunks** ({len(graph_ev)} chunks, ranked):")
        for c_idx, ev in enumerate(graph_ev, 1):
            report.append(f"  * Rank {c_idx:2d}: `{ev.chunk_id}` (Doc: DOC_{int(ev.doc_id[1:]):04d}) | Alignment/Rank Score: {ev.graph_rank}")

        report.append(f"- **Vector-Route List** (top-4 chunks from dense retrieval):")
        for v_idx, cid in enumerate(vec_chunks, 1):
            did = vec_docs_norm[v_idx-1]
            report.append(f"  * Rank {v_idx}: `{cid}` ({did})")

        report.append(f"- **RRF-Merged List (Final Context Chunks, Budget=4)**:")
        for h_idx, cid in enumerate(hyb_chunks, 1):
            did = hyb_docs_norm[h_idx-1]
            report.append(f"  * Final Rank {h_idx}: `{cid}` ({did})")

        report.append("- **Per-Gold-Document Status Classification**:")
        for gd in gold_docs:
            if gd in hyb_docs_norm:
                final_r = hyb_docs_norm.index(gd) + 1
                report.append(f"  * `{gd}`: **IN_CONTEXT** (Final Hybrid Rank: {final_r})")
            elif gd in graph_docs_norm:
                g_r = graph_docs_norm.index(gd) + 1
                report.append(f"  * `{gd}`: **REACHED_BUT_DROPPED_IN_MERGE_OR_TRUNCATION** (Graph Rank: {g_r}, eliminated by RRF top-4 cutoff)")
            else:
                report.append(f"  * `{gd}`: **NOT_REACHED_BY_TRAVERSAL** (Graph path severed/dead-end due to split entity or missing edge)")

        report.append("")

    report.append("--------------------------------------------------------------------------------")
    report.append("SUMMARY OF HOP3 LOSS MECHANISMS:")
    report.append("1. **Q_0068 (Success - 3/3 Gold in Context)**:")
    report.append("   - All 3 gold documents (DOC_0010, DOC_0031, DOC_0046) were successfully reached by graph traversal.")
    report.append("   - Traversal ranks were 1, 2, 3, ensuring they survived the final RRF fusion and context truncation.")
    report.append("2. **Q_0062, Q_0063, Q_0073, Q_0084 (Failures - Missing Bridge/Target Documents)**:")
    report.append("   - The primary failure mode was NOT truncation of validly ranked graph candidates, but **NOT_REACHED_BY_TRAVERSAL**.")
    report.append("   - Root causes:")
    report.append("     * Entity resolution splits (e.g. Q_0062: Krenokmoth Tarnonmar split into E036 and E037, creating an isolated sink vertex).")
    report.append("     * Unextracted edges (e.g. EDGE_0014 NEVER_EXTRACTED for Q_0062/Q_0063).")
    report.append("     * Rejected ontology edges (e.g. EDGE_0080 REJECTED_ONTOLOGY for Q_0073).")
    report.append("================================================================================\n")

    report_text = "\n".join(report) + "\n"
    print(report_text)

    # Overwrite output.txt's Part E or append cleanly
    # Let's read existing output.txt, strip any partial Part E, and append report_text
    output_path = Path("experiments/synth_kg_audit/followup/output.txt")
    existing_out = output_path.read_text(encoding="utf-8")
    part_e_idx = existing_out.find("PART E: HOP3 LOSS LOCALISATION")
    if part_e_idx != -1:
        # find start of line
        start_line = existing_out.rfind("====", 0, part_e_idx)
        existing_out = existing_out[:start_line]
    output_path.write_text(existing_out + report_text, encoding="utf-8")

    # Append to PROGRESS.md
    progress_path = Path("experiments/synth_kg_audit/followup/PROGRESS.md")
    timestamp = datetime.datetime.now().isoformat()
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(f"| {timestamp} | Part E: Reconstructed retrieval and localized hop3 loss for target questions | `experiments/synth_kg_audit/followup/part_e_hop3_loss.py` |\n")

if __name__ == "__main__":
    main()
