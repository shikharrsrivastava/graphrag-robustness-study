#!/usr/bin/env python3
"""
experiments/synth_kg_audit/followup/part_a_verify_citations.py
Verification of all citations from previous audit report.
"""

import sys
from pathlib import Path
import datetime

def format_citation(file_path: str, start_line: int, end_line: int, claim: str, verdict: str, explanation: str) -> str:
    p = Path(file_path)
    out = []
    out.append("--------------------------------------------------------------------------------")
    out.append(f"CITATION: {file_path}:{start_line}-{end_line}")
    out.append(f"CLAIM: {claim}")
    out.append(f"VERDICT: {verdict}")
    out.append(f"EXPLANATION: {explanation}")
    out.append("CONTEXT (with 3 lines before and after):")
    
    if not p.exists():
        out.append(f"  [ERROR] File does not exist: {file_path}")
        return "\n".join(out)
    
    lines = p.read_text(encoding="utf-8").splitlines()
    ctx_start = max(1, start_line - 3)
    ctx_end = min(len(lines), end_line + 3)
    for idx in range(ctx_start, ctx_end + 1):
        prefix = ">>" if start_line <= idx <= end_line else "  "
        out.append(f"{prefix} {idx:4d}: {lines[idx-1]}")
    return "\n".join(out)

def main():
    report = []
    report.append("================================================================================")
    report.append("PART A: VERIFICATION OF EVERY CITATION (READ-ONLY)")
    report.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
    report.append("================================================================================\n")

    # 1. src/resolution/entity_resolver.py line 197
    report.append(format_citation(
        "src/resolution/entity_resolver.py", 197, 197,
        "aliases initialised empty and never appended",
        "MATCHES",
        "Line 197 initializes aliases=[] on new CanonicalEntity creation. Inspection of the rest of "
        "entity_resolver.py confirms that _merge_into_canonical does not append mention surface names to canonical.aliases."
    ))

    # 2. src/resolution/entity_resolver.py line 210
    report.append(format_citation(
        "src/resolution/entity_resolver.py", 208, 212,
        "hardcoded c.entity_type == 'PERSON' gates surname merging",
        "MATCHES",
        "Lines 208-212 filter single_token_persons with `c.entity_type == 'PERSON'` and line 215 filters "
        "`multi_token_persons` with `c.entity_type == 'PERSON'`. Since company.yaml defines the entity type as 'EMPLOYEE', "
        "this post-pass never executes for company employees."
    ))

    # 3. src/resolution/entity_resolver.py line 327
    report.append(format_citation(
        "src/resolution/entity_resolver.py", 326, 328,
        "require_type_match skips differently-typed candidates",
        "MATCHES",
        "Line 327 checks `if self.resolution_config.require_type_match and mention.entity_type != canonical.entity_type: continue`. "
        "This strictly skips canonical entities whose entity_type does not match the mention's entity_type."
    ))

    # 4. src/retrieval/seed_retriever.py lines 136-138
    report.append(format_citation(
        "src/retrieval/seed_retriever.py", 136, 138,
        "name compatibility gated on entity_type == 'PERSON'",
        "MATCHES",
        "Line 136 checks `if expected_type == 'PERSON' or any(e.entity_type.upper() == 'PERSON' for e in exact_matches):` "
        "and line 138 checks `if e.entity_type.upper() == 'PERSON' and e not in exact_matches:`. This gates surname compatibility "
        "strictly to 'PERSON', bypassing 'EMPLOYEE' entities."
    ))

    # 5. src/graph/graph_builder.py lines 84-140 (claims: self-loop filter at 108; dedup key (source, target, relation_type))
    report.append(format_citation(
        "src/graph/graph_builder.py", 106, 110,
        "self-loop filter at line 108; dedup key (source, target, relation_type)",
        "DOES NOT MATCH",
        "Line 108 is the end of build_graph (`return g` at 107, blank at 108). The self-loop filter is actually located at lines 60-62 "
        "(`if src == tgt: continue`). Furthermore, the aggregation deduplication key is defined at line 64 as "
        "`(src, r.relation_type, tgt, r.scope_id)`, which includes scope_id and orders relation_type between source and target, "
        "not `(source, target, relation_type)`."
    ))
    report.append(format_citation(
        "src/graph/graph_builder.py", 60, 64,
        "Actual self-loop filter and deduplication key in src/graph/graph_builder.py",
        "MATCHES (Actual Location)",
        "Lines 60-64 contain the actual self-loop check (`if src == tgt: continue`) and deduplication dictionary key."
    ))

    # 6. src/extraction/relation_extractor.py lines 120-160
    report.append(format_citation(
        "src/extraction/relation_extractor.py", 120, 160,
        "prompt lacks disambiguation between BELONGS_TO / MANAGES / DEVELOPS",
        "MATCHES (Substance)",
        "Lines 120-160 comprise RelationExtractor.__init__ and the start of extract_relationships. The extraction prompt template "
        "(DEFAULT_RELATION_EXTRACTION_PROMPT, defined at lines 49-111 and assigned at line 144) provides general instructions on "
        "directionality and polarity, but contains NO disambiguation guidelines distinguishing BELONGS_TO, MANAGES, and DEVELOPS."
    ))

    # 7. src/retrieval/graph_retriever.py line 75
    report.append(format_citation(
        "src/retrieval/graph_retriever.py", 74, 76,
        "BFS depth 3",
        "DOES NOT MATCH",
        "Line 75 in src/retrieval/graph_retriever.py assigns `self.max_description_candidates_per_seed = retrieval_config.max_description_candidates_per_seed`. "
        "Graph traversal depth is not implemented or configured on line 75 of graph_retriever.py. Instead, traversal is delegated to "
        "GraphTraverser, where `max_hops: int = 3` is defined as default in src/graph/traversal.py line 46."
    ))

    # 8. src/retrieval/community_retriever.py lines 179 and 248, src/engine.py line 335
    report.append(format_citation(
        "src/retrieval/community_retriever.py", 178, 180,
        "final context truncated to final_context_chunks=4 after fusion (hybrid route)",
        "MATCHES",
        "Line 179 truncates the merged evidence using `reranked = self.reranker.rerank(merged, top_k=self.final_context_chunks)`."
    ))
    report.append(format_citation(
        "src/retrieval/community_retriever.py", 247, 249,
        "final context truncated to final_context_chunks=4 after fusion (community route)",
        "MATCHES",
        "Line 248 truncates the merged evidence using `reranked = self.reranker.rerank(merged, top_k=self.final_context_chunks)`."
    ))
    report.append(format_citation(
        "src/engine.py", 334, 336,
        "final_context_chunks configured in HybridRetrieverOrchestrator",
        "MATCHES",
        "Line 335 passes `final_context_chunks=self.config.retrieval.final_context_chunks` to HybridRetrieverOrchestrator."
    ))

    # 9. src/generation/answer_generator.py lines 45-85 and prompts/ files
    prompts_dir = Path("prompts")
    prompt_files = sorted([f.name for f in prompts_dir.iterdir() if f.is_file()])
    report.append("--------------------------------------------------------------------------------")
    report.append("PROMPTS DIRECTORY AUDIT:")
    report.append(f"Files in prompts/: {', '.join(prompt_files)}")
    report.append(format_citation(
        "src/generation/answer_generator.py", 45, 85,
        "prompt lacks relation-type compatibility instruction",
        "MATCHES",
        "Lines 45-80 define the tail of DEFAULT_ANSWER_GENERATION_PROMPT_V1_0, and lines 81-85 begin DEFAULT_ANSWER_GENERATION_PROMPT_V1_1. "
        "When prompt_version == 'v1.1', AnswerGenerator loads prompts/answer_generation_v1_1.txt (lines 773-776). "
        "Inspection of prompts/answer_generation_v1_1.txt confirms that while it contains sections for Grounding, Citations, Comparative & Selection, "
        "and Multi-hop & Bridge questions, it contains NO instructions instructing the LLM to verify relation-type compatibility "
        "(e.g., distinguishing whether an entity belongs to, manages, or develops another entity)."
    ))

    # 10. Ontology domain/range constraints definition and enforcement
    report.append("--------------------------------------------------------------------------------")
    report.append("ONTOLOGY DOMAIN/RANGE CONSTRAINTS DEFINITION & ENFORCEMENT:")
    report.append("Definition in configs/domains/company.yaml:")
    company_yaml = Path("configs/domains/company.yaml").read_text(encoding="utf-8")
    for line in company_yaml.splitlines()[:36]:
        report.append(f"  {line}")
    report.append("\nEnforcement in codebase:")
    report.append(format_citation(
        "src/core/ontology.py", 74, 85,
        "Ontology.validate_triple_types enforces allowed_source_types and allowed_target_types",
        "MATCHES",
        "Checks source_type against allowed_source_types and target_type against allowed_target_types."
    ))
    report.append(format_citation(
        "src/extraction/relation_extractor.py", 332, 339,
        "RelationExtractor enforces ontology validation before storing relationships",
        "MATCHES",
        "Lines 333-339 call `self.ontology.validate_triple_types(...)`. If False, validation_status is set to REJECTED_ONTOLOGY."
    ))
    report.append(format_citation(
        "src/graph/graph_builder.py", 44, 46,
        "IGraphBuilder filters for VALIDATED relationships",
        "MATCHES",
        "Line 45 filters `if r.validation_status != ValidationStatus.VALIDATED: continue`. The graph builder does NOT re-evaluate "
        "ontology domain/range directly, but inherits the enforcement executed during extraction validation."
    ))

    report_text = "\n".join(report) + "\n"
    print(report_text)

    # Write to output.txt
    output_path = Path("experiments/synth_kg_audit/followup/output.txt")
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(report_text)

    # Append to PROGRESS.md
    progress_path = Path("experiments/synth_kg_audit/followup/PROGRESS.md")
    timestamp = datetime.datetime.now().isoformat()
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(f"| {timestamp} | Part A: Verified all citations and ontology constraints | `experiments/synth_kg_audit/followup/part_a_verify_citations.py` |\n")

if __name__ == "__main__":
    main()
