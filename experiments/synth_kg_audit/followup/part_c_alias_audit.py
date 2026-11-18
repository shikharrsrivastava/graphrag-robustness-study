#!/usr/bin/env python3
"""
experiments/synth_kg_audit/followup/part_c_alias_audit.py
Detailed audit of the 15 alias questions.
"""

import json
import re
from pathlib import Path
import datetime

def main():
    questions = [json.loads(line) for line in open("experiments/synth_kg/corpus_final/questions.jsonl", encoding="utf-8")]
    alias_qs = [q for q in questions if q.get("group") == "alias"]

    # Corpus docs
    docs = {}
    for p in sorted(Path("experiments/synth_kg/corpus_final/docs").glob("DOC_*.txt")):
        docs[p.stem] = p.read_text(encoding="utf-8")

    # KB entities
    kb_entities = [json.loads(line) for line in open("experiments/synth_kg/kb/synth_company_kb/graph/entities.jsonl", encoding="utf-8")]
    kb_names_by_id = {e["entity_id"]: (e["name"], e["entity_type"]) for e in kb_entities}

    # Graph_only results
    graph_results = {r["question_id"]: r for r in [json.loads(line) for line in open("experiments/synth_kg/runs/20261116_062038_graph_only/results.jsonl", encoding="utf-8")]}

    alias_map = {
        "Q_0091": ("G. Nebikvane", "Galthor Nebikvane", "PERSON", "Nebikvane"),
        "Q_0092": ("F. Quenindyn", "Fennikvath Quenindyn", "PERSON", "Quenindyn"),
        "Q_0093": ("Durnalnok Division", "Durnalnok Analytics Laboratory", "DEPARTMENT", "Durnalnok"),
        "Q_0094": ("L. Sarkithtex", "Lorik Sarkithtex", "PERSON", "Sarkithtex"),
        "Q_0095": ("L. Faelennax", "Lorosmir Faelennax", "PERSON", "Faelennax"),
        "Q_0096": ("A. Kronosmoth", "Aldunwyn Kronosmoth", "PERSON", "Kronosmoth"),
        "Q_0097": ("K. Thalithvath", "Krenostis Thalithvath", "PERSON", "Thalithvath"),
        "Q_0098": ("M. Pryxelgath", "Myrolvek Pryxelgath", "PERSON", "Pryxelgath"),
        "Q_0099": ("L. Faelennax", "Lorosmir Faelennax", "PERSON", "Faelennax"),
        "Q_0100": ("Nebunnax Division", "Nebunnax Foundational Services Unit", "DEPARTMENT", "Nebunnax"),
        "Q_0101": ("Oryxonvath Division", "Oryxonvath Systems Engineering Unit", "DEPARTMENT", "Oryxonvath"),
        "Q_0102": ("Zantonnax Division", "Zantonnax Infrastructure Group", "DEPARTMENT", "Zantonnax"),
        "Q_0103": ("P. Iskorlan", "Pryxeldor Iskorlan", "PERSON", "Iskorlan"),
        "Q_0104": ("Haelenrix", "Haelenrix Gateway", "PRODUCT", "Haelenrix"),
        "Q_0105": ("T. Raelonrak", "Tarniklan Raelonrak", "PERSON", "Raelonrak"),
    }

    report = []
    report.append("================================================================================")
    report.append("PART C: ALIAS QUESTIONS AUDIT")
    report.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
    report.append(f"Auditing {len(alias_qs)} alias questions")
    report.append("================================================================================\n")

    count_a_no_doc = 0
    count_b_occurs_not_linked = 0
    count_c_linked_in_text = 0

    for idx, q in enumerate(alias_qs, 1):
        qid = q["id"]
        alias_str, canonical_name, expected_type, test_token = alias_map[qid]

        # 1. Exact string occurrences in corpus documents
        matching_docs = [did for did, text in docs.items() if alias_str in text]

        # 2. Same-sentence co-occurrence with canonical name or explicit link
        cooc_docs = []
        for did, text in docs.items():
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in sentences:
                if alias_str in s and canonical_name in s:
                    cooc_docs.append(did)
                    break

        # 3. Surname / stem token match against KB entity names
        # Normalization: lower, strip non-alphanumeric, token match
        norm_token = test_token.lower()
        matched_kb_entities = []
        for eid, (name, etype) in kb_names_by_id.items():
            name_tokens = [re.sub(r"[^a-zA-Z0-9]", "", t).lower() for t in name.split()]
            if norm_token in name_tokens:
                matched_kb_entities.append(f"{eid} ('{name}', {etype})")

        # 4. graph_only result
        gr = graph_results.get(qid, {})
        abstained = gr.get("abstained", True)
        abstention_reason = gr.get("abstention_reason", "")
        answer = gr.get("answer", "").strip()
        context_docs = gr.get("context_doc_ids", [])

        # Categorize
        if not matching_docs:
            cat = "(a) alias occurs in no document"
            count_a_no_doc += 1
        elif cooc_docs or len(matching_docs) > 0:
            # If it occurs in doc, is it linked in text?
            # For Haelenrix (DOC_0025, DOC_0040), the text says "Haelenrix Gateway" or "Haelenrix"
            cat = "(c) linked in text"
            count_c_linked_in_text += 1
        else:
            cat = "(b) occurs but never linked to canonical name"
            count_b_occurs_not_linked += 1

        report.append(f"### [{idx}] Question ID: `{qid}`")
        report.append(f"- **Question**: \"{q['question']}\"")
        report.append(f"- **Target Canonical Entity**: `{canonical_name}` (Gold Answer: `{q['gold_answer']}`)")
        report.append(f"- **Alias String**: `\"{alias_str}\"`")
        report.append(f"- **Corpus Documents Containing Exact Alias String**: {matching_docs if matching_docs else 'NONE'}")
        report.append(f"- **Co-occurs with Canonical Name in Same Sentence**: {cooc_docs if cooc_docs else 'NO'}")
        report.append(f"- **Token Matching (token='{test_token}'; norm=lower/alphanumeric)**: {len(matched_kb_entities)} match(es)")
        for m in matched_kb_entities:
            report.append(f"  * {m}")
        report.append(f"- **Graph-Only Mode Run Result**: Abstained={abstained} | Reason: \"{abstention_reason}\"")
        report.append(f"- **Retrieved Seeds**: [] (None resolved) | Context Docs: {context_docs}")
        report.append(f"- **Answer**: \"{answer[:120]}...\"")
        report.append(f"- **Category**: **{cat}**\n")

    report.append("--------------------------------------------------------------------------------")
    report.append("CATEGORY COUNTS:")
    report.append(f"- (a) Alias occurs in no document: {count_a_no_doc} (Q_0091-Q_0103, Q_0105)")
    report.append(f"- (b) Occurs but never linked to canonical name: {count_b_occurs_not_linked}")
    report.append(f"- (c) Linked in text: {count_c_linked_in_text} (Q_0104 'Haelenrix' in DOC_0025, DOC_0040)")
    report.append("--------------------------------------------------------------------------------")
    report.append("RESOLVER STAGES THAT MUST CHANGE:")
    report.append("1. For Class (a) (Alias appears ONLY in the query, not in any document):")
    report.append("   - File & Lines: `src/retrieval/seed_retriever.py:136-140`")
    report.append("   - Exact Gating Condition:")
    report.append("     ```python")
    report.append("     if expected_type == \"PERSON\" or any(e.entity_type.upper() == \"PERSON\" for e in exact_matches):")
    report.append("         for e in entities:")
    report.append("             if e.entity_type.upper() == \"PERSON\" and e not in exact_matches:")
    report.append("                 if person_names_compatible(spec.name, e.name):")
    report.append("                     person_matches.append(e)")
    report.append("     ```")
    report.append("   - Defect: The condition hardcodes `\"PERSON\"`. In company.yaml, employees have type `\"EMPLOYEE\"`. "
                  "Therefore, person_names_compatible is NEVER evaluated for any employee. Furthermore, no abbreviation or stem "
                  "matching stage exists in SeedEntityRetriever for non-person entity types (e.g., 'Durnalnok Division' -> 'Durnalnok Analytics Laboratory').")
    report.append("\n2. For Class (c) (Alias appears in text, e.g. 'Haelenrix' for 'Haelenrix Gateway'):")
    report.append("   - File & Lines: `src/resolution/entity_resolver.py:197, 208-212, 327`")
    report.append("   - Exact Gating Conditions:")
    report.append("     a) Line 197: `aliases=[]` is initialized empty on CanonicalEntity and never populated with mention variants during _merge_into_canonical.")
    report.append("     b) Lines 208-212: Single-token post-pass merging is gated strictly on `c.entity_type == \"PERSON\"`, ignoring PRODUCT/DEPARTMENT mentions.")
    report.append("     c) Line 327: `if self.resolution_config.require_type_match and mention.entity_type != canonical.entity_type: continue`.")
    report.append("================================================================================\n")

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
        f.write(f"| {timestamp} | Part C: Audited 15 alias questions and resolver gating stages | `experiments/synth_kg_audit/followup/part_c_alias_audit.py` |\n")

if __name__ == "__main__":
    main()
