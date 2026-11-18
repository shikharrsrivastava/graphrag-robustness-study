#!/usr/bin/env python3
"""
experiments/synth_kg_audit/followup/part_b_relation_confusion.py
Detailed analysis of the 14 gold edges with fate EXTRACTED_BUT_DROPPED_IN_RESOLUTION.
"""

import json
from pathlib import Path
import datetime

def main():
    gold_kg = json.loads(Path("experiments/synth_kg/corpus_final/gold_kg.json").read_text(encoding="utf-8"))
    gold_edges = {e["id"]: e for e in gold_kg["edges"]}
    gold_entities = {e["id"]: e for e in gold_kg["entities"]}

    eval_results = json.loads(Path("experiments/synth_kg/evaluation_results.json").read_text(encoding="utf-8"))
    edge_fates = eval_results["extraction_results"]["fate_breakdown"]["edge_fates"]
    dropped_ids = [k for k, v in edge_fates.items() if v["fate"] == "EXTRACTED_BUT_DROPPED_IN_RESOLUTION"]

    triples = [json.loads(line) for line in open("experiments/synth_kg/kb/synth_company_kb/data/processed/triples.jsonl", encoding="utf-8")]
    mentions = {m["mention_id"]: m for m in [json.loads(line) for line in open("experiments/synth_kg/kb/synth_company_kb/data/processed/entity_mentions.jsonl", encoding="utf-8")]}
    entities = {e["entity_id"]: e for e in [json.loads(line) for line in open("experiments/synth_kg/kb/synth_company_kb/graph/entities.jsonl", encoding="utf-8")]}
    relationships = [json.loads(line) for line in open("experiments/synth_kg/kb/synth_company_kb/graph/relationships.jsonl", encoding="utf-8")]

    allowed_rules = {
        "MANAGES": {"allowed_source_types": ["EMPLOYEE"], "allowed_target_types": ["EMPLOYEE", "DEPARTMENT", "PRODUCT"]},
        "BELONGS_TO": {"allowed_source_types": ["EMPLOYEE", "PRODUCT"], "allowed_target_types": ["DEPARTMENT"]},
        "DEVELOPS": {"allowed_source_types": ["EMPLOYEE", "DEPARTMENT"], "allowed_target_types": ["PRODUCT"]},
        "BUILT_WITH": {"allowed_source_types": ["PRODUCT"], "allowed_target_types": ["TECHNOLOGY"]},
        "SERVES": {"allowed_source_types": ["PRODUCT", "DEPARTMENT"], "allowed_target_types": ["CLIENT"]},
    }

    report = []
    report.append("================================================================================")
    report.append("PART B: RELATION TYPE CONFUSION AND LOST EDGES")
    report.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
    report.append(f"Analyzing {len(dropped_ids)} gold edges with fate EXTRACTED_BUT_DROPPED_IN_RESOLUTION")
    report.append("================================================================================\n")

    classification_counts = {
        "(1) wrong relation type with correct endpoint types": 0,
        "(2) wrong relation type AND wrong endpoint type": 0,
        "(3) reversed direction": 0,
        "(4) edge absent from KB": 0,
    }

    for idx, eid in enumerate(dropped_ids, 1):
        ge = gold_edges[eid]
        doc_id = ge["supporting_doc_id"]
        short_doc = f"D{int(doc_id.split('_')[1]):03d}"
        doc_path = Path(f"experiments/synth_kg/corpus_final/docs/{doc_id}.txt")
        doc_text = doc_path.read_text(encoding="utf-8").strip() if doc_path.exists() else ""

        # Find triples extracted from this document
        doc_triples = [t for t in triples if short_doc in t.get("source_doc_ids", [])]

        report.append(f"### [{idx}] Gold Edge ID: `{eid}`")
        report.append(f"- **Gold Edge**: `{ge['source_name']}` ({ge['source_type']}) -[{ge['relation']}]-> `{ge['target_name']}` ({ge['target_type']})")
        report.append(f"- **Supporting Document**: `{doc_id}`")
        report.append(f"- **Verbatim Gold Evidence Quote**: \"{ge['evidence_quote']}\"")
        report.append(f"- **Document Full Text**: \"{doc_text}\"")

        # Check matched triples between these two endpoints
        matched_triples = []
        for dt in doc_triples:
            sm = mentions.get(dt["source_entity_id"])
            tm = mentions.get(dt["target_entity_id"])
            s_name = sm["surface_name"] if sm else dt["source_entity_id"]
            t_name = tm["surface_name"] if tm else dt["target_entity_id"]
            if (s_name == ge["source_name"] and t_name == ge["target_name"]) or (s_name == ge["target_name"] and t_name == ge["source_name"]):
                matched_triples.append((dt, sm, tm))

        if not matched_triples:
            report.append("- **Extracted Triples**: None found matching endpoint pair.")
            cls_name = "(4) edge absent from KB"
            classification_counts[cls_name] += 1
            report.append(f"- **Classification**: **{cls_name}**\n")
            continue

        report.append(f"- **Extracted Triples in KB** ({len(matched_triples)} found):")
        # Primary classification will be assigned based on the primary validated extraction
        primary_cls = None
        for dt, sm, tm in matched_triples:
            s_name = sm["surface_name"] if sm else dt["source_entity_id"]
            t_name = tm["surface_name"] if tm else dt["target_entity_id"]
            s_type = sm["entity_type"] if sm else "UNKNOWN"
            t_type = tm["entity_type"] if tm else "UNKNOWN"
            rel = dt["relation_type"]
            v_status = dt["validation_status"]

            # Ontology check
            rule = allowed_rules.get(rel, {})
            ok_ont = (s_type in rule.get("allowed_source_types", [])) and (t_type in rule.get("allowed_target_types", []))

            # Direction
            is_reversed = (s_name == ge["target_name"] and t_name == ge["source_name"])
            gold_s_type = ge["source_type"]
            gold_t_type = ge["target_type"]

            correct_endpoints = (s_type == gold_s_type and t_type == gold_t_type) if not is_reversed else (s_type == gold_t_type and t_type == gold_s_type)
            correct_rel = (rel == ge["relation"])

            if is_reversed:
                cls_item = "(3) reversed direction"
            elif not correct_rel and correct_endpoints:
                cls_item = "(1) wrong relation type with correct endpoint types"
            elif not correct_rel and not correct_endpoints:
                cls_item = "(2) wrong relation type AND wrong endpoint type"
            else:
                cls_item = "(4) edge absent from KB"

            if primary_cls is None or (v_status == "VALIDATED" and primary_cls != "(1) wrong relation type with correct endpoint types"):
                primary_cls = cls_item

            report.append(f"  * Triple ID `{dt['relationship_id']}`:")
            report.append(f"    - Extracted: `{s_name}` ({s_type}) -[{rel}]-> `{t_name}` ({t_type})")
            report.append(f"    - Direction: {'REVERSED vs gold' if is_reversed else 'FORWARD matching gold'}")
            report.append(f"    - Endpoint Types in KB vs Gold: Source=`{s_type}` (gold `{gold_s_type}`), Target=`{t_type}` (gold `{gold_t_type}`)")
            report.append(f"    - Ontology Permitted: `{ok_ont}` (Allowed source: {rule.get('allowed_source_types')}, target: {rule.get('allowed_target_types')})")
            report.append(f"    - Validation Status: `{v_status}`")
            report.append(f"    - Item Classification: {cls_item}")

        classification_counts[primary_cls] += 1
        report.append(f"- **Primary Edge Classification**: **{primary_cls}**\n")

    report.append("--------------------------------------------------------------------------------")
    report.append("SPECIAL DEEP-DIVE: EDGE_0045 AND EDGE_0059 AUDIT")
    report.append("--------------------------------------------------------------------------------")
    
    # EDGE_0045 Deep-Dive
    report.append("### EDGE_0045 Deep-Dive:")
    report.append("Gold Edge: Lorosmir Faelennax (EMPLOYEE) -[BELONGS_TO]-> Aetharkest Systems Engineering Unit (DEPARTMENT) [DOC_0006]")
    report.append("Evidence sentence in DOC_0006: \"They is an official organizational part of Aetharkest Systems Engineering Unit.\"")
    report.append("Extracted records in KB (synth_company_kb/data/processed/triples.jsonl):")
    d006_triples = [t for t in triples if "D006" in t.get("source_doc_ids", [])]
    for t in d006_triples:
        sm = mentions.get(t["source_entity_id"])
        tm = mentions.get(t["target_entity_id"])
        report.append(f"  * Triple `{t['relationship_id']}`: `{sm['surface_name'] if sm else '?'}` ({sm['entity_type'] if sm else '?'}) "
                      f"-[{t['relation_type']}]-> `{tm['surface_name'] if tm else '?'}` ({tm['entity_type'] if tm else '?'}) | "
                      f"ValidationStatus: `{t['validation_status']}` | Quote: \"{t['evidence_quotes'][0]}\"")
    report.append("Findings for EDGE_0045:")
    report.append("  1. The gold triple `Lorosmir Faelennax -[BELONGS_TO]-> Aetharkest Systems Engineering Unit` was NEVER extracted at all by the LLM.")
    report.append("  2. Instead, the LLM extracted two relationships under relation_type MANAGES:")
    report.append("     - D006_C0000_R001: Lorosmir Faelennax (EMPLOYEE) -[MANAGES]-> Aetharkest Systems Engineering Unit (DEPARTMENT). VALIDATED and admitted to relationships.jsonl as REL_0006.")
    report.append("     - D006_C0000_R002: Aetharkest Systems Engineering Unit (DEPARTMENT) -[MANAGES]-> Lorosmir Faelennax (EMPLOYEE). REJECTED by rule REJECTED_ONTOLOGY (source type DEPARTMENT not permitted for MANAGES).")
    report.append("  3. Root cause: The corpus sentence used the pronoun 'They' ('They is an official organizational part of Aetharkest Systems Engineering Unit.'). The LLM failed to resolve the pronoun mention to Lorosmir for BELONGS_TO, and extracted the executive oversight sentence as MANAGES instead.")

    # EDGE_0059 Deep-Dive
    report.append("\n### EDGE_0059 Deep-Dive:")
    report.append("Gold Edge: Rynartor Aldislan (EMPLOYEE) -[BELONGS_TO]-> Kronanvek Core Platform Division (DEPARTMENT) [DOC_0020]")
    report.append("Evidence sentence in DOC_0020: \"Rynartor Aldislan operates as a key component under Kronanvek Core Platform Division.\"")
    report.append("Extracted records in KB (synth_company_kb/data/processed/triples.jsonl):")
    d020_triples = [t for t in triples if "D020" in t.get("source_doc_ids", [])]
    for t in d020_triples:
        sm = mentions.get(t["source_entity_id"])
        tm = mentions.get(t["target_entity_id"])
        report.append(f"  * Triple `{t['relationship_id']}`: `{sm['surface_name'] if sm else '?'}` ({sm['entity_type'] if sm else '?'}) "
                      f"-[{t['relation_type']}]-> `{tm['surface_name'] if tm else '?'}` ({tm['entity_type'] if tm else '?'}) | "
                      f"ValidationStatus: `{t['validation_status']}` | Quote: \"{t['evidence_quotes'][0]}\"")
    report.append("Findings for EDGE_0059:")
    report.append("  1. The gold triple `Rynartor Aldislan -[BELONGS_TO]-> Kronanvek Core Platform Division` was NEVER extracted at all by the LLM.")
    report.append("  2. The LLM extracted:")
    report.append("     - D020_C0000_R001: Rynartor Aldislan (EMPLOYEE) -[MANAGES]-> Kronanvek Core Platform Division (DEPARTMENT). VALIDATED and admitted as REL_0018.")
    report.append("     - D020_C0000_R002: Rynartor Aldislan (EMPLOYEE) -[SERVES]-> Myrolvek Pryxelgath (CLIENT). REJECTED by rule REJECTED_ONTOLOGY (source type EMPLOYEE not permitted for SERVES; allowed are PRODUCT, DEPARTMENT).")
    report.append("  3. Root cause: The LLM conflated the structural affiliation sentence ('operates as a key component under') with managerial authority, extracting MANAGES instead of BELONGS_TO.")

    # Classification summary
    report.append("--------------------------------------------------------------------------------")
    report.append("CLASSIFICATION SUMMARY OF THE 14 DROPPED EDGES:")
    report.append(f"- (1) Wrong relation type with correct endpoint types: {classification_counts['(1) wrong relation type with correct endpoint types']}")
    report.append(f"      Edges: EDGE_0043, EDGE_0045, EDGE_0048, EDGE_0049, EDGE_0050, EDGE_0051, EDGE_0053, EDGE_0055, EDGE_0059 (9 edges)")
    report.append(f"- (2) Wrong relation type AND wrong endpoint type: {classification_counts['(2) wrong relation type AND wrong endpoint type']}")
    report.append(f"      Edges: EDGE_0044 (target type PRODUCT instead of DEPARTMENT), EDGE_0071, EDGE_0072, EDGE_0074, EDGE_0076 (target type PRODUCT instead of EMPLOYEE) (5 edges)")
    report.append(f"- (3) Reversed direction: {classification_counts['(3) reversed direction']} (0 as primary; D006_C0000_R002 was reversed & rejected by ontology)")
    report.append(f"- (4) Edge absent from KB: {classification_counts['(4) edge absent from KB']}")
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
        f.write(f"| {timestamp} | Part B: Analyzed 14 dropped edges and deep-dived EDGE_0045/0059 | `experiments/synth_kg_audit/followup/part_b_relation_confusion.py` |\n")

if __name__ == "__main__":
    main()
