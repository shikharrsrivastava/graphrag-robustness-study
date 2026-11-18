#!/usr/bin/env python3
"""Procedural Synthetic Knowledge Graph and Corpus Generator.

Generates:
1. Gold Knowledge Graph (gold_kg.json): strictly respecting company.yaml ontology.
2. Synthetic Documents (.txt): 3-6 sentences per doc, with paraphrasing, aliases, pronouns, distractors.
3. Questions (questions.jsonl): 1-hop, 2-hop, 3-hop, alias, and unanswerable questions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Real-world name blacklist to guarantee zero real-world names
REAL_WORLD_NAME_BLACKLIST = {
    "google", "microsoft", "apple", "amazon", "meta", "facebook", "ibm", "intel",
    "oracle", "cisco", "nvidia", "uber", "airbnb", "netflix", "tesla", "twitter",
    "john", "mary", "david", "michael", "james", "robert", "william", "sarah",
    "emily", "jessica", "smith", "johnson", "williams", "brown", "jones", "miller",
    "davis", "garcia", "rodriguez", "wilson", "martinez", "anderson", "taylor",
    "london", "paris", "berlin", "tokyo", "beijing", "new york", "chicago",
    "boston", "seattle", "austin", "california", "texas", "washington", "england",
    "france", "germany", "japan", "china", "america", "united states", "usa", "uk"
}

# Invented syllable components
SYLLABLE_PREFIXES = [
    "Velt", "Kael", "Zor", "Thal", "Syn", "Neb", "Aeth", "Kron", "Myr", "Xan",
    "Brec", "Drak", "Fael", "Gor", "Jor", "Lyr", "Mor", "Prax", "Ryn", "Sark",
    "Tael", "Vand", "Wyr", "Xylo", "Yv", "Zant", "Ald", "Bryn", "Cael", "Durn",
    "Eld", "Fenn", "Gryx", "Hael", "Isk", "Jol", "Kren", "Lor", "Mael", "Nox",
    "Oryx", "Pryx", "Quen", "Rael", "Styx", "Tarn", "Ulx", "Vael", "Wren", "Zan"
]

SYLLABLE_MIDDLES = [
    "ar", "el", "in", "or", "un", "ath", "eth", "ith", "oth", "uth",
    "an", "en", "on", "ik", "ok", "al", "ol", "is", "os", "us"
]

SYLLABLE_SUFFIXES = [
    "ron", "vath", "dor", "nax", "tor", "lok", "vix", "rak", "zor", "moth",
    "ric", "lan", "tis", "vane", "gath", "nok", "reth", "wyn", "kest", "vek",
    "mir", "zar", "fen", "vor", "dyn", "tex", "gon", "lux", "mar", "rix"
]

PERSON_TITLES = [
    "Lead Architect", "Principal Engineer", "Director of Research",
    "Staff Systems Engineer", "Senior VP of Operations", "Chief Specialist"
]

DEPT_SUFFIXES = [
    "Core Platform Division", "Systems Engineering Unit", "Analytics Laboratory",
    "Infrastructure Group", "Applied Automation Wing", "Foundational Services Unit"
]

PRODUCT_SUFFIXES = [
    "Mesh", "Engine", "Gateway", "Nexus", "Matrix", "Protocol", "Fabric", "Workspace",
    "Pipeline", "Relay", "Cluster", "Hub"
]

TECH_SUFFIXES = [
    "Lang", "DB", "RPC", "QL", "FS", "Net", "VM", "Stream", "Graph", "Store"
]

CLIENT_SUFFIXES = [
    "Enterprises", "Aerospace", "Logistics", "Energy Group", "Technologies",
    "Holdings", "Networks", "Syndicate", "Consortium"
]

PARAPHRASE_TEMPLATES: Dict[str, List[str]] = {
    "MANAGES": [
        "{src} serves as the primary manager of {tgt}.",
        "{src} actively manages and directs {tgt}.",
        "{src} oversees the day-to-day operations of {tgt}.",
        "Executive oversight of {tgt} is handled by {src}."
    ],
    "BELONGS_TO": [
        "{src} formally belongs to {tgt}.",
        "{src} is an official organizational part of {tgt}.",
        "{src} operates as a key component under {tgt}.",
        "{src} is structurally integrated into {tgt}."
    ],
    "DEVELOPS": [
        "{src} is directly responsible for developing {tgt}.",
        "{src} builds and maintains {tgt}.",
        "{src} designs and engineers {tgt}.",
        "{src} spearheads the technical development of {tgt}."
    ],
    "BUILT_WITH": [
        "{src} is engineered primarily using {tgt}.",
        "{src} relies on {tgt} as its foundational technology stack.",
        "{src} was constructed upon the {tgt} framework.",
        "{src} utilizes {tgt} for core functionality."
    ],
    "SERVES": [
        "{src} provides specialized commercial solutions for {tgt}.",
        "{src} directly serves the enterprise requirements of {tgt}.",
        "{src} delivers dedicated platform support to {tgt}.",
        "{src} operates in production serving {tgt}."
    ],
}


def generate_invented_word(rng: random.Random) -> str:
    """Generate a single invented word from syllables."""
    p = rng.choice(SYLLABLE_PREFIXES)
    m = rng.choice(SYLLABLE_MIDDLES)
    s = rng.choice(SYLLABLE_SUFFIXES)
    word = f"{p}{m}{s}"
    assert word.lower() not in REAL_WORLD_NAME_BLACKLIST
    return word


def generate_invented_person_name(rng: random.Random) -> Tuple[str, List[str]]:
    """Generate invented person canonical name and aliases."""
    first = generate_invented_word(rng)
    last = generate_invented_word(rng)
    canonical = f"{first} {last}"
    aliases = [f"{first[0]}. {last}", last]
    return canonical, aliases


def generate_invented_entity_name(etype: str, rng: random.Random) -> Tuple[str, List[str]]:
    """Generate typed entity canonical name and aliases."""
    root = generate_invented_word(rng)
    if etype == "EMPLOYEE":
        return generate_invented_person_name(rng)
    elif etype == "DEPARTMENT":
        suff = rng.choice(DEPT_SUFFIXES)
        canonical = f"{root} {suff}"
        aliases = [f"{root} Division", root]
        return canonical, aliases
    elif etype == "PRODUCT":
        suff = rng.choice(PRODUCT_SUFFIXES)
        canonical = f"{root} {suff}"
        aliases = [root]
        return canonical, aliases
    elif etype == "TECHNOLOGY":
        suff = rng.choice(TECH_SUFFIXES)
        canonical = f"{root}{suff}"
        aliases = [root]
        return canonical, aliases
    elif etype == "CLIENT":
        suff = rng.choice(CLIENT_SUFFIXES)
        canonical = f"{root} {suff}"
        aliases = [root]
        return canonical, aliases
    else:
        raise ValueError(f"Unknown entity type: {etype}")


class SyntheticCorpusGenerator:
    """Generates the synthetic gold KG, documents, and questions deterministically."""

    def __init__(
        self,
        seed: int = 7,
        n_docs: int = 60,
        alias_rate: float = 0.3,
        pronoun_rate: float = 0.2,
        distractor_rate: float = 0.3,
        questions_per_hop: int = 30,
        n_unanswerable: int = 15,
        n_alias: int = 15,
    ) -> None:
        self.seed = seed
        self.n_docs = n_docs
        self.alias_rate = alias_rate
        self.pronoun_rate = pronoun_rate
        self.distractor_rate = distractor_rate
        self.questions_per_hop = questions_per_hop
        self.n_unanswerable = n_unanswerable
        self.n_alias = n_alias

        self.rng = random.Random(seed)
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.questions: List[Dict[str, Any]] = []

    def build_kg(self) -> None:
        """Create typed entities and deterministic single-outgoing-relation edges."""
        # Distribute entity counts across types
        # For N docs, subject entities:
        # EMPLOYEE: ~35%, DEPARTMENT: ~25%, PRODUCT: ~20%, TECH: ~10%, CLIENT: ~10%
        n_emp = max(8, int(self.n_docs * 0.35))
        n_dept = max(6, int(self.n_docs * 0.25))
        n_prod = max(5, int(self.n_docs * 0.20))
        n_tech = max(3, int(self.n_docs * 0.10))
        n_client = self.n_docs - (n_emp + n_dept + n_prod + n_tech)
        if n_client < 3:
            n_client = 3
            self.n_docs = n_emp + n_dept + n_prod + n_tech + n_client

        type_counts = [
            ("EMPLOYEE", n_emp),
            ("DEPARTMENT", n_dept),
            ("PRODUCT", n_prod),
            ("TECHNOLOGY", n_tech),
            ("CLIENT", n_client),
        ]

        used_names: Set[str] = set()
        entity_id_counter = 1

        entities_by_type: Dict[str, List[Dict[str, Any]]] = {
            "EMPLOYEE": [],
            "DEPARTMENT": [],
            "PRODUCT": [],
            "TECHNOLOGY": [],
            "CLIENT": [],
        }

        for etype, count in type_counts:
            for _ in range(count):
                while True:
                    canonical, aliases = generate_invented_entity_name(etype, self.rng)
                    if canonical not in used_names and canonical.lower() not in REAL_WORLD_NAME_BLACKLIST:
                        used_names.add(canonical)
                        for a in aliases:
                            used_names.add(a)
                        break

                eid = f"ENT_{entity_id_counter:04d}"
                entity_id_counter += 1
                doc_id = f"DOC_{len(self.entities) + 1:04d}"

                ent = {
                    "id": eid,
                    "canonical_name": canonical,
                    "type": etype,
                    "aliases": aliases,
                    "doc_id": doc_id,
                }
                self.entities[eid] = ent
                entities_by_type[etype].append(ent)

        # Build clean multi-hop paths ensuring unique outgoing edge per relation
        # Edge 1: EMPLOYEE -> BELONGS_TO -> DEPARTMENT
        # Edge 2: DEPARTMENT -> DEVELOPS -> PRODUCT
        # Edge 3: PRODUCT -> BUILT_WITH -> TECHNOLOGY
        # Edge 4: PRODUCT -> SERVES -> CLIENT
        # Edge 5: EMPLOYEE -> MANAGES -> DEPARTMENT / EMPLOYEE / PRODUCT

        edge_counter = 1
        # Track outgoing edges: (source_id, relation) -> target_id
        outgoing_map: Dict[Tuple[str, str], str] = {}

        def add_edge(src: Dict[str, Any], rel: str, tgt: Dict[str, Any]) -> None:
            nonlocal edge_counter
            key = (src["id"], rel)
            if key in outgoing_map:
                return  # enforce at most 1 outgoing edge per relation type for determinism
            outgoing_map[key] = tgt["id"]

            edge_id = f"EDGE_{edge_counter:04d}"
            edge_counter += 1
            self.edges.append({
                "id": edge_id,
                "source": src["id"],
                "source_name": src["canonical_name"],
                "source_type": src["type"],
                "relation": rel,
                "target": tgt["id"],
                "target_name": tgt["canonical_name"],
                "target_type": tgt["type"],
                "supporting_doc_id": src["doc_id"],
                "evidence_quote": "",  # populated during document generation
            })

        # Connect DEPARTMENT -> DEVELOPS -> PRODUCT
        for idx, dept in enumerate(entities_by_type["DEPARTMENT"]):
            prod = entities_by_type["PRODUCT"][idx % len(entities_by_type["PRODUCT"])]
            add_edge(dept, "DEVELOPS", prod)

        # Connect PRODUCT -> BUILT_WITH -> TECHNOLOGY
        for idx, prod in enumerate(entities_by_type["PRODUCT"]):
            tech = entities_by_type["TECHNOLOGY"][idx % len(entities_by_type["TECHNOLOGY"])]
            add_edge(prod, "BUILT_WITH", tech)

        # Connect PRODUCT -> SERVES -> CLIENT
        for idx, prod in enumerate(entities_by_type["PRODUCT"]):
            client = entities_by_type["CLIENT"][idx % len(entities_by_type["CLIENT"])]
            add_edge(prod, "SERVES", client)

        # Connect EMPLOYEE -> BELONGS_TO -> DEPARTMENT
        for idx, emp in enumerate(entities_by_type["EMPLOYEE"]):
            dept = entities_by_type["DEPARTMENT"][idx % len(entities_by_type["DEPARTMENT"])]
            add_edge(emp, "BELONGS_TO", dept)

        # Connect some EMPLOYEE -> MANAGES -> DEPARTMENT or EMPLOYEE
        for idx in range(len(entities_by_type["EMPLOYEE"]) // 2):
            emp = entities_by_type["EMPLOYEE"][idx]
            dept = entities_by_type["DEPARTMENT"][idx % len(entities_by_type["DEPARTMENT"])]
            add_edge(emp, "MANAGES", dept)

        for idx in range(len(entities_by_type["EMPLOYEE"]) // 2, len(entities_by_type["EMPLOYEE"]) - 1):
            emp = entities_by_type["EMPLOYEE"][idx]
            subord = entities_by_type["EMPLOYEE"][idx + 1]
            add_edge(emp, "MANAGES", subord)

        # Connect some DEPARTMENT -> SERVES -> CLIENT
        for idx, dept in enumerate(entities_by_type["DEPARTMENT"]):
            if (dept["id"], "SERVES") not in outgoing_map:
                client = entities_by_type["CLIENT"][(idx + 1) % len(entities_by_type["CLIENT"])]
                add_edge(dept, "SERVES", client)

    def generate_documents(self) -> None:
        """Create text documents with exact verbatim evidence quotes."""
        # Group edges by source doc_id
        edges_by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for edge in self.edges:
            doc_id = edge["supporting_doc_id"]
            edges_by_doc.setdefault(doc_id, []).append(edge)

        for eid, ent in self.entities.items():
            doc_id = ent["doc_id"]
            cname = ent["canonical_name"]
            etype = ent["type"]
            aliases = ent["aliases"]

            # Subject name variation in document text
            use_alias = (self.rng.random() < self.alias_rate) and len(aliases) > 0
            subj_name = aliases[0] if use_alias else cname

            sentences = []
            # Sentence 1: Intro
            if etype == "EMPLOYEE":
                sentences.append(f"{cname} is a certified staff professional within the enterprise infrastructure.")
            elif etype == "DEPARTMENT":
                sentences.append(f"{cname} operates as a central functional division of the organization.")
            elif etype == "PRODUCT":
                sentences.append(f"{cname} is a proprietary enterprise software solution deployed across production clusters.")
            elif etype == "TECHNOLOGY":
                sentences.append(f"{cname} is an internal software specification maintained for operational stability.")
            elif etype == "CLIENT":
                sentences.append(f"{cname} is an external commercial partner collaborating on enterprise initiatives.")

            # Outgoing edge sentences
            doc_edges = edges_by_doc.get(doc_id, [])
            for edge in doc_edges:
                rel = edge["relation"]
                tgt_name = edge["target_name"]

                templates = PARAPHRASE_TEMPLATES[rel]
                tmpl = self.rng.choice(templates)

                # Subject representation for this sentence
                use_pronoun = self.rng.random() < self.pronoun_rate
                if use_pronoun:
                    pron = "They" if etype == "EMPLOYEE" else "It"
                    sent = tmpl.format(src=pron, tgt=tgt_name)
                    # Evidence quote must be unambiguous: store the full sentence
                    edge["evidence_quote"] = sent
                    sentences.append(sent)
                else:
                    sent = tmpl.format(src=cname, tgt=tgt_name)
                    edge["evidence_quote"] = sent
                    sentences.append(sent)

            # Optional distractor entity sentence
            if self.rng.random() < self.distractor_rate:
                # Create a distractor entity sharing first word
                first_word = cname.split()[0]
                distractor_name = f"{first_word} Systems" if etype != "EMPLOYEE" else f"{first_word} Kren"
                distractor_fact = f"In contrast, {distractor_name} maintains entirely distinct non-production responsibilities."
                sentences.append(distractor_fact)

            # Filler sentence to guarantee 3-6 sentences
            while len(sentences) < 3:
                sentences.append(f"All formal operational guidelines regarding {cname} are cataloged in internal records.")
            if len(sentences) > 6:
                sentences = sentences[:6]

            doc_text = " ".join(sentences)

            self.documents[doc_id] = {
                "doc_id": doc_id,
                "subject_entity_id": eid,
                "subject_name": cname,
                "text": doc_text,
                "sentences": sentences,
            }

    def generate_questions(self) -> None:
        """Generate 1-hop, 2-hop, 3-hop, alias, and unanswerable questions."""
        adj: Dict[str, List[Dict[str, Any]]] = {}
        for edge in self.edges:
            adj.setdefault(edge["source"], []).append(edge)

        # 1. 1-hop paths
        paths_1hop = []
        for e1 in self.edges:
            paths_1hop.append([e1])

        # 2. 2-hop paths
        paths_2hop = []
        for e1 in self.edges:
            s2 = e1["target"]
            for e2 in adj.get(s2, []):
                # Ensure no cycle and different documents
                if e2["target"] != e1["source"] and e2["supporting_doc_id"] != e1["supporting_doc_id"]:
                    paths_2hop.append([e1, e2])

        # 3. 3-hop paths
        paths_3hop = []
        for e1 in self.edges:
            s2 = e1["target"]
            for e2 in adj.get(s2, []):
                if e2["target"] == e1["source"] or e2["supporting_doc_id"] == e1["supporting_doc_id"]:
                    continue
                s3 = e2["target"]
                for e3 in adj.get(s3, []):
                    if (
                        e3["target"] not in (e1["source"], s2)
                        and e3["supporting_doc_id"] not in (e1["supporting_doc_id"], e2["supporting_doc_id"])
                    ):
                        paths_3hop.append([e1, e2, e3])

        # Shuffle deterministically
        self.rng.shuffle(paths_1hop)
        self.rng.shuffle(paths_2hop)
        self.rng.shuffle(paths_3hop)

        def make_question_text(path: List[Dict[str, Any]], use_alias: bool = False) -> str:
            first_ent = self.entities[path[0]["source"]]
            start_name = first_ent["aliases"][0] if use_alias and first_ent["aliases"] else first_ent["canonical_name"]

            rels = [e["relation"] for e in path]

            if len(path) == 1:
                r = rels[0]
                if r == "BELONGS_TO":
                    return f"Which department does {start_name} belong to?"
                elif r == "DEVELOPS":
                    return f"What product is developed by {start_name}?"
                elif r == "BUILT_WITH":
                    return f"What technology is {start_name} built with?"
                elif r == "SERVES":
                    return f"Which client does {start_name} serve?"
                elif r == "MANAGES":
                    return f"What entity or unit is managed by {start_name}?"
                else:
                    return f"What is the target of {r} for {start_name}?"

            elif len(path) == 2:
                r1, r2 = rels
                if r1 == "BELONGS_TO" and r2 == "DEVELOPS":
                    return f"What product is developed by the department that {start_name} belongs to?"
                elif r1 == "BELONGS_TO" and r2 == "SERVES":
                    return f"Which client is served by the department that {start_name} belongs to?"
                elif r1 == "DEVELOPS" and r2 == "BUILT_WITH":
                    return f"What technology is used in the product developed by {start_name}?"
                elif r1 == "DEVELOPS" and r2 == "SERVES":
                    return f"Which client is served by the product developed by {start_name}?"
                elif r1 == "MANAGES" and r2 == "DEVELOPS":
                    return f"What product is developed by the department managed by {start_name}?"
                elif r1 == "MANAGES" and r2 == "BELONGS_TO":
                    return f"Which department does the employee managed by {start_name} belong to?"
                elif r1 == "PRODUCT" and r2 == "BUILT_WITH":
                    return f"What technology is used in {start_name}?"
                else:
                    return f"Starting from {start_name}, traversing {r1} then {r2}, what is the target entity?"

            elif len(path) == 3:
                r1, r2, r3 = rels
                if r1 == "BELONGS_TO" and r2 == "DEVELOPS" and r3 == "BUILT_WITH":
                    return f"What technology is used in the product developed by the department that {start_name} belongs to?"
                elif r1 == "BELONGS_TO" and r2 == "DEVELOPS" and r3 == "SERVES":
                    return f"Which client is served by the product developed by the department that {start_name} belongs to?"
                elif r1 == "MANAGES" and r2 == "DEVELOPS" and r3 == "BUILT_WITH":
                    return f"What technology is used in the product developed by the unit managed by {start_name}?"
                elif r1 == "MANAGES" and r2 == "DEVELOPS" and r3 == "SERVES":
                    return f"Which client is served by the product developed by the unit managed by {start_name}?"
                elif r1 == "MANAGES" and r2 == "BELONGS_TO" and r3 == "DEVELOPS":
                    return f"What product is developed by the department of the employee managed by {start_name}?"
                elif r1 == "MANAGES" and r2 == "MANAGES" and r3 == "BELONGS_TO":
                    return f"Which department does the employee managed by the staff under {start_name} belong to?"
                elif r1 == "MANAGES" and r2 == "MANAGES" and r3 == "DEVELOPS":
                    return f"What product is developed by the unit managed under {start_name}?"
                else:
                    return f"Starting from {start_name}, traversing {r1}, {r2}, and {r3}, what is the target entity?"

            raise ValueError(f"Unsupported path length: {len(path)}")

        q_id_counter = 1

        # Helper to filter bridge document leakage
        def is_path_leak_free(path: List[Dict[str, Any]]) -> bool:
            if len(path) <= 1:
                return True
            first_ent = self.entities[path[0]["source"]]
            f_name = first_ent["canonical_name"].lower()
            f_aliases = [a.lower() for a in first_ent["aliases"]]

            # Check all bridge and target documents (indices 1 to len(path)-1)
            for edge in path[1:]:
                doc_text = self.documents[edge["supporting_doc_id"]]["text"].lower()
                if f_name in doc_text:
                    return False
                for a in f_aliases:
                    if a in doc_text:
                        return False
            return True

        # Generate Hop 1 questions
        hop1_selected = []
        for p in paths_1hop:
            if len(hop1_selected) >= self.questions_per_hop:
                break
            hop1_selected.append(p)

        for p in hop1_selected:
            q_text = make_question_text(p)
            target_ent = self.entities[p[-1]["target"]]
            self.questions.append({
                "id": f"Q_{q_id_counter:04d}",
                "group": "hop1",
                "hop_count": 1,
                "question": q_text,
                "gold_answer": target_ent["canonical_name"],
                "gold_answer_aliases": target_ent["aliases"],
                "gold_supporting_doc_ids": [p[0]["supporting_doc_id"]],
                "gold_path": [p[0]["id"]],
            })
            q_id_counter += 1

        # Generate Hop 2 questions
        hop2_selected = []
        for p in paths_2hop:
            if len(hop2_selected) >= self.questions_per_hop:
                break
            if is_path_leak_free(p):
                hop2_selected.append(p)

        for p in hop2_selected:
            q_text = make_question_text(p)
            target_ent = self.entities[p[-1]["target"]]
            self.questions.append({
                "id": f"Q_{q_id_counter:04d}",
                "group": "hop2",
                "hop_count": 2,
                "question": q_text,
                "gold_answer": target_ent["canonical_name"],
                "gold_answer_aliases": target_ent["aliases"],
                "gold_supporting_doc_ids": [e["supporting_doc_id"] for e in p],
                "gold_path": [e["id"] for e in p],
            })
            q_id_counter += 1

        # Generate Hop 3 questions
        hop3_selected = []
        for p in paths_3hop:
            if len(hop3_selected) >= self.questions_per_hop:
                break
            if is_path_leak_free(p):
                hop3_selected.append(p)

        for p in hop3_selected:
            q_text = make_question_text(p)
            target_ent = self.entities[p[-1]["target"]]
            self.questions.append({
                "id": f"Q_{q_id_counter:04d}",
                "group": "hop3",
                "hop_count": 3,
                "question": q_text,
                "gold_answer": target_ent["canonical_name"],
                "gold_answer_aliases": target_ent["aliases"],
                "gold_supporting_doc_ids": [e["supporting_doc_id"] for e in p],
                "gold_path": [e["id"] for e in p],
            })
            q_id_counter += 1

        # Generate Alias questions (1-hop or 2-hop using first entity's alias)
        alias_candidates = [p for p in (paths_1hop + paths_2hop) if is_path_leak_free(p)]
        self.rng.shuffle(alias_candidates)
        alias_selected = []
        for p in alias_candidates:
            if len(alias_selected) >= self.n_alias:
                break
            first_ent = self.entities[p[0]["source"]]
            if first_ent["aliases"]:
                alias_selected.append(p)

        for p in alias_selected:
            q_text = make_question_text(p, use_alias=True)
            target_ent = self.entities[p[-1]["target"]]
            self.questions.append({
                "id": f"Q_{q_id_counter:04d}",
                "group": "alias",
                "hop_count": len(p),
                "question": q_text,
                "gold_answer": target_ent["canonical_name"],
                "gold_answer_aliases": target_ent["aliases"],
                "gold_supporting_doc_ids": [e["supporting_doc_id"] for e in p],
                "gold_path": [e["id"] for e in p],
            })
            q_id_counter += 1

        # Generate Unanswerable questions (ask about non-existent relations)
        unans_questions = []
        all_entity_ids = list(self.entities.keys())
        self.rng.shuffle(all_entity_ids)

        for eid in all_entity_ids:
            if len(unans_questions) >= self.n_unanswerable:
                break
            ent = self.entities[eid]
            cname = ent["canonical_name"]
            etype = ent["type"]
            out_rels = [e["relation"] for e in adj.get(eid, [])]

            # Ask about a relation this entity does NOT have
            if etype == "EMPLOYEE" and "DEVELOPS" not in out_rels:
                q_text = f"What product is directly developed by {cname}?"
            elif etype == "DEPARTMENT" and "SERVES" not in out_rels:
                q_text = f"Which commercial client is directly served by {cname}?"
            elif etype == "PRODUCT" and "BUILT_WITH" not in out_rels:
                q_text = f"What technology framework is {cname} built with?"
            elif etype in ("TECHNOLOGY", "CLIENT"):
                q_text = f"What product does {cname} develop or manage?"
            else:
                q_text = f"What client organization is managed by {cname}?"

            unans_questions.append({
                "id": f"Q_{q_id_counter:04d}",
                "group": "unanswerable",
                "hop_count": 0,
                "question": q_text,
                "gold_answer": None,
                "gold_answer_aliases": [],
                "gold_supporting_doc_ids": [],
                "gold_path": [],
            })
            q_id_counter += 1

        self.questions.extend(unans_questions)

    def save_all(self, out_dir: Path) -> None:
        """Write gold_kg.json, documents, and questions.jsonl."""
        out_dir.mkdir(parents=True, exist_ok=True)
        docs_dir = out_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save documents
        for doc_id, doc in self.documents.items():
            doc_file = docs_dir / f"{doc_id}.txt"
            doc_file.write_text(doc["text"] + "\n", encoding="utf-8")

        # 2. Save gold_kg.json
        kg_path = out_dir / "gold_kg.json"
        kg_data = {
            "seed": self.seed,
            "entities": list(self.entities.values()),
            "edges": self.edges,
            "stats": {
                "num_entities": len(self.entities),
                "num_edges": len(self.edges),
                "num_documents": len(self.documents),
                "num_questions": len(self.questions),
            },
        }
        with open(kg_path, "w", encoding="utf-8") as f:
            json.dump(kg_data, f, indent=2, ensure_ascii=False)

        # 3. Save questions.jsonl
        q_path = out_dir / "questions.jsonl"
        with open(q_path, "w", encoding="utf-8") as f:
            for q in self.questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        print(f"Generated successfully into {out_dir}:")
        print(f"  Entities:  {len(self.entities)}")
        print(f"  Edges:     {len(self.edges)}")
        print(f"  Documents: {len(self.documents)} in {docs_dir}")
        print(f"  Questions: {len(self.questions)} in {q_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic KG and Corpus Generator")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic generation")
    parser.add_argument("--n_docs", type=int, default=60, help="Target number of documents (subject entities)")
    parser.add_argument("--out", type=Path, default=Path("experiments/synth_kg/corpus_S"), help="Output directory")
    parser.add_argument("--alias_rate", type=float, default=0.3, help="Probability of using alias in document text")
    parser.add_argument("--pronoun_rate", type=float, default=0.2, help="Probability of leading pronoun in sentence")
    parser.add_argument("--distractor_rate", type=float, default=0.3, help="Probability of distractor entity in doc")
    parser.add_argument("--questions_per_hop", type=int, default=30, help="Questions per hop count (1, 2, 3)")
    parser.add_argument("--n_unanswerable", type=int, default=15, help="Number of unanswerable questions")
    parser.add_argument("--n_alias", type=int, default=15, help="Number of alias questions")
    args = parser.parse_args()

    gen = SyntheticCorpusGenerator(
        seed=args.seed,
        n_docs=args.n_docs,
        alias_rate=args.alias_rate,
        pronoun_rate=args.pronoun_rate,
        distractor_rate=args.distractor_rate,
        questions_per_hop=args.questions_per_hop,
        n_unanswerable=args.n_unanswerable,
        n_alias=args.n_alias,
    )
    gen.build_kg()
    gen.generate_documents()
    gen.generate_questions()
    gen.save_all(args.out)


if __name__ == "__main__":
    main()
