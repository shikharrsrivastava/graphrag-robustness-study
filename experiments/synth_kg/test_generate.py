"""Unit tests for synthetic procedural KG and corpus generator."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import shutil

from experiments.synth_kg.generate import SyntheticCorpusGenerator, REAL_WORLD_NAME_BLACKLIST
from experiments.synth_kg.validate import validate_corpus


@pytest.fixture(scope="module")
def sample_corpus(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("test_corpus")
    gen = SyntheticCorpusGenerator(
        seed=7,
        n_docs=60,
        alias_rate=0.3,
        pronoun_rate=0.2,
        distractor_rate=0.3,
        questions_per_hop=10,
        n_unanswerable=5,
        n_alias=5,
    )
    gen.build_kg()
    gen.generate_documents()
    gen.generate_questions()
    gen.save_all(out_dir)
    return out_dir


def test_determinism(tmp_path: Path) -> None:
    """Same seed must produce bit-for-bit identical gold_kg.json and questions.jsonl."""
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    gen1 = SyntheticCorpusGenerator(seed=42, n_docs=30, questions_per_hop=5, n_unanswerable=3, n_alias=3)
    gen1.build_kg()
    gen1.generate_documents()
    gen1.generate_questions()
    gen1.save_all(dir1)

    gen2 = SyntheticCorpusGenerator(seed=42, n_docs=30, questions_per_hop=5, n_unanswerable=3, n_alias=3)
    gen2.build_kg()
    gen2.generate_documents()
    gen2.generate_questions()
    gen2.save_all(dir2)

    kg1 = (dir1 / "gold_kg.json").read_text(encoding="utf-8")
    kg2 = (dir2 / "gold_kg.json").read_text(encoding="utf-8")
    assert kg1 == kg2, "gold_kg.json is not deterministic!"

    q1 = (dir1 / "questions.jsonl").read_text(encoding="utf-8")
    q2 = (dir2 / "questions.jsonl").read_text(encoding="utf-8")
    assert q1 == q2, "questions.jsonl is not deterministic!"

    docs1 = sorted([f.name for f in (dir1 / "docs").iterdir()])
    docs2 = sorted([f.name for f in (dir2 / "docs").iterdir()])
    assert docs1 == docs2
    for dname in docs1:
        assert (dir1 / "docs" / dname).read_text(encoding="utf-8") == (dir2 / "docs" / dname).read_text(encoding="utf-8")


def test_validation_suite(sample_corpus: Path) -> None:
    """Run full validation suite on sample corpus."""
    res = validate_corpus(sample_corpus)
    assert res["valid"], f"Validation failed: {res['errors']}"
    assert res["num_documents"] == 60
    assert res["num_questions"] == 40  # 10*3 + 5 + 5


def test_verbatim_evidence_quotes(sample_corpus: Path) -> None:
    """Every single evidence quote in gold_kg.json must exist verbatim in its document."""
    with open(sample_corpus / "gold_kg.json") as f:
        kg = json.load(f)
    for edge in kg["edges"]:
        doc_id = edge["supporting_doc_id"]
        doc_path = sample_corpus / "docs" / f"{doc_id}.txt"
        assert doc_path.exists(), f"Doc {doc_id} not found"
        doc_text = doc_path.read_text(encoding="utf-8")
        assert edge["evidence_quote"] in doc_text, f"Quote '{edge['evidence_quote']}' missing in {doc_id}"


def test_entity_names_invented_and_unique(sample_corpus: Path) -> None:
    """Entity names must be unique and contain no blacklisted real-world names."""
    with open(sample_corpus / "gold_kg.json") as f:
        kg = json.load(f)
    seen = set()
    for ent in kg["entities"]:
        cname = ent["canonical_name"]
        assert cname not in seen, f"Duplicate entity name: {cname}"
        seen.add(cname)
        for bad in REAL_WORLD_NAME_BLACKLIST:
            assert bad not in cname.lower().split(), f"Found real-world name {bad} in {cname}"


def test_bridge_document_non_leakage(sample_corpus: Path) -> None:
    """Bridge documents for multi-hop questions must not leak the first entity's name."""
    with open(sample_corpus / "gold_kg.json") as f:
        kg = json.load(f)
    entities = {e["id"]: e for e in kg["entities"]}

    with open(sample_corpus / "questions.jsonl") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    for q in questions:
        if q["hop_count"] < 2:
            continue
        first_doc_id = q["gold_supporting_doc_ids"][0]
        # First doc subject is the start entity
        first_doc_text = (sample_corpus / "docs" / f"{first_doc_id}.txt").read_text(encoding="utf-8")
        
        # Identify first entity canonical name
        start_ent = None
        for ent in entities.values():
            if ent["doc_id"] == first_doc_id:
                start_ent = ent
                break
        assert start_ent is not None

        start_name = start_ent["canonical_name"].lower()
        start_aliases = [a.lower() for a in start_ent["aliases"]]

        for b_doc_id in q["gold_supporting_doc_ids"][1:]:
            b_text = (sample_corpus / "docs" / f"{b_doc_id}.txt").read_text(encoding="utf-8").lower()
            assert start_name not in b_text, f"Question {q['id']} leaked {start_name} into {b_doc_id}"
            for a in start_aliases:
                assert a not in b_text, f"Question {q['id']} leaked alias {a} into {b_doc_id}"
