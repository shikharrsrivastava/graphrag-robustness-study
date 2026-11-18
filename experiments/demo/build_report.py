#!/usr/bin/env python3
"""Builds docs/PROJECT_REPORT.md. All tables and measured numbers are copied from
experiments/synth_kg_fix/FINAL_REPORT.md and diagnosis/DIAGNOSIS.md by this script; the prose carries no numbers."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "experiments/synth_kg_fix"
F = (FIX / "FINAL_REPORT.md").read_text(encoding="utf-8").splitlines()
D = (FIX / "diagnosis/DIAGNOSIS.md").read_text(encoding="utf-8").splitlines()
README = (ROOT / "experiments/demo/README.md").read_text(encoding="utf-8")


def idx(lines, marker, start=0):
    return next(k for k in range(start, len(lines)) if marker in lines[k])


def table(lines, marker, start=0):
    i = idx(lines, marker, start)
    while not lines[i].startswith("|"):
        i += 1
    j = i
    while j < len(lines) and lines[j].startswith("|"):
        j += 1
    return "\n".join(lines[i:j])


def line(lines, marker):
    return next(l.strip() for l in lines if marker in l)


def sentence(lines, marker):
    text = line(lines, marker)
    for s in re.split(r"(?<=[.])\s+(?=[A-Z*(])", text):
        if marker in s:
            return s.strip()
    return text


fate = table(F, "Gold-edge fate (95 gold edges")
recall7 = table(F, "Retrieval-only all-gold evidence recall at K=4")
ent7 = line(F, "Entities: canonical 97")
alias7 = line(F, "Alias seeds resolved by the graph route")
paired7 = line(F, "Paired changes before -> all fixes")
reading7 = line(F, "Reading: the alias improvement")
recall11 = table(F, "## Result: all-gold evidence recall at K=4")
prereg = table(F, "## Pre-registered primary outcomes")
desc11 = [l for l in F[idx(F, "## Descriptive (no decision)"):idx(F, "## Skipped / failed", idx(F, "## Descriptive (no decision)"))] if l.startswith("- ")]
hashes = "\n".join(F[idx(F, "### Hashes") + 1: idx(F, "### Hashes") + 18])
tbl11 = table(D, "### Table 1.1 All-gold recall by candidate list")
ptr = idx(D, "## Part 1.")
p1 = line(D, "Three separate effects")
_s = sentence(D, "the extractor assigns the right relation type")
p2a = "The " + _s[_s.index("extractor assigns"):].replace("**", "", 1)
p2b = sentence(D, "Over both seeds 23 of 30")
l4a = line(D, "triples with final status REJECTED_ONTOLOGY: 17")
l4b = line(D, "exact validated gold edges (A+E) before: 47")
p0 = line(D, "Seed 11 has now been used for diagnosis")
repro = re.search(r"```bash\n(.*?)```", README, re.S).group(1).strip()
corpus = line(F, "Corpus: `generate.py --seed 11")
ingest = line(F, "Ingestion: single attempt")
bl = "\n".join("> " + l for l in desc11)

md = f"""# Hybrid GraphRAG: project report

Status as of 2026-11-19, branch `hybrid-graphrag-build`. All tables in section 4, and every measured number in sections 5 and 6, are copied by `experiments/demo/build_report.py` from `experiments/synth_kg_fix/FINAL_REPORT.md` and `experiments/synth_kg_fix/diagnosis/DIAGNOSIS.md`. The prose in this file contains no hand-typed results.

## 1. Goal and scope
The project builds a modular, domain-agnostic Hybrid GraphRAG engine that runs locally (gemma3:4b and embeddinggemma through Ollama, a MacBook Air M2 with 16 GB RAM, no GPU) and evaluates it with scientific discipline. The question the evaluation asks is specific: does adding a knowledge graph to dense retrieval improve multi-hop retrieval, and does it do so beyond what simple graph-free baselines achieve?

In scope: ingestion, entity and relationship extraction, entity resolution, graph construction and validation, communities, hybrid retrieval, cited answers or abstention, and a retrieval-only evaluation harness. Out of scope: large-scale corpora, cloud models, and answer-quality evaluation of the final fixes (only retrieval was evaluated after the fixes).

## 2. Architecture
**Ingestion pipeline.** load -> chunk -> entity extraction -> relationship extraction (validated against an ontology) -> entity resolution -> graph build and validation (python-igraph) -> chunk embeddings (ChromaDB) -> Leiden communities with generated reports.

**Query pipeline.** An LLM planner produces intent, seed entities, a target type and a relationship path. Seeds are resolved against canonical entities. Three routes produce candidate chunks: graph (traversal from the seeds), vector (dense top-k) and community (reports). Candidates are merged by chunk ID, fused with reciprocal rank fusion (k = 60), cut to the context size, and passed to the answer generator, which returns a cited answer or abstains.

**Query modes** (`HybridGraphRAG.query(question, mode=...)`): `hybrid`, `vector_only`, `graph_only`, `community_aware`.

**Public API** (`hybrid_graphrag.HybridGraphRAG`): `from_config`, `from_existing`, `ingest`, `query`, `get_kb_stats`, `close`.

**Modules** (`src/`): `ingestion`, `extraction` (entity and relation extractors, schemas), `resolution` (entity resolver), `graph` (builder, validator, traversal, clustering), `embeddings`, `communities`, `retrieval` (planner, seed retriever, context builder), `generation` (answer generator), `storage`, `core` (configuration, models, ontology, interfaces), and `engine.py` (orchestration). Entity and relation types come from an ontology config (`configs/domains/`), and new behaviour is added behind config flags with unchanged defaults.

## 3. Evaluation method
- **Benchmark.** HotpotQA was tried first and abandoned for this setup: ingestion was too slow and most extracted relationships were rejected. The project moved to a procedural synthetic benchmark (`experiments/synth_kg/`): a company ontology, 60 documents, a gold knowledge graph, and 120 questions in groups hop1, hop2, hop3, alias and unanswerable.
- **Seeds.** Seed 7 is the development corpus. Seed 11 is a fresh corpus used once as a pre-registered confirmation; it was later consumed by the diagnosis (section 6), so it cannot confirm anything further.
- **Metric.** Retrieval-only, document-level all-gold recall at K = 4 (every gold supporting document is among the 4 context documents). Query plans are cached, answer generation is stubbed, and the LLM client raises on any call during evaluation.
- **Hygiene rules.** ERROR is counted separately from abstentions; temperature 0; equal limits across modes; Wilson 95% intervals; exact sign tests on paired discordant questions; Holm correction across families of tests; decision rules written before the run; no tuning after seeing results; scripts that produced reported numbers are not edited afterwards.
- **Baselines.** Graph-free baselines were defined to test whether the graph itself matters: B1 dense top-4, B7 iterative dense, B9 dense top-1 seed with capitalized-name overlap expansion, B10 two-step.
- **Integrity.** The `git diff` hash was frozen before the seed-11 run and checked after (hashes below).

```
{hashes}
```

## 4. Results (pasted by script)
### 4.1 Seed 7 (development; retrieval-only, K = 4; before vs after the fixes)
{recall7}

Gold-edge fate on seed 7 (95 gold edges):

{fate}

### 4.2 Seed 7, candidate lists (source: DIAGNOSIS.md, Table 1.1, seed 7 `kb_dev` only)
{tbl11}

### 4.3 Seed 11 (fresh corpus, pre-registered)
{corpus}

{ingest}

{recall11}

Pre-registered primary outcomes (paired exact sign test, Holm over 3):

{prereg}

Descriptive findings reported with those results:

{bl}

## 5. Defects found and fixed, with measured effect
Four defects were verified in the engine and fixed behind config flags (defaults unchanged). The fixes were evaluated as bundles, so per-fix attribution is limited to what the runs below separate. The 'merge+initial only' run is identical to 'all fixes' except that post-resolution revalidation is off.

| defect / fix | what changed | measured effect (source lines below) |
|---|---|---|
| Hardcoded `"PERSON"` literals in the resolver and seed retriever | `person_like_types` is read from the ontology (company ontology: EMPLOYEE); literals replaced in `entity_resolver.py`, `seed_retriever.py`, `relation_extractor.py` | enables surname and initial matching for non-PERSON ontologies; not separable from the merge fix in the runs |
| Initial+surname matching | `person_names_compatible` accepts initial+surname; merge of such mentions behind `enable_initial_surname_person_merge` | no measurable effect: no document in these corpora contains an initial+surname form (FINAL_REPORT section 6, item 2) |
| Entities split by inconsistent mention types | `merge_exact_name_across_types` with a type vote and ontology-order tie-break | see the entity and fate lines below (gold entities split, edge fates, alias seeds) |
| Relationship ontology validation ran on mention types before resolution | `revalidate_relationships_after_resolution` recomputes validity with canonical types | trade-off, see 'Revalidation' below |
| Provenance marker text leaks into answers | `graph_context_mode: "off"` omits the graph path block | not evaluated: no answer generation was run after this change |

Measured on seed 7 (copied from FINAL_REPORT.md section 2b):

- {ent7}
- {alias7}
- {paired7}
- {reading7}

Revalidation: compare the 'merge+initial only' and 'all fixes' rows in the table in 4.1. The report states the trade-off explicitly: revalidation removes wrong-type edges and also removes some traversal links, so graph_only hop3 is lower with it. Seed 11 shows the same mechanism on a smaller number of changed triples (see the descriptive findings in 4.3).

Prompt v1.1 (a multi-relation instruction, defect 4 in the original list) failed its pre-written dev rule and stays disabled; it was evaluated on a subset of seed-7 documents only.

## 6. Findings
1. **Negative result against graph-free baselines.** On the fresh seed, the pre-registered decisions were that a graph-free baseline matches or exceeds hybrid on hop2 (P1) and hop3 (P2), and inconclusive on alias (P3); see the table in 4.3. Hybrid does beat plain dense retrieval on hop2, but a name-overlap baseline beats hybrid. **Graph-specific benefit is not demonstrated on this benchmark.** The benchmark's procedurally unique names favour name-overlap baselines, which limits how far this generalizes in either direction.
2. **Hop3 fails mainly because of extraction coverage, which is the ceiling.** {p1}
3. **Reversed edge direction in DEVELOPS extraction.** {p2a} {p2b} The ontology correctly rejects the reversed pair, so this is the extractor not following the stated direction, not an ontology defect.
4. **Ontology-guided direction repair, computed on stored triples (exploratory; the seed was already used, so it is not evidence of a fix):**

> {l4a}
> {l4b}

5. **Protocol consequence.** {p0}

## 7. Limitations
See `docs/KNOWN_LIMITATIONS.md` (general design limits and the synthetic-KG audit findings in its last section) and FINAL_REPORT.md section 7 for the evaluation-specific limits. In short: one ontology; two seeds with wide intervals; procedurally unique names; flags chosen on development data; retrieval metric only; the revalidation trade-off; the PRODUCT/TECHNOLOGY type-vote weakness; no coreference step; partial-name aliases unresolved; a remaining hardcoded `"PROJECT"` literal in `src/resolution/entity_resolver.py`.

## 8. Next steps
1. **Direction repair behind a flag** (default off): swap source and target of an ontology-rejected triple when the swapped pair satisfies the same relation's rule.
2. **Confirm on fresh seed 13**, generated with the same knobs as seeds 7 and 11, with the protocol and decision rules written first, the diff hash frozen, one ingestion, retrieval-only evaluation against B9 and B10 using the same statistics as above. No tuning after seeing seed 13.
3. Open and not yet investigated: why hybrid hop3 is far below graph_only (RRF crowding and K mismatch were measured in the diagnosis, no fix applied); the type-vote tie-break for PRODUCT versus TECHNOLOGY; partial-name aliases; coreference.
4. Housekeeping: reusability across ontologies (open-domain config), documentation and release.

## Reproduce
Demo on the seed-7 fixed knowledge base (commands from `experiments/demo/README.md`):

```bash
{repro}
```
"""
(ROOT / "docs/PROJECT_REPORT.md").write_text(md, encoding="utf-8")
print("wrote docs/PROJECT_REPORT.md", len(md.splitlines()), "lines")
