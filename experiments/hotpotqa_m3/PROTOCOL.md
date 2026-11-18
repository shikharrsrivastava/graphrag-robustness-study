# Milestone 3 Evaluation Protocol: Generator Isolation on HotpotQA Dev Set

## 1. Objective and Scope
The goal of Milestone 3 is to evaluate the factual answering, grounding, and reasoning capabilities of the hybrid GraphRAG engine's generation component in isolation from its knowledge graph traversal and full-corpus indexing pipelines.

Specifically, this benchmark measures how well the generator answers HotpotQA multi-hop questions when provided with:
1. **Gold Context Ceiling (Condition A_v11 & A_v10)**: Only the 2 ground-truth supporting paragraphs.
2. **Distractor Sensitivity (Condition B)**: All 10 paragraphs provided in the official HotpotQA distractor dataset (2 gold + 8 distractors).
3. **Dense Retrieval Degradation (Condition C)**: The top-3 paragraphs selected by dense cosine retrieval over the 10 candidate paragraphs using the engine's embedding model (`embeddinggemma`).

This test strictly isolates generator competence and distractor robustness from graph construction and entity extraction. No knowledge graph is constructed for this milestone.

---

## 2. Dataset Selection and Exclusion Rules

### 2.1 Source Dataset
The evaluation corpus is the official validation split of the HotpotQA distractor dataset (`hotpot_dev_distractor_v1.json`, 7,405 questions, 51.22 MB), retrieved from Hugging Face (`hotpotqa/hotpot_qa`, distractor configuration, validation split).

### 2.2 Exclusion Set
To guarantee zero benchmark contamination, every HotpotQA question ID or question text that appears anywhere in this repository (prior tests, pilot evaluations in `knowledge_bases/hotpotqa_pilot`, documentation, or test fixtures) was identified and excluded. Exactly **20 questions** were identified and recorded in `experiments/hotpotqa_m3/exclusion_set.json`.

### 2.3 Sampling Protocol
From the 7,385 uncontaminated dev questions:
- Bridge candidates: 5,904 questions
- Comparison candidates: 1,481 questions

Using a fixed random seed `42` with Python's standard `random.sample`, exactly **30 questions** were drawn:
- 15 `bridge` questions
- 15 `comparison` questions

The selection was performed solely on question metadata (`_id`, `type`, `question`) before inspecting answers or paragraphs, and before executing any model calls. The selected questions are saved to `experiments/hotpotqa_m3/selected_questions.json`.

---

## 3. Experimental Conditions

All conditions execute through the engine's own answer-generation entry point:
- **Component**: `src.generation.answer_generator.OllamaAnswerGenerator`
- **Method**: `generate_structured(retrieval_result)`
- **Configuration**:
  - LLM Model: `gemma3:4b` via Ollama (`http://localhost:11434`)
  - Temperature: `0.0`
  - `comparative_reasoning_mode`: `"off"` (production default)
- **Context Chunk Formatting**:
  Each HotpotQA paragraph is represented as an `Evidence` chunk with ID `<question_id>_P<k>` and text formatted identically to the engine's `ContextBuilder`:
  ```text
  Title: <paragraph_title>
  <sentence_1> <sentence_2> ...
  ```

### Evaluated Conditions:
1. **Condition A_v11 (Gold Context Ceiling, Prompt v1.1)**:
   - Context: Exactly the 2 gold supporting paragraphs identified by `supporting_facts`.
   - Prompt Version: `v1.1` (`prompts/answer_generation_v1_1.txt`).
2. **Condition A_v10 (Gold Context Ceiling, Prompt v1.0)**:
   - Context: Exactly the 2 gold supporting paragraphs.
   - Prompt Version: `v1.0` (`prompts/answer_generation_v1_0.txt`).
3. **Condition B (Full Distractor Context, Prompt v1.1)**:
   - Context: All 10 paragraphs (2 gold + 8 distractors) in original dataset order.
   - Prompt Version: `v1.1`.
4. **Condition C (Dense Retrieval Context, Prompt v1.1)**:
   - Retrieval: Dense embedding of the question and all 10 candidate paragraphs using `embeddinggemma:latest` via `src.core.interfaces.EmbeddingProvider` / `OllamaEmbeddingService`.
   - Selection: Top-3 paragraphs by cosine similarity.
   - Context: The top-3 retrieved paragraphs in ranking order.
   - Prompt Version: `v1.1`.

---

## 4. Metrics and Evaluation Criteria

Metrics are implemented in `experiments/hotpotqa_m3/score.py` reading persisted JSONL run outputs.

### 4.1 Normalization
Standard HotpotQA answer normalization:
1. Lowercase text.
2. Remove punctuation characters (`!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`).
3. Remove English articles (`a`, `an`, `the`) as whole words.
4. Collapse consecutive whitespace.

### 4.2 Accuracy Metrics
- **Exact Match (EM)**: `normalize(prediction) == normalize(gold_answer)`.
- **Token F1**: Token-level harmonic mean of precision and recall between normalized prediction and normalized gold answer.
- **Contains Accuracy (Primary Metric)**:
  - For standard questions: `normalize(gold_answer)` appears as a whole token sequence in `normalize(prediction)`, AND `abstained == False`, AND `generation_error == False`.
  - For Boolean questions (`yes` / `no` gold): The first normalized affirmative/negative token in the predicted text matches the gold answer, AND `abstained == False`, AND `generation_error == False`.
  - Full-sentence generations naturally contain the direct answer phrase; "contains" serves as the primary headline metric, reported alongside EM and F1.

### 4.3 Diagnostic Counters
- **ABSTAINED**: Total count where `abstained == True` on answerable questions.
- **ERROR**: Total count where `generation_error == True` (e.g. LLM timeout, malformed JSON).
- **Abstained-but-Answer-Present**: Count where `abstained == True`, but the generated answer text actually contained the gold answer.

### 4.4 Condition C Retrieval Metrics
- **Gold Paragraph Recall @ Top 3**:
  - `Both (2/2)`: Both gold paragraphs in the top 3.
  - `One (1/2)`: Exactly one gold paragraph in the top 3.
  - `None (0/2)`: Neither gold paragraph in the top 3.
- Accuracy broken down by gold paragraph recall bucket.

### 4.5 Statistical Confidence
Wilson 95% score confidence intervals are computed for all accuracy proportions:
$$CI = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
where $z = 1.96$.

All metrics are reported overall ($n=30$) and split by question type (Bridge $n=15$, Comparison $n=15$).

---

## 5. Non-Interference Commitment
- Zero code modifications to `src/`, `hybrid_graphrag/`, `configs/`, `prompts/`, or `tests/`.
- No post-hoc tuning of prompts, parameters, thresholds, question lists, or scoring criteria after seeing run outputs.
- Temperature is fixed at `0.0`.
- All outputs are stored exclusively in `experiments/hotpotqa_m3/`.
