# Milestone 4 Final Report: Pooled Dense Retrieval & Ingestion Benchmark

## 1. Git State
- **Branch**: `hybrid-graphrag-build`
- **HEAD Commit**: `3877a4d02be92ca499fbb001e3a7afad971d4a4a` (`data: synthetic academic evaluation run records`)
- **Working Tree State**:
  - Tracked files: Completely clean (zero changes to `src/`, `hybrid_graphrag/`, `configs/`, `prompts/`, `tests/`, or `knowledge_bases/`).
  - Untracked directories:
    - `experiments/hotpotqa_m3/` (read-only)
    - `experiments/hotpotqa_m4/` (active milestone directory)

---

## 2. Part 1: Adjudication Dump from M3 Run Files

### 2.1 Part 1(a): Questions Non-Correct (FAIL, ABS, or ERR) in >=1 Condition
*Total non-correct in >=1 condition*: **15 / 30 questions**.

#### 1. `5ae5669755429960a22e02ec` (`bridge`)
- **Question**: Which of the campaign that brought out the term Vichy Republican on social media was formally launched on June 16, 2015, at Trump Tower in New York City?
- **Gold Answer**: `presidential campaign of Donald Trump`
- **Non-correct Conditions**:
  - **A_v11** | Status: `FAIL` | Canonical: `"Donald Trump's 2016 presidential campaign"` | Abs: `False`, Err: `False`
    - *Full Answer*: The evidence indicates that the campaign that brought out the term Vichy Republican was Donald Trump's 2016 presidential campaign, which was formally launched on June 16, 2015, at Trump Tower in New York City [D000_C0000]. Chunk [D001_C0000] defines Vichy Republican as a term that emerged on social media in 2016 in regards to Donald Trump's candida...
  - **A_v10** | Status: `FAIL` | Canonical: `"Donald Trump's 2016 presidential campaign"` | Abs: `False`, Err: `False`
    - *Full Answer*: The Donald Trump’s 2016 presidential campaign was formally launched on June 16, 2015, at Trump Tower in New York City [D002_C0000]. The term Vichy Republican emerged on social media in 2016 in regards to Trump’s candidacy [D001_C0000].
  - **B** | Status: `FAIL` | Canonical: `'Donald Trump campaign'` | Abs: `False`, Err: `False`
    - *Full Answer*: The Donald Trump campaign was formally launched on June 16, 2015, at Trump Tower in New York City. The term "Vichy Republican" emerged on social media in 2016 to describe supporters of Donald Trump within the Republican Party [D003_C0000, D009_C0000].
  - **C** | Status: `FAIL` | Canonical: `'Donald Trump'` | Abs: `False`, Err: `False`
    - *Full Answer*: The Donald Trump presidential campaign was formally launched on June 16, 2015, at Trump Tower in New York City [D001_C0000]. George Pataki's campaign was launched on May 28, 2015 [D002_C0000].

#### 2. `5adf0cbd5542993a75d263e0` (`bridge`)
- **Question**: When was the English-born South Australian brewer whose brewery was amalgamated with W. K. Simms's West End Brewery. born
- **Gold Answer**: `6 April 1830`
- **Non-correct Conditions**:
  - **B** | Status: `FAIL` | Canonical: `'1820'` | Abs: `False`, Err: `False`
    - *Full Answer*: Edward Fitzgerald was born in 1820, as stated in chunk [5adf0cbd5542993a75d263e0_P0]. The chunk describes him as an Australian brewer and founder of the Castlemaine Brewery, and provides his birth year.
  - **C** | Status: `FAIL` | Canonical: `'1830'` | Abs: `False`, Err: `False`
    - *Full Answer*: The English-born South Australian brewer whose brewery was amalgamated with W. K. Simms's West End Brewery was Sir Edwin Thomas Smith, who was born on April 6, 1830 [D001_C0000].

#### 3. `5a7caae75542996dd594b971` (`bridge`)
- **Question**: What, known as AAS, is commonly used in bodybuilding?
- **Gold Answer**: `Anabolic steroids`
- **Non-correct Conditions**:
  - **A_v10** | Status: `FAIL` | Canonical: `'Anabolic/androgenic steroids'` | Abs: `False`, Err: `False`
    - *Full Answer*: The context states that anabolic/androgenic steroids are used for muscle-building and strength gain for cosmetic reasons as well as for performance-enhancement in athletics and bodybuilding [D001_P1].

#### 4. `5abbcba05542992ccd8e7fa3` (`bridge`)
- **Question**: With whom did Lever Brothers merge to create an Indonesian soap factory?
- **Gold Answer**: `Margarine Unie`
- **Non-correct Conditions**:
  - **A_v11** | Status: `FAIL` | Canonical: `'Lever Brothers'` | Abs: `False`, Err: `False`
    - *Full Answer*: The evidence states that PT Unilever Indonesia Tbk was established as Lever Zeepfabrieken N.V., a subsidiary of Lever Brothers [D001_C0000]. Therefore, Lever Brothers merged to create the Indonesian soap factory.
  - **A_v10** | Status: `FAIL` | Canonical: `'Lever Brothers'` | Abs: `False`, Err: `False`
    - *Full Answer*: According to Chunk [5abbcba05542992ccd8e7fa3_P1], PT Unilever Indonesia Tbk was established as a subsidiary of Lever Brothers. Therefore, Lever Brothers was with whom Lever Brothers merged to create an Indonesian soap factory.
  - **C** | Status: `FAIL` | Canonical: `'Lever Brothers'` | Abs: `False`, Err: `False`
    - *Full Answer*: The evidence indicates that PT Unilever Indonesia Tbk was initially established as a subsidiary of Lever Brothers [D001_C0000]. This establishes a connection between the two entities, but does not describe a merger to create an Indonesian soap factory.

#### 5. `5a77ddb855429967ab1052d8` (`bridge`)
- **Question**: The True Colors World Tour was by what artist whose debut album was "She's So Unusual"?
- **Gold Answer**: `Cynthia Ann Stephanie "Cyndi" Lauper`
- **Non-correct Conditions**:
  - **A_v11** | Status: `FAIL` | Canonical: `'Cyndi Lauper'` | Abs: `False`, Err: `False`
    - *Full Answer*: The question asks for the artist whose debut album was "She's So Unusual". According to [D000_C0000], this album was released by Cyndi Lauper. The True Colors World Tour was a concert tour by Cyndi Lauper.
  - **A_v10** | Status: `FAIL` | Canonical: `'Cyndi Lauper'` | Abs: `False`, Err: `False`
    - *Full Answer*: The True Colors World Tour was by Cyndi Lauper, whose debut album was "She's So Unusual" [D002_C0000]. This is confirmed by [D001_C0000] which states the tour supported "True Colors" and "She's So Unusual".
  - **B** | Status: `FAIL` | Canonical: `'Cyndi Lauper'` | Abs: `False`, Err: `False`
    - *Full Answer*: Cyndi Lauper released the album "She's So Unusual" [D000_C0000]. This album was her debut solo album and charted four top-five hits on the "Billboard" Hot 100, including "Girls Just Want to Have Fun", "Time After Time", "She Bop", and "All Through the Night".
  - **C** | Status: `FAIL` | Canonical: `'Cyndi Lauper'` | Abs: `False`, Err: `False`
    - *Full Answer*: The artist whose debut album was "She's So Unusual" is Cyndi Lauper. This is stated in [D002_C0000] which says "Her second record "True Colors" (1986) was Lauper's debut solo album "She's So Unusual" (1983)".

#### 6. `5a76067e55429976ec32bce9` (`bridge`)
- **Question**: What Box Codax group musician was also the guitarist, backing/lead vocalist and keyboardist of another Glasgow-based band?
- **Gold Answer**: `Nicholas John "Nick" McCarthy`
- **Non-correct Conditions**:
  - **A_v11** | Status: `FAIL` | Canonical: `'Nick McCarthy'` | Abs: `False`, Err: `False`
    - *Full Answer*: Nick McCarthy was a member of Box Codax [5a76067e55429976ec32bce9_P1] and also the guitarist, backing/lead vocalist, and keyboardist of Franz Ferdinand [5a76067e55429976ec32bce9_P1].
  - **A_v10** | Status: `FAIL` | Canonical: `'Nick McCarthy'` | Abs: `False`, Err: `False`
    - *Full Answer*: Nick McCarthy is a member of Box Codax [5a76067e55429976ec32bce9_P0] and was also the guitarist, backing/lead vocalist, and keyboardist of Franz Ferdinand [5a76067e55429976ec32bce9_P1], which is a Glasgow-based band.
  - **B** | Status: `FAIL` | Canonical: `'Nick McCarthy'` | Abs: `False`, Err: `False`
    - *Full Answer*: Nick McCarthy is identified as a member of both Box Codax [D000_C0000] and Franz Ferdinand [D001_C0000], fulfilling the criteria of being the guitarist, backing/lead vocalist and keyboardist of another Glasgow-based band. The evidence explicitly states this relationship.
  - **C** | Status: `FAIL` | Canonical: `'Nick McCarthy'` | Abs: `False`, Err: `False`
    - *Full Answer*: Nick McCarthy was a member of Box Codax [5a76067e55429976ec32bce9_P1] and also the guitarist, backing/lead vocalist and keyboardist of the Glasgow-based band Franz Ferdinand [5a76067e55429976ec32bce9_P1].

#### 7. `5ade0a275542997545bbbe2d` (`bridge`)
- **Question**: Where is the American hip hop group, that released the song "RedEye to Paris", from?
- **Gold Answer**: `Flatbush section of Brooklyn, New York City`
- **Non-correct Conditions**:
  - **C** | Status: `ABS` | Canonical: `''` | Abs: `True`, Err: `False`
    - *Full Answer*: The evidence does not state the origin of the group that released the song "RedEye to Paris". Chunk D001 describes Cypress Hill as being from South Gate, California [5ade0a275542997545bbbe2d_P1], and Chunk D002 describes Stetsasonic as being from Brooklyn, New York City [5ade0a275542997545bbbe2d_P2]. However, the song "RedEye to Paris" is by Flatbu...

#### 8. `5ade24e35542992fa25da6e6` (`bridge`)
- **Question**: Mamie Gummer played the role of Nancy Crozier on the TV series airing on what network?
- **Gold Answer**: `CBS`
- **Non-correct Conditions**:
  - **A_v11** | Status: `ERR` | Canonical: `''` | Abs: `True`, Err: `True`
    - *Full Answer*: Error generating answer: Failed to generate valid JSON matching AnswerGenerationOutput after 3 attempts. Last error: Expecting ',' delimiter: line 17 column 553 (char 1472)

#### 9. `5ae07b0155429945ae95935d` (`comparison`)
- **Question**: Which novelist was German, David Guterson or Thomas Mann?
- **Gold Answer**: `Paul Thomas Mann`
- **Non-correct Conditions**:
  - **A_v11** | Status: `FAIL` | Canonical: `'Thomas Mann'` | Abs: `False`, Err: `False`
    - *Full Answer*: Thomas Mann was a German novelist, as stated in [D001_C0000], while David Guterson is described as an American novelist in [D000_C0000].
  - **A_v10** | Status: `FAIL` | Canonical: `'Thomas Mann'` | Abs: `False`, Err: `False`
    - *Full Answer*: Thomas Mann was a German novelist, as stated in Chunk D001 [5ae07b0155429945ae95935d_P1], while David Guterson is described as an American novelist in Chunk D000 [5ae07b0155429945ae95935d_P0].
  - **C** | Status: `FAIL` | Canonical: `'Thomas Mann'` | Abs: `False`, Err: `False`
    - *Full Answer*: Thomas Mann was described as a German novelist in [D001_C0000], while David Guterson is described as an American novelist in [D000_C0000].

#### 10. `5a70f9335542994082a3e46a` (`comparison`)
- **Question**: Which magazine is published more in a year, Essence or Alt for Damerne?
- **Gold Answer**: `ALT for Damerne`
- **Non-correct Conditions**:
  - **A_v10** | Status: `ABS` | Canonical: `''` | Abs: `True`, Err: `False`
    - *Full Answer*: The context does not provide information about the number of copies published annually for either Essence or Alt for Damerne. Therefore, it is impossible to determine which magazine is published more in a year [D000_C0000, D001_C0000].

#### 11. `5a8a3dd55542995153361360` (`comparison`)
- **Question**: Did Nicholas Hytner and Maya Deren work in the same role in the film industry?
- **Gold Answer**: `no`
- **Non-correct Conditions**:
  - **A_v11** | Status: `ABS` | Canonical: `''` | Abs: `True`, Err: `False`
    - *Full Answer*: The evidence indicates that Nicholas Hytner was a theatre director, film director, and film producer, while Maya Deren was a filmmaker, choreographer, dancer, film theorist, poet, lecturer, writer, and photographer. There is no information in the provided evidence to suggest that they worked together or in the same role in the film industry. Theref...
  - **A_v10** | Status: `ABS` | Canonical: `''` | Abs: `True`, Err: `False`
    - *Full Answer*: The evidence does not state that Nicholas Hytner and Maya Deren worked in the same role in the film industry. [Chunk: 5a8a3dd55542995153361360_P0] describes Maya Deren as a filmmaker, while [Chunk: 5a8a3dd55542995153361360_P1] describes Nicholas Hytner as a theatre and film director. There is no overlap in their described roles.
  - **B** | Status: `ABS` | Canonical: `''` | Abs: `True`, Err: `False`
    - *Full Answer*: The provided evidence does not state that Nicholas Hytner and Maya Deren worked in the same role in the film industry. Chunk D005 describes Hytner as a theatre and film director, while Chunk D003 describes Deren as a filmmaker and experimental film artist. There is no mention of any collaboration or shared role between them. Therefore, I must absta...

#### 12. `5ae26905554299492dc91c17` (`comparison`)
- **Question**: Are Dennis Locorriere and Maja Ivarsson both lead singers?
- **Gold Answer**: `yes`
- **Non-correct Conditions**:
  - **A_v11** | Status: `FAIL` | Canonical: `'no'` | Abs: `False`, Err: `False`
    - *Full Answer*: Dennis Locorriere was the lead vocalist and guitarist of Dr. Hook & the Medicine Show [5ae26905554299492dc91c17_P1], while Maja Ivarsson is the lead vocalist of The Sounds [5ae26905554299492dc91c17_P0]. Therefore, they are not both lead singers.
  - **B** | Status: `FAIL` | Canonical: `'no'` | Abs: `False`, Err: `False`
    - *Full Answer*: Dennis Locorriere is the lead vocalist and guitarist of The Temptations [D003_C0000], while Maja Ivarsson is the lead vocalist of The Sounds [D001_C0000]. Therefore, they were not both lead singers.
  - **C** | Status: `FAIL` | Canonical: `'no'` | Abs: `False`, Err: `False`
    - *Full Answer*: Maja Ivarsson is stated to be the lead vocalist of The Sounds [5ae26905554299492dc91c17_P0], while Dennis Locorriere was the lead vocalist and guitarist of Dr. Hook & the Medicine Show [5ae26905554299492dc91c17_P1]. Therefore, they are not both lead singers.

#### 13. `5a7fe0b55542994857a76818` (`comparison`)
- **Question**: Are The Maine and Black both from Tempe, Arizona?
- **Gold Answer**: `no`
- **Non-correct Conditions**:
  - **C** | Status: `ABS` | Canonical: `''` | Abs: `True`, Err: `False`
    - *Full Answer*: The Maine is from Tempe, Arizona [D000_C0000], but Black is not mentioned in the provided evidence. Therefore, I must abstain.

#### 14. `5ab2b630554299545a2cfa17` (`comparison`)
- **Question**: Are the films Gasland from 2010 and Pumping Iron from 1977 both documentaries?
- **Gold Answer**: `yes`
- **Non-correct Conditions**:
  - **A_v11** | Status: `FAIL` | Canonical: `'no'` | Abs: `False`, Err: `False`
    - *Full Answer*: Pumping Iron is described as a "docudrama" [D000_C0000], while Gasland is explicitly identified as a "documentary" [D001_C0000]. These terms represent distinct film genres, therefore the films are not both documentaries.
  - **A_v10** | Status: `FAIL` | Canonical: `'no'` | Abs: `False`, Err: `False`
    - *Full Answer*: Gasland is described as a "2010 American documentary" [D001_C0000], while Pumping Iron is described as a "1977 docudrama" [D002_C0000]. A documentary presents factual information, whereas a docudrama is a fictionalized account based on real events. Therefore, they are not both documentaries.
  - **B** | Status: `FAIL` | Canonical: `'no'` | Abs: `False`, Err: `False`
    - *Full Answer*: ‘Pumping Iron’ (1977) is described as a "docudrama" which is a staged presentation of real events, while ‘Gasland’ (2010) is explicitly identified as a "documentary" film [D006_C0000, D008_C0000].
  - **C** | Status: `FAIL` | Canonical: `'no'` | Abs: `False`, Err: `False`
    - *Full Answer*: Gasland [D000_C0000] is described as a 2010 American documentary, while Pumping Iron [D001_C0000] is described as a 1977 docudrama. These descriptions indicate different documentary types.

#### 15. `5a8b82a05542995d1e6f13d0` (`comparison`)
- **Question**: Are University of Virginia and Massachusetts Institute of Technology in the same country?
- **Gold Answer**: `yes`
- **Non-correct Conditions**:
  - **A_v11** | Status: `FAIL` | Canonical: `'no'` | Abs: `False`, Err: `False`
    - *Full Answer*: The University of Virginia is located in Virginia [5a8b82a05542995d1e6f13d0_P0], while the Massachusetts Institute of Technology is located in Massachusetts [5a8b82a05542995d1e6f13d0_P1]. These are two different states, and therefore different countries.
  - **A_v10** | Status: `FAIL` | Canonical: `'no'` | Abs: `False`, Err: `False`
    - *Full Answer*: The University of Virginia is located in Virginia [5a8b82a05542995d1e6f13d0_P0], while the Massachusetts Institute of Technology is located in Massachusetts [5a8b82a05542995d1e6f13d0_P1]. These are different states, and therefore different countries.
  - **C** | Status: `FAIL` | Canonical: `'no'` | Abs: `False`, Err: `False`
    - *Full Answer*: University of Virginia is located in Virginia [D001_C0000], while Massachusetts Institute of Technology is located in Massachusetts [D002_C0000]. Therefore, they are not in the same country.

---

### 2.2 Part 1(b): Full Untrimmed Answer Texts for 4 Focal Questions in A_v11 and A_v10

#### 1. `5a8a3dd55542995153361360` (Type: `comparison`, Gold: `no`)
*Question*: Did Nicholas Hytner and Maya Deren work in the same role in the film industry?
- **Condition A_v11**:
  - *Status*: `ABS` | *Abstained*: `True` | *Error*: `False`
  - *Canonical*: `""`
  - *Full Answer*:
    > The evidence indicates that Nicholas Hytner was a theatre director, film director, and film producer, while Maya Deren was a filmmaker, choreographer, dancer, film theorist, poet, lecturer, writer, and photographer. There is no information in the provided evidence to suggest that they worked together or in the same role in the film industry. Therefore, we must abstain.
- **Condition A_v10**:
  - *Status*: `ABS` | *Abstained*: `True` | *Error*: `False`
  - *Canonical*: `""`
  - *Full Answer*:
    > The evidence does not state that Nicholas Hytner and Maya Deren worked in the same role in the film industry. [Chunk: 5a8a3dd55542995153361360_P0] describes Maya Deren as a filmmaker, while [Chunk: 5a8a3dd55542995153361360_P1] describes Nicholas Hytner as a theatre and film director. There is no overlap in their described roles.

#### 2. `5a8b82a05542995d1e6f13d0` (Type: `comparison`, Gold: `yes`)
*Question*: Are University of Virginia and Massachusetts Institute of Technology in the same country?
- **Condition A_v11**:
  - *Status*: `FAIL` | *Abstained*: `False` | *Error*: `False`
  - *Canonical*: `'no'`
  - *Full Answer*:
    > The University of Virginia is located in Virginia [5a8b82a05542995d1e6f13d0_P0], while the Massachusetts Institute of Technology is located in Massachusetts [5a8b82a05542995d1e6f13d0_P1]. These are two different states, and therefore different countries.
- **Condition A_v10**:
  - *Status*: `FAIL` | *Abstained*: `False` | *Error*: `False`
  - *Canonical*: `'no'`
  - *Full Answer*:
    > The University of Virginia is located in Virginia [5a8b82a05542995d1e6f13d0_P0], while the Massachusetts Institute of Technology is located in Massachusetts [5a8b82a05542995d1e6f13d0_P1]. These are different states, and therefore different countries.

#### 3. `5ae26905554299492dc91c17` (Type: `comparison`, Gold: `yes`)
*Question*: Are Dennis Locorriere and Maja Ivarsson both lead singers?
- **Condition A_v11**:
  - *Status*: `FAIL` | *Abstained*: `False` | *Error*: `False`
  - *Canonical*: `'no'`
  - *Full Answer*:
    > Dennis Locorriere was the lead vocalist and guitarist of Dr. Hook & the Medicine Show [5ae26905554299492dc91c17_P1], while Maja Ivarsson is the lead vocalist of The Sounds [5ae26905554299492dc91c17_P0]. Therefore, they are not both lead singers.
- **Condition A_v10**:
  - *Status*: `PASS` | *Abstained*: `False` | *Error*: `False`
  - *Canonical*: `'yes'`
  - *Full Answer*:
    > Dennis Locorriere is described as the "lead vocalist and guitarist" of Dr. Hook & the Medicine Show [D001_C0000], while Maja Ivarsson is described as the "lead vocalist" of The Sounds [D000_C0000]. Both roles are lead singer.

#### 4. `5ab2b630554299545a2cfa17` (Type: `comparison`, Gold: `yes`)
*Question*: Are the films Gasland from 2010 and Pumping Iron from 1977 both documentaries?
- **Condition A_v11**:
  - *Status*: `FAIL` | *Abstained*: `False` | *Error*: `False`
  - *Canonical*: `'no'`
  - *Full Answer*:
    > Pumping Iron is described as a "docudrama" [D000_C0000], while Gasland is explicitly identified as a "documentary" [D001_C0000]. These terms represent distinct film genres, therefore the films are not both documentaries.
- **Condition A_v10**:
  - *Status*: `FAIL` | *Abstained*: `False` | *Error*: `False`
  - *Canonical*: `'no'`
  - *Full Answer*:
    > Gasland is described as a "2010 American documentary" [D001_C0000], while Pumping Iron is described as a "1977 docudrama" [D002_C0000]. A documentary presents factual information, whereas a docudrama is a fictionalized account based on real events. Therefore, they are not both documentaries.

---

### 2.3 Part 1(c): Gold Answer Breakdown (Yes / No / Other) & Subset Accuracy
- **Gold Answer Counts (N=30)**:
  - Gold = `'yes'`: **4** questions
  - Gold = `'no'`: **2** questions
  - Gold = `'other'`: **24** questions

| Condition | Gold = 'yes' (4) | Gold = 'no' (2) | Other (24) | Overall (30) |
| :--- | :---: | :---: | :---: | :---: |
| **`A_v11`** | 1 / 4 (25.0%) | 1 / 2 (50.0%) | 18 / 24 (75.0%) | **20 / 30 (66.7%)** |
| **`A_v10`** | 2 / 4 (50.0%) | 1 / 2 (50.0%) | 17 / 24 (70.8%) | **20 / 30 (66.7%)** |
| **`B`** | 2 / 4 (50.0%) | 1 / 2 (50.0%) | 20 / 24 (83.3%) | **23 / 30 (76.7%)** |
| **`C`** | 1 / 4 (25.0%) | 1 / 2 (50.0%) | 17 / 24 (70.8%) | **19 / 30 (63.3%)** |

### 2.4 Part 1(d): Disk Usage and Git Status Check
- `du -sh experiments/hotpotqa_m3/data`: **51M**
- `git status -s`:
  ```text
  ?? experiments/hotpotqa_m3/
  ?? experiments/hotpotqa_m4/
  ```

---

## 3. Part 2: Pooled Dense Retrieval Benchmark (Embeddinggemma)

- **Corpus Construction**:
  - Sample: 200 questions (100 bridge, 100 comparison, seed 43) from `hotpot_dev_distractor_v1.json`.
  - Raw context paragraphs processed: **2,000** paragraphs.
  - Identical duplicates removed: **19** paragraphs.
  - Final unique pooled corpus size ($N$): **1,981 paragraphs**.
- **Embedding Generation**:
  - Model: `embeddinggemma:latest` (768 dimensions) via `OllamaEmbeddingService`.
  - Total texts embedded: 200 questions + 1,981 corpus paragraphs = **2,181 texts**.
  - Wall clock time: **65.21 seconds** (1.09 minutes).
  - Saved to: [embeddings/corpus_embeddings.npy](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/embeddings/corpus_embeddings.npy) and [embeddings/question_embeddings.npy](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/embeddings/question_embeddings.npy).

### 3.1 Table 1: Pooled Recall@k Metrics across 1,981 Paragraphs (with 95% Wilson CI)

| Metric | Subset | N | k=3 | k=5 | k=10 | k=20 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Both Gold in Top-k** | Overall | 200 | **74.0%** (148/200)<br>[0.675, 0.796] | **88.5%** (177/200)<br>[0.833, 0.922] | **95.5%** (191/200)<br>[0.917, 0.976] | **98.0%** (196/200)<br>[0.950, 0.992] |
| At Least One Gold in Top-k | Overall | 200 | 99.5% (199/200)<br>[0.972, 0.999] | 99.5% (199/200)<br>[0.972, 0.999] | 99.5% (199/200)<br>[0.972, 0.999] | 100.0% (200/200)<br>[0.981, 1.000] |
| Hard Gold in Top-k | Overall | 200 | 74.0% (148/200)<br>[0.675, 0.796] | 88.5% (177/200)<br>[0.833, 0.922] | 95.5% (191/200)<br>[0.917, 0.976] | 98.0% (196/200)<br>[0.950, 0.992] |
| **Both Gold in Top-k** | Bridge | 100 | **62.0%** (62/100)<br>[0.522, 0.709] | **79.0%** (79/100)<br>[0.700, 0.858] | **92.0%** (92/100)<br>[0.850, 0.959] | **96.0%** (96/100)<br>[0.902, 0.984] |
| At Least One Gold in Top-k | Bridge | 100 | 99.0% (99/100)<br>[0.946, 0.998] | 99.0% (99/100)<br>[0.946, 0.998] | 99.0% (99/100)<br>[0.946, 0.998] | 100.0% (100/100)<br>[0.963, 1.000] |
| Hard Gold in Top-k | Bridge | 100 | 62.0% (62/100)<br>[0.522, 0.709] | 79.0% (79/100)<br>[0.700, 0.858] | 92.0% (92/100)<br>[0.850, 0.959] | 96.0% (96/100)<br>[0.902, 0.984] |
| **Both Gold in Top-k** | Comparison | 100 | **86.0%** (86/100)<br>[0.779, 0.915] | **98.0%** (98/100)<br>[0.930, 0.994] | **99.0%** (99/100)<br>[0.946, 0.998] | **100.0%** (100/100)<br>[0.963, 1.000] |
| At Least One Gold in Top-k | Comparison | 100 | 100.0% (100/100)<br>[0.963, 1.000] | 100.0% (100/100)<br>[0.963, 1.000] | 100.0% (100/100)<br>[0.963, 1.000] | 100.0% (100/100)<br>[0.963, 1.000] |
| Hard Gold in Top-k | Comparison | 100 | 86.0% (86/100)<br>[0.779, 0.915] | 98.0% (98/100)<br>[0.930, 0.994] | 99.0% (99/100)<br>[0.946, 0.998] | 100.0% (100/100)<br>[0.963, 1.000] |

### 3.2 Table 2: Hard Gold Rank Statistics & Drop from Own-10 to Pooled Setting

| Subset | N | Hard Gold Median Rank [IQR] | Own-10 Recall@3 (95% CI) | Pooled Recall@3 (95% CI) | Absolute Drop (Δ) | Relative Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 200 | 2.0 [2.0 - 4.0] | 78.0% (156/200)<br>[0.718, 0.832] | 74.0% (148/200)<br>[0.675, 0.796] | **-4.0 pts** | **-5.1%** |
| **Bridge** | 100 | 3.0 [2.0 - 5.0] | 68.0% (68/100)<br>[0.583, 0.763] | 62.0% (62/100)<br>[0.522, 0.709] | **-6.0 pts** | **-8.8%** |
| **Comparison** | 100 | 2.0 [2.0 - 3.0] | 88.0% (88/100)<br>[0.802, 0.930] | 86.0% (86/100)<br>[0.779, 0.915] | **-2.0 pts** | **-2.3%** |

---

## 4. Part 3: Knowledge Base Ingestion Benchmark (Public API)

- **Configuration Used**: `configs/domains/academic.yaml` merged with `configs/default.yaml`.
  - *Justification*: `academic.yaml` provides the broadest general-domain ontology (`PERSON`, `ORGANIZATION`, `LOCATION`, `EVENT`, `WORK`, `FILM`, `BOOK`, `COUNTRY`, `CONCEPT`) matching general Wikipedia context paragraphs, unlike the domain-restricted `company.yaml`, `legal.yaml`, or `medical_research.yaml`.
  - *Storage Root*: `experiments/hotpotqa_m4/scratch_kb` (completely outside `knowledge_bases/`).
- **Input Documents**: 10 context paragraphs of the first bridge question (`5ae56697`) written as individual `.txt` files under [scratch_docs/](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/scratch_docs/).

### Ingestion Metrics (Measured Live from Disk)
- **Total Ingestion Wall Time**: **635.17 seconds** (10.59 minutes)
- **Seconds per Paragraph**: **63.52 seconds / paragraph**
- **Documents / Pages / Chunks**: 10 documents / 10 pages / **12 chunks**
- **Entities**:
  - Entity Mentions Extracted: **96**
  - Resolved Canonical Entities: **78**
- **Relationships**:
  - Extracted Relationship Candidates: **63**
  - **Validated Relationships**: **11**
  - **Rejected Relationships**: **52** (Rejection rate: 82.5%)
  - *Rejection Breakdown by Reason Code*:
    - `REJECTED_ENDPOINT_NOT_IN_EVIDENCE`: **45**
    - `REJECTED_ONTOLOGY`: **5**
    - `REJECTED_QUOTE_MISMATCH`: **2**
- **Leiden Community Detection**: Ran: **True** (**68 communities** detected).
- **Community Summarization**: Ran: **True** (**68 community reports** generated and indexed into ChromaDB).
- **Graph Topology**: **78 vertices, 11 edges**.

### Ingestion Stage Time Breakdown
| Indexing Stage | Elapsed Time | Proportion of Total |
| :--- | :---: | :---: |
| `relationship_extraction` | 357.60 s | 56.3% |
| `entity_extraction` | 187.55 s | 29.5% |
| `community_summarization` | 58.35 s | 9.2% |
| `entity_resolution` | 30.12 s | 4.7% |
| `embedding_and_vector_indexing` | 1.41 s | 0.2% |
| `graph_construction_and_validation` | 0.02 s | 0.0% |
| `community_detection` (Leiden) | 0.02 s | 0.0% |
| `document_loading` & `chunking` | 0.00 s | 0.0% |
| **Total Pipeline** | **635.11 s** | **100.0%** |

### Linear Extrapolations (Explicitly Linear Based on 10-Paragraph Sample)
> [!NOTE]
> These projections assume linear scaling $\mathcal{O}(N)$ dominated by per-chunk LLM entity and relationship extraction. Superlinear scaling from graph operations ($\mathcal{O}(V + E)$ or community detection) is minimal at these sizes.
- **300 Paragraphs**: $300 \times 63.52\text{s} = 19,056\text{s}$ = **317.58 minutes (5.29 hours)**.
- **1,000 Paragraphs**: $1,000 \times 63.52\text{s} = 63,517\text{s}$ = **1,058.62 minutes (17.64 hours)**.

---

## 5. Honest Assessment

### 5.1 Retrieval Headroom in the Pooled Setting
1. **Comparison Questions**:
   - Dense retrieval with `embeddinggemma` is exceptionally strong on comparison questions. Even across 1,981 pooled paragraphs, recall@3 is **86.0%**, dropping only 2 percentage points from own-10 (**88.0%**).
   - At $k=5$, comparison recall is **98.0%**; at $k=10$, it is **99.0%**.
   - *Implication*: There is negligible headroom (~1–2%) for graph retrieval to improve paragraph recall on comparison questions when $k \ge 5$. The comparison entities are named explicitly in the question prompt (e.g. *"Did X and Y..."*), allowing dense embeddings to match both entities directly.

2. **Bridge Questions**:
   - On bridge questions, a substantial retrieval gap exists at small $k$:
     - At $k=3$, pooled recall for both gold paragraphs is only **62.0%** (a -6.0 point drop from own-10). The median rank of the hard gold paragraph is **3.0**, with 38% of questions missing the second gold paragraph.
     - At $k=5$, recall rises to **79.0%** (21% missing).
     - At $k=10$, recall reaches **92.0%**; at $k=20$, it reaches **96.0%**.
   - *Where Headroom Exists*: Graph traversal has meaningful headroom specifically for **bridge questions at small retrieval budgets ($k \le 5$)**. In a bridge question, the second gold entity is not named in the user question; dense retrieval can find the first entity easily (at least one gold is in top-3 for 99.0% of bridge questions), but fails on the second entity. Graph retrieval traversing relationship paths from entity 1 to entity 2 can recover that second paragraph at $k=3$ without flooding the generator's context window with 20 chunks.
   - At $k \ge 10$, dense retrieval already recovers 92%–96% of bridge gold pairs, meaning the graph advantage at large $k$ is bounded to 4%–8%.

### 5.2 Ingestion Cost and Graph Scale on this Machine
1. **Local Compute Bottleneck**:
   - Single-stream local LLM inference with Ollama (`gemma3:4b` on Apple Silicon) averages **63.52 seconds per paragraph**.
   - Over 85% of ingestion time is consumed by LLM extraction prompts (Stage 4 relationship extraction: 56.3%; Stage 4 entity extraction: 29.5%).
   - Ingestion of a 300-paragraph corpus (the 30 questions $\times$ 10 paragraphs from M2/M3) will take **~5.3 hours**. A 1,000-paragraph corpus will take **~17.6 hours**.
   - *Implication*: Any full-graph multi-hop evaluation conducted locally without batched parallel LLM workers must be strictly bounded (e.g. 100–300 paragraphs) to remain executable within a reasonable working session.

2. **Graph Sparsity in HotpotQA**:
   - In the 10-paragraph sample, the model extracted 63 relationship candidates, but **52 of 63 (82.5%) were rejected** by the strict validation pipeline (45 due to endpoints not grounded in the chunk evidence).
   - Only 11 relationships were validated, leaving 78 canonical entities in 68 communities (mostly disconnected singletons).
   - *Implication*: For graph retrieval to succeed on HotpotQA, either multiple overlapping documents must share entities, or co-occurrence / document-title discovery heuristics (Tasks 3 and 5) must bridge the gap across sparse extracted edges.

### 5.3 Limitations
1. **Sample Size & Seeds**: Part 2 evaluated 200 questions sampled under a single seed (seed 43). Part 3 measured ingestion on a single 10-paragraph sample.
2. **Embedding Model**: Dense retrieval was measured exclusively using `embeddinggemma:latest` (768 dimensions).
3. **Linear Extrapolation**: Ingestion projections assume linear $\mathcal{O}(N)$ scaling; while extraction dominates at $N \le 1,000$, graph clustering and community summarization could introduce non-linear overhead if graph density increases.
4. **No Git Modifications**: Zero commits were created; all artifacts remain isolated in `experiments/hotpotqa_m4/`.

---

### Persisted Artifacts
- **Adjudication Dump**: [adjudication_dump.txt](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/adjudication_dump.txt)
- **Sampled 200 Questions**: [pooled_questions.json](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/pooled_questions.json)
- **Corpus Metadata**: [pooled_corpus.json](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/pooled_corpus.json)
- **Embeddings**:
  - Corpus: [embeddings/corpus_embeddings.npy](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/embeddings/corpus_embeddings.npy)
  - Questions: [embeddings/question_embeddings.npy](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/embeddings/question_embeddings.npy)
- **Pooled Retrieval Results**: [pooled_retrieval_results.json](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/pooled_retrieval_results.json)
- **Pooled Retrieval Summary**: [pooled_recall_summary.json](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/pooled_recall_summary.json)
- **Scratch Ingestion Docs**: [scratch_docs/](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/scratch_docs/)
- **Scratch Knowledge Base**: [scratch_kb/](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/scratch_kb/)
- **Ingestion Benchmark Stats**: [ingestion_stats.json](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/ingestion_stats.json)
- **Progress Log**: [PROGRESS.md](file:///Users/shikharsrivastava/Documents/Documents/Python/hybrid-graphrag-system/experiments/hotpotqa_m4/PROGRESS.md)
