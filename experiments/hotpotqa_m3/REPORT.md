# Milestone 3 Generator Evaluation Results

## 1. Summary Performance Across Conditions

| Condition | Subset | N | Contains Acc (Count) | 95% Wilson CI | Exact Match (EM) | Mean F1 | ABSTAINED | ERROR | Abs-but-Present |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A_v11` | Overall | 30 | 66.7% (20/30) | [0.488, 0.808] | 46.7% (14/30) | 0.616 | 2 | 1 | 1 |
| `A_v11` | Bridge | 15 | 66.7% (10/15) | [0.417, 0.848] | 53.3% (8/15) | 0.746 | 1 | 1 | 0 |
| `A_v11` | Comparison | 15 | 66.7% (10/15) | [0.417, 0.848] | 40.0% (6/15) | 0.487 | 1 | 0 | 1 |
| `A_v10` | Overall | 30 | 66.7% (20/30) | [0.488, 0.808] | 50.0% (15/30) | 0.661 | 2 | 0 | 2 |
| `A_v10` | Bridge | 15 | 66.7% (10/15) | [0.417, 0.848] | 53.3% (8/15) | 0.768 | 0 | 0 | 0 |
| `A_v10` | Comparison | 15 | 66.7% (10/15) | [0.417, 0.848] | 46.7% (7/15) | 0.553 | 2 | 0 | 2 |
| `B` | Overall | 30 | 76.7% (23/30) | [0.591, 0.882] | 43.3% (13/30) | 0.589 | 1 | 0 | 1 |
| `B` | Bridge | 15 | 73.3% (11/15) | [0.480, 0.891] | 53.3% (8/15) | 0.714 | 0 | 0 | 0 |
| `B` | Comparison | 15 | 80.0% (12/15) | [0.548, 0.930] | 33.3% (5/15) | 0.464 | 1 | 0 | 1 |
| `C` | Overall | 30 | 63.3% (19/30) | [0.455, 0.781] | 43.3% (13/30) | 0.586 | 2 | 0 | 0 |
| `C` | Bridge | 15 | 60.0% (9/15) | [0.357, 0.802] | 46.7% (7/15) | 0.674 | 1 | 0 | 0 |
| `C` | Comparison | 15 | 66.7% (10/15) | [0.417, 0.848] | 40.0% (6/15) | 0.498 | 1 | 0 | 0 |


## 2. Compact 30-Question Symbol Matrix

| # | Question ID | Type | `A_v11` | `A_v10` | `B` | `C` | Gold Answer |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `5ae56697` | bridge | FAIL | FAIL | FAIL | FAIL | `presidential campaign of ...` |
| 2 | `5adf0cbd` | bridge | PASS | PASS | FAIL | FAIL | `6 April 1830` |
| 3 | `5a7caae7` | bridge | PASS | FAIL | PASS | PASS | `Anabolic steroids` |
| 4 | `5a8aed15` | bridge | PASS | PASS | PASS | PASS | `URO VAMTAC` |
| 5 | `5aba71b1` | bridge | PASS | PASS | PASS | PASS | `first flume ride in Ireland` |
| 6 | `5a888b8e` | bridge | PASS | PASS | PASS | PASS | `New York City, Long Islan...` |
| 7 | `5abbcba0` | bridge | FAIL | FAIL | PASS | FAIL | `Margarine Unie` |
| 8 | `5a77ba6c` | bridge | PASS | PASS | PASS | PASS | `June 11, 1986` |
| 9 | `5ae07e5b` | bridge | PASS | PASS | PASS | PASS | `Big Bad Wolf` |
| 10 | `5ae5dae2` | bridge | PASS | PASS | PASS | PASS | `Sarajevo` |
| 11 | `5a77ddb8` | bridge | FAIL | FAIL | FAIL | FAIL | `Cynthia Ann Stephanie "Cy...` |
| 12 | `5a76067e` | bridge | FAIL | FAIL | FAIL | FAIL | `Nicholas John "Nick" McCa...` |
| 13 | `5ade0a27` | bridge | PASS | PASS | PASS | ABS | `Flatbush section of Brook...` |
| 14 | `5ade24e3` | bridge | ERR | PASS | PASS | PASS | `CBS` |
| 15 | `5ae68fcb` | bridge | PASS | PASS | PASS | PASS | `Bay Ridge, Brooklyn` |
| 16 | `5ae07b01` | comparison | FAIL | FAIL | PASS | FAIL | `Paul Thomas Mann` |
| 17 | `5a7ca0ef` | comparison | PASS | PASS | PASS | PASS | `Raffaella Reggi` |
| 18 | `5ab2e3c4` | comparison | PASS | PASS | PASS | PASS | `yes` |
| 19 | `5a70f933` | comparison | PASS | ABS | PASS | PASS | `ALT for Damerne` |
| 20 | `5a8a3dd5` | comparison | ABS | ABS | ABS | PASS | `no` |
| 21 | `5a8e3dc0` | comparison | PASS | PASS | PASS | PASS | `1964` |
| 22 | `5ae26905` | comparison | FAIL | PASS | FAIL | FAIL | `yes` |
| 23 | `5a7a2be2` | comparison | PASS | PASS | PASS | PASS | `Del Amitri` |
| 24 | `5a7fe0b5` | comparison | PASS | PASS | PASS | ABS | `no` |
| 25 | `5a7380b9` | comparison | PASS | PASS | PASS | PASS | `Monsters, Inc.` |
| 26 | `5abbb116` | comparison | PASS | PASS | PASS | PASS | `University of Rochester` |
| 27 | `5ab2b630` | comparison | FAIL | FAIL | FAIL | FAIL | `yes` |
| 28 | `5a85ee63` | comparison | PASS | PASS | PASS | PASS | `Dal Lake` |
| 29 | `5a8b82a0` | comparison | FAIL | FAIL | PASS | FAIL | `yes` |
| 30 | `5abfd3ba` | comparison | PASS | PASS | PASS | PASS | `Ross Case` |


## 3. Condition C Dense Retrieval Recall Breakdown

| Gold Recall @ Top 3 | Paragraphs Retrieved | Count (Proportion) | Contains Correct | Contains Acc within Bucket |
| :--- | :---: | :---: | :---: | :---: |
| Both Gold Present | `2/2` | 24/30 (80.0%) | 15/24 | 62.5% |
| One Gold Present | `1/2` | 6/30 (20.0%) | 4/6 | 66.7% |
| Neither Gold Present | `0/2` | 0/30 (0.0%) | 0/0 | 0.0% |

