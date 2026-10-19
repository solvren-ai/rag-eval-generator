---
license: cc0-1.0
language:
- en
pretty_name: HIPAA and FDA RAG Eval
size_categories:
- n<1K
task_categories:
- question-answering
- text-retrieval
tags:
- rag
- rag-evaluation
- synthetic
- hipaa
- healthcare
- compliance
- regulatory
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test.jsonl
- config_name: corpus
  data_files:
  - split: test
    path: data/corpus.jsonl
---

# HIPAA and FDA RAG Eval

162 synthetic question/answer/context triples for evaluating retrieval-augmented generation (RAG) over public US healthcare regulation, together with the full 345-chunk retrieval corpus they were drawn from. Questions are written from the point of view of a compliance officer or healthcare privacy analyst.

> **Data quality warning.** A fourth judge critique (`answer_correctness`) now checks the reference answer, and it removed the outright polarity errors found earlier. It does not make the answers reliable. In an LLM-agent spot-check of 25 kept rows, **18 were fully correct, 6 partly right (a dropped condition, "should" stated as "must", an incomplete list) and 1 wrong** (7 of 25, about 28%, not fully correct). The answers are also skewed: 107 of 162 questions (66%) are yes/no, and **91 answers open with "Yes" against 6 with "No"**, so an always-"yes" system scores well on answers. Use the set mainly for retrieval evaluation via `chunk_id`, and verify `answer` before treating it as ground truth. See [Limitations](#limitations).

```python
from datasets import load_dataset
qa = load_dataset("SolvrenAI/hipaa-fda-rag-eval", split="test")                    # 162 QA rows
corpus = load_dataset("SolvrenAI/hipaa-fda-rag-eval", "corpus", split="test")      # 345 chunks
```

Each QA row's `chunk_id` joins to a corpus row. The corpus `text` is byte-identical to the QA row's `source_chunk`. The 183 corpus chunks with no kept question are realistic distractors for retrieval evaluation (recall@k, MRR).

## Sources

All source text is published by the US government (public domain):

| Source | URL | Corpus chunks | Kept QA rows |
|---|---|---|---|
| 45 CFR Part 164 (eCFR snapshot of 2026-10-06): Subpart A General, C Security Rule, D Breach Notification, E Privacy Rule | https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part=164 | 195 (A 7, C 20, D 12, E 156) | 92 (A 5, C 13, D 4, E 70) |
| FDA guidance: *Cybersecurity in Medical Devices: Quality Management System Considerations and Content of Premarket Submissions* (Feb 3, 2026) | https://www.fda.gov/media/119933/download | 150 | 70 |

## Overlap with SolvrenAI/bge-base-en-v1.5-hipaa

The companion embedding model [SolvrenAI/bge-base-en-v1.5-hipaa](https://huggingface.co/SolvrenAI/bge-base-en-v1.5-hipaa) was trained on 45 CFR Part 160 and Part 164 Subpart E (Privacy Rule). Rows from Subpart E have `in_embedding_train = true` (70 of 162 QA rows, 156 of 345 corpus chunks). **When evaluating that model, filter these rows out** to avoid train/test leakage:

```python
qa_heldout = qa.filter(lambda r: not r["in_embedding_train"])   # 92 rows
```

## Generation method

This is a local port of the AWS sample [generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems](https://github.com/aws-samples/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems) (MIT-0). The code is at [solvren-ai/rag-eval-generator@8022d66](https://github.com/solvren-ai/rag-eval-generator/tree/8022d66a30fb73f70f9cd363dc270684c2a2f654) (`run.py`, `--local` mode; it reproduces the committed chunks and filter output exactly). Everything ran locally on an Apple M5 (16 GB) with `mlx-lm` 0.32.2 and greedy decoding. No cloud APIs were used.

1. **Chunking.** The CFR is split one document per section, the FDA PDF one document per page. Each document is chunked with `RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)`, dropping chunks under 300 characters, giving 345 chunks. Every chunk is used.
2. **Generation.** [`mlx-community/Qwen3-4B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) (Apache-2.0) produces, for each chunk: a question, a compressed rewrite, a reference answer grounded only in the chunk, and the supporting sentences (`context`). It also self-scores the question (`self_*_score`).
3. **Independent judge.** [`mlx-community/Qwen3-8B-4bit`](https://huggingface.co/mlx-community/Qwen3-8B-4bit) (Apache-2.0, thinking disabled) is a different and larger model from the same family, loaded only after the generator was unloaded. It scores every row from 1 to 5 with four critiques. The first three rate the question only; the fourth reads the reference answer:
   - **groundedness**: whether the chunk fully answers the question (original notebook prompt)
   - **relevance**: usefulness to a compliance officer or healthcare privacy analyst (notebook prompt with the persona rewritten)
   - **standalone**: whether the question is understandable without the passage (added). The prompt says explicitly that standard regulatory abbreviations (HIPAA, PHI, ePHI, CE, BA, HHS, FDA, CFR) and section citations are not vague referents.
   - **answer_correctness**: whether the reference answer is correct and fully supported by the chunk (added). The prompt tells the judge to check exceptions and conditions ("except", "unless", "if", "only when"), "may" versus "must"/"shall"/"should", yes/no polarity, and overgeneralization.
4. **Filter.** A row is kept only when all four **judge** scores are ≥ 4. 345 generated → 162 kept (47%). Filtering on the 4B self-scores (three critiques) instead would have kept 247 rows; the first three judge critiques alone would keep 201, and `answer_correctness` removes 39 more.

**Changes from the first release (133 rows).** The first release used only the first three critiques. Its standalone rubric already allowed abbreviations, but the judge ignored that: 134 questions scored 3 on standalone, 95 of them with reasoning that called "CE" or "PHI" vague. With the reworded prompt, 30 score 3 (31 score 3 or lower). All 345 rows were re-judged with the same model. 109 rows are in both releases; 24 were dropped (22 by `answer_correctness` alone); 53 are new (38 of them Privacy Rule questions the abbreviation penalty used to remove).

Runtime: generation took 2,674 s (about 45 min) for 345 chunks. Re-judging all 345 rows with four critiques took 1,892 s (about 32 min).

### Judge score distribution (Qwen3-8B), all 345 → kept 162

| Score | groundedness | relevance | standalone | answer_correctness |
|---|---|---|---|---|
| 5 | 141 → 102 | 223 → 132 | 259 → 139 | 181 → 115 |
| 4 | 113 → 60 | 58 → 30 | 55 → 23 | 82 → 47 |
| 3 | 75 | 56 | 30 | 79 |
| 2 | 11 | 4 | 1 | 3 |
| 1 | 5 | 4 | 0 | 0 |
| mean (kept) | 4.63 | 4.81 | 4.86 | 4.71 |

Self-scores (Qwen3-4B, all 345): groundedness 5:113 / 4:185 / 3:20 / 2:25 / 1:2; relevance 5:203 / 4:108 / 3:24 / 2:8 / 1:2; standalone 5:311 / 3:34.

### Spot-check

An LLM agent, not a human, spot-checked kept rows, checking each answer against its `source_chunk` with attention to exceptions, may/must and yes/no polarity. Nobody has reviewed the rows by hand.

**Current release (162 rows):** 25 rows (`random.Random(11).sample` over `data/test.jsonl`). **18 fully correct, 6 partly right, 1 wrong.**

| chunk_id | Verdict | Problem |
|---|---|---|
| `FDA Guidance: Cybersecurity in Medical Devices, p.31#1` | wrong | Attributes security-testing standards (footnote 48) to labeling requirements |
| `45 CFR 164.512#11` | partial | Presents the list of information for locating a suspect as complete; the chunk ends mid-list and the rule has one more item |
| `45 CFR 164.502#0` | partial | "Yes" is right, but it ties the permission to the required-disclosure clause and skips the general permission in (a)(1)(i) |
| `45 CFR 164.501#2` | partial | Mixes up the conditions: the "not generalizable knowledge" condition belongs to quality assessment, the "no treatment" condition to related functions |
| `FDA Guidance: Cybersecurity in Medical Devices, p.36#0` | partial | Overgeneralizes: Wi-Fi/Bluetooth makes a device internet-capable, one criterion for a cyber device, not on its own a monitoring-plan requirement |
| `FDA Guidance: Cybersecurity in Medical Devices, p.12#1` | partial | States a recommendation ("should take into account") as a requirement ("must", "requiring"), and adds claims not in the chunk |
| `45 CFR 164.508#8` | partial | Drops the condition: the copy is required when the covered entity itself seeks the authorization |

**Earlier failures.** Of the 8 rows that failed or were partly right in the first release's spot-checks (35 rows, seeds 42 and 7: 27 clean, 8 wrong or partial), 6 are now removed by `answer_correctness` (scored 3), including all four polarity and self-contradiction errors (`45 CFR 164.528#0`, `164.506#0`, `164.508#5`, `164.524#6`), plus `164.501#4` and FDA `p.49#0`. Two are still kept: FDA `p.31#1` (still wrong; the judge's own reasoning names the error but it scored 4) and `45 CFR 164.308#1` (still partial: it asks about clearance and termination but answers only termination, and says "must" for addressable specifications).

**Compared with the first release.** The strict pass rate is about the same (18 of 25, 72%, against 27 of 35, 77%; with samples this small the difference is noise). Outright wrong answers fell from 6 of 35 to 1 of 25. What remains is mostly partial: a dropped condition, "should" stated as "must", or an incomplete list. Expect roughly 1 in 4 reference answers to be imprecise and a few to be wrong. The questions and `source_chunk` are more reliable than the answers.

## Fields

`data/test.jsonl` (default config) and `data/unfiltered.jsonl` (all 345 rows):

| Field | Description |
|---|---|
| `chunk_id` | `<source>#<chunk index>`; joins to `corpus.jsonl` |
| `question` | Generated question |
| `question_compressed` | Shorter, more indirect rewrite (for robustness testing) |
| `answer` | Reference answer generated from the chunk only |
| `context` | Supporting sentences extracted from the chunk by the LLM (usually verbatim, not guaranteed) |
| `source_chunk` | Full source chunk (the gold retrieval passage) |
| `source` | Citation: CFR section (e.g. `45 CFR 164.408`) or FDA guidance page |
| `source_url` | Download URL of the source document |
| `cfr_subpart` | `A`/`C`/`D`/`E` for CFR rows, `null` for FDA |
| `in_embedding_train` | `true` for Subpart E rows (see overlap note) |
| `groundedness_score`, `relevance_score`, `standalone_score`, `answer_correctness_score` | Judge (Qwen3-8B) scores, 1–5; all four used for filtering |
| `groundedness_reasoning`, `relevance_reasoning`, `standalone_reasoning`, `answer_correctness_reasoning` | Judge's one-sentence rationale |
| `self_groundedness_score`, `self_relevance_score`, `self_standalone_score` | Generator (Qwen3-4B) self-scores, 1–5 |
| `generator_model`, `judge_model` | Model IDs |

`data/corpus.jsonl` (`corpus` config) has the fields `chunk_id`, `source`, `source_url`, `cfr_subpart`, `in_embedding_train` and `text`.

## Limitations

- **Reference answers are imprecise.** 7 of 25 spot-checked (about 28%) are not fully correct: 6 partly right, 1 wrong. The `answer_correctness` critique removes clear contradictions but passes dropped conditions and should/must slips, and it once scored a misattribution 4 while naming the error. Verify answers before using them as ground truth. The set works better for retrieval evaluation (via `chunk_id`) than for answer grading.
- **Strong "Yes" skew.** 107 of 162 questions (66%) are yes/no ("Can…", "Should…", "Does…"), and 91 answers open with "Yes" against 6 with "No". This is mostly a pipeline artifact, not a property of the text. The generator turns statements in the chunk into questions, so most yes/no questions restate a rule and get "Yes" (140 Yes, 62 No over all 345 rows). The filter then keeps 65% of the "Yes" rows but only 10% of the "No" rows, because "No" answers are where the generator most often misreads an exception. The Privacy Rule is full of prohibitions, so a real question set would have many more "No" answers. An always-"yes" system will look much better than it is on answer metrics; report answer scores on yes/no and open questions separately.
- **Synthetic, no human review.** Questions, answers and scores come from small open models (4B generator, 8B judge, same family). The spot-check was done by an LLM agent.
- **Judge limits.** The 8B judge is from the same family as the generator, and it misses partial errors (see the spot-check). The abbreviation penalty of the first release is largely fixed: Privacy Rule rows now survive about as often (70 of 156) as FDA rows (70 of 150).
- **Single-hop.** One question per chunk, answerable from that chunk alone. No multi-hop or cross-section questions.
- **Not legal advice.** Answers paraphrase regulatory text and may be incomplete. The CFR is a 2026-10-06 snapshot and the FDA guidance is the Feb 2026 version. Later amendments are not reflected.
- **No PHI.** All inputs are public regulatory text. The dataset contains no real patient data or other personal information.
- **Coverage skew.** Subpart E (Privacy Rule, 70) and the FDA guidance (70) dominate. Subparts A, C and D together contribute 22 kept rows.

## Reproduce

```sh
git clone https://github.com/solvren-ai/rag-eval-generator.git
cd rag-eval-generator
git checkout 8022d66a30fb73f70f9cd363dc270684c2a2f654
uv run run.py generate --local   # Qwen3-4B; resumable
uv run run.py judge --local      # Qwen3-8B, four critiques; resumable
uv run run.py filter             # data/unfiltered.jsonl, data/test.jsonl, data/corpus.jsonl
```

## Citation

If you use this dataset in your work, please cite:

```bibtex
@misc{solvren2026hipaafdaragqa,
  title        = {HIPAA and FDA RAG Eval: Synthetic QA for Evaluating Retrieval-Augmented Generation on US Healthcare Regulation},
  author       = {{Solvren AI}},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/SolvrenAI/hipaa-fda-rag-eval}},
  note         = {Code: https://github.com/solvren-ai/rag-eval-generator/tree/8022d66a30fb73f70f9cd363dc270684c2a2f654}
}
```

Source text: US eCFR and FDA guidance (public domain). Adapted from aws-samples/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems (MIT-0).
