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

168 synthetic question/answer/context triples for evaluating retrieval-augmented generation (RAG) over public US healthcare regulation, together with the full 345-chunk retrieval corpus they were drawn from. Questions are written from the point of view of a compliance officer or healthcare privacy analyst.

> **Data quality warning.** A fourth judge critique (`answer_correctness`) checks the reference answer, but it does not make the answers reliable. In an LLM-agent spot-check of 20 kept rows, **13 were fully correct, 5 partly right ("should" stated as "must", a dropped condition, a list cut off at the chunk boundary) and 2 wrong** (7 of 20, 35%, not fully correct). This release has no yes/no questions (the previous one had 107 of 162, with 91 "Yes" against 6 "No"), so an always-"yes" system no longer scores well. 76 of 168 questions (45%) start with "Under what conditions". Use the set mainly for retrieval evaluation via `chunk_id`, and verify `answer` before treating it as ground truth. See [Limitations](#limitations).

```python
from datasets import load_dataset
qa = load_dataset("SolvrenAI/hipaa-fda-rag-eval", split="test")                    # 168 QA rows
corpus = load_dataset("SolvrenAI/hipaa-fda-rag-eval", "corpus", split="test")      # 345 chunks
```

Each QA row's `chunk_id` joins to a corpus row. The corpus `text` is byte-identical to the QA row's `source_chunk`. The 177 corpus chunks with no kept question are realistic distractors for retrieval evaluation (recall@k, MRR).

## Sources

All source text is published by the US government (public domain):

| Source | URL | Corpus chunks | Kept QA rows |
|---|---|---|---|
| 45 CFR Part 164 (eCFR snapshot of 2026-10-06): Subpart A General, C Security Rule, D Breach Notification, E Privacy Rule | https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part=164 | 195 (A 7, C 20, D 12, E 156) | 117 (A 4, C 10, D 11, E 92) |
| FDA guidance: *Cybersecurity in Medical Devices: Quality Management System Considerations and Content of Premarket Submissions* (Feb 3, 2026) | https://www.fda.gov/media/119933/download | 150 | 51 |

## Overlap with SolvrenAI/bge-base-en-v1.5-hipaa

The companion embedding model [SolvrenAI/bge-base-en-v1.5-hipaa](https://huggingface.co/SolvrenAI/bge-base-en-v1.5-hipaa) was trained on 45 CFR Part 160 and Part 164 Subpart E (Privacy Rule). Rows from Subpart E have `in_embedding_train = true` (92 of 168 QA rows, 156 of 345 corpus chunks). **When evaluating that model, filter these rows out** to avoid train/test leakage:

```python
qa_heldout = qa.filter(lambda r: not r["in_embedding_train"])   # 76 rows
```

## Generation method

This is a local port of the AWS sample [generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems](https://github.com/aws-samples/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems) (MIT-0). The code is at [solvren-ai/rag-eval-generator@0371fe3](https://github.com/solvren-ai/rag-eval-generator/tree/0371fe3b890f669d792fc42ae52f8ceb9d4d5ee5) (`run.py`, `--local` mode; it reproduces the committed chunks, and its filter reproduces the committed output from the run's scores). Everything ran locally on an Apple M5 (16 GB) with `mlx-lm` 0.32.2 and greedy decoding. No cloud APIs were used.

1. **Chunking.** The CFR is split one document per section, the FDA PDF one document per page. Each document is chunked with `RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)`, dropping chunks under 300 characters, giving 345 chunks. Every chunk is used.
2. **Generation.** [`mlx-community/Qwen3-4B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) (Apache-2.0) produces, for each chunk: a question, a compressed rewrite, a reference answer grounded only in the chunk, and the supporting sentences (`context`). It also self-scores the question (`self_*_score`). The question prompt is the notebook prompt with one added rule, repeated in its last line: prefer open questions (what, when, who, how long, under what conditions), and ask yes/no only when the chunk states a prohibition or exception, phrased so the answer is often "No".
3. **Independent judge.** [`mlx-community/Qwen3-8B-4bit`](https://huggingface.co/mlx-community/Qwen3-8B-4bit) (Apache-2.0, thinking disabled) is a different and larger model from the same family, loaded only after the generator was unloaded. It scores every row from 1 to 5 with four critiques. The first three rate the question only; the fourth reads the reference answer:
   - **groundedness**: whether the chunk fully answers the question (original notebook prompt)
   - **relevance**: usefulness to a compliance officer or healthcare privacy analyst (notebook prompt with the persona rewritten)
   - **standalone**: whether the question is understandable without the passage (added). The prompt says explicitly that standard regulatory abbreviations (HIPAA, PHI, ePHI, CE, BA, HHS, FDA, CFR) and section citations are not vague referents.
   - **answer_correctness**: whether the reference answer is correct and fully supported by the chunk (added). The prompt tells the judge to check exceptions and conditions ("except", "unless", "if", "only when"), "may" versus "must"/"shall"/"should", yes/no polarity, and overgeneralization.
4. **Filter.** A row is kept only when all four **judge** scores are ≥ 4. 345 generated → 168 kept (49%). Filtering on the 4B self-scores (three critiques) instead would have kept 259 rows; the first three judge critiques alone would keep 192, and `answer_correctness` removes 24 more.

**Changes from the previous release (162 rows).** The previous release had a strong "Yes" skew: 107 of 162 questions were yes/no, 91 answered "Yes" and 6 "No". The generator turned rules into "Can…?" questions, and the filter removed most "No" answers. All 345 questions were regenerated with the open-question rule above and judged with the same models and judge prompts. Over all 345 rows, 1 question is yes/no (the filter removed it), against about 200 before. The rule overshoots: the generator also wrote no "No"-answer questions about prohibitions. 97 chunks are in both releases, 65 were dropped and 71 are new. FDA rows fell from 70 to 51 and Privacy Rule rows rose from 70 to 92. More FDA questions now score below 4 on groundedness (64 of 150, against 46) and relevance (54, against 41).

**Changes from the first release (133 rows).** The first release used only the first three critiques. Its standalone rubric already allowed abbreviations, but the judge ignored that: 134 questions scored 3 on standalone, 95 of them with reasoning that called "CE" or "PHI" vague. With the reworded prompt, 30 score 3 (31 score 3 or lower). All 345 rows were re-judged with the same model. 109 rows are in both releases; 24 were dropped (22 by `answer_correctness` alone); 53 are new (38 of them Privacy Rule questions the abbreviation penalty used to remove).

Runtime: generation took about 1,950 s (about 33 min) for 345 chunks. Judging all 345 rows with four critiques took 1,846 s (about 31 min).

### Judge score distribution (Qwen3-8B), all 345 → kept 168

| Score | groundedness | relevance | standalone | answer_correctness |
|---|---|---|---|---|
| 5 | 82 → 59 | 184 → 129 | 267 → 148 | 199 → 124 |
| 4 | 172 → 109 | 73 → 39 | 55 → 20 | 93 → 44 |
| 3 | 82 | 77 | 23 | 52 |
| 2 | 6 | 7 | 0 | 0 |
| 1 | 3 | 4 | 0 | 1 |
| mean (kept) | 4.35 | 4.77 | 4.88 | 4.74 |

Self-scores (Qwen3-4B, all 345): groundedness 5:71 / 4:232 / 3:23 / 2:16 / 1:3; relevance 5:205 / 4:93 / 3:32 / 2:11 / 1:4; standalone 5:332 / 3:13.

### Spot-check

An LLM agent, not a human, spot-checked kept rows, checking each answer against its `source_chunk` with attention to exceptions, may/must and yes/no polarity. Nobody has reviewed the rows by hand.

**Current release (168 rows):** 20 rows (`random.Random(5).sample` over `data/test.jsonl`). **13 fully correct, 5 partly right, 2 wrong.**

| chunk_id | Verdict | Problem |
|---|---|---|
| `FDA Guidance: Cybersecurity in Medical Devices, p.6#0` | wrong | The chunk does not say what FDA considers when evaluating a 510(k); the answer restates its pointer to the consensus-standards guidances |
| `45 CFR 164.502#0` | wrong | Asks when a CE *may* disclose PHI to the individual; the answer gives the *required*-disclosure condition (§ 164.524/528), but (a)(1)(i) permits it without conditions |
| `FDA Guidance: Cybersecurity in Medical Devices, p.40#1` | partial | Says the pathways "require" a reasonable assurance of cybersecurity; the chunk says it "can be part of" and "is relevant to" them |
| `FDA Guidance: Cybersecurity in Medical Devices, p.54#0` | partial | States FDA's recommended IDE documentation as what an IDE application "must include" |
| `FDA Guidance: Cybersecurity in Medical Devices, p.30#1` | partial | "should provide" stated as "must provide" |
| `45 CFR 164.308#1` | partial | Says "must" for addressable specifications and turns "authorization and/or supervision" into "and" |
| `45 CFR 164.512#12` | partial | Presents the conditions as complete; the chunk ends after (A) and the rule has more |

**Compared with the previous release (162 rows; 25 rows, seed 11: 18 correct, 6 partial, 1 wrong).** The yes/no skew is gone. Answer quality is about the same: 13 of 20 fully correct (65%) against 18 of 25 (72%), and 2 wrong against 1. With samples this small the difference is noise. The problems are the same kinds as before: "should" stated as "must" (all in FDA rows), a dropped or cut-off condition, and a question the chunk does not answer. `164.308#1` and `164.502#0` were also flagged in the previous release. Expect roughly 1 in 3 reference answers to be imprecise and a few to be wrong. The questions and `source_chunk` are more reliable than the answers.

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

- **Reference answers are imprecise.** 7 of 20 spot-checked (35%) are not fully correct: 5 partly right, 2 wrong. The `answer_correctness` critique removes clear contradictions but passes dropped conditions and should/must slips. Verify answers before using them as ground truth. The set works better for retrieval evaluation (via `chunk_id`) than for answer grading.
- **No yes/no questions, many "Under what conditions".** The previous release was skewed to "Yes" (91 "Yes" against 6 "No"). The regenerated questions are all open: 85 start with "What", 76 with "Under what conditions", 4 with "When" and 3 with "Who". The prompt allows yes/no questions about prohibitions with a "No" answer, but the 4B generator wrote none, so the set does not test whether a system can say "No" to a prohibited action. The many "Under what conditions" questions have long, list-like answers, which are hard to grade with exact-match metrics.
- **Synthetic, no human review.** Questions, answers and scores come from small open models (4B generator, 8B judge, same family). The spot-check was done by an LLM agent.
- **Judge limits.** The 8B judge is from the same family as the generator, and it misses partial errors (see the spot-check). The abbreviation penalty of the first release is largely fixed: In this release Privacy Rule rows survive more often (92 of 156) than FDA rows (51 of 150).
- **Single-hop.** One question per chunk, answerable from that chunk alone. No multi-hop or cross-section questions.
- **Not legal advice.** Answers paraphrase regulatory text and may be incomplete. The CFR is a 2026-10-06 snapshot and the FDA guidance is the Feb 2026 version. Later amendments are not reflected.
- **No PHI.** All inputs are public regulatory text. The dataset contains no real patient data or other personal information.
- **Coverage skew.** Subpart E (Privacy Rule, 92) and the FDA guidance (51) dominate. Subparts A, C and D together contribute 25 kept rows.

## Reproduce

```sh
git clone https://github.com/solvren-ai/rag-eval-generator.git
cd rag-eval-generator
git checkout 0371fe3b890f669d792fc42ae52f8ceb9d4d5ee5
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
  note         = {Code: https://github.com/solvren-ai/rag-eval-generator/tree/0371fe3b890f669d792fc42ae52f8ceb9d4d5ee5}
}
```

Source text: US eCFR and FDA guidance (public domain). Adapted from aws-samples/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems (MIT-0).
