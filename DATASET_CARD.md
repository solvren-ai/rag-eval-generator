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

133 synthetic question/answer/context triples for evaluating retrieval-augmented generation (RAG) over public US healthcare regulation, together with the full 345-chunk retrieval corpus they were drawn from. Questions are written from the point of view of a compliance officer or healthcare privacy analyst.

```python
from datasets import load_dataset
qa = load_dataset("SolvrenAI/hipaa-fda-rag-eval", split="test")                    # 133 QA rows
corpus = load_dataset("SolvrenAI/hipaa-fda-rag-eval", "corpus", split="test")      # 345 chunks
```

Each QA row's `chunk_id` joins to a corpus row. The corpus `text` is byte-identical to the QA row's `source_chunk`. The 212 corpus chunks with no kept question are realistic distractors for retrieval evaluation (recall@k, MRR).

## Sources

All source text is published by the US government (public domain):

| Source | URL | Corpus chunks | Kept QA rows |
|---|---|---|---|
| 45 CFR Part 164 (eCFR snapshot of 2026-10-06): Subpart A General, C Security Rule, D Breach Notification, E Privacy Rule | https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part=164 | 195 (A 7, C 20, D 12, E 156) | 60 (A 2, C 9, D 3, E 46) |
| FDA guidance: *Cybersecurity in Medical Devices: Quality Management System Considerations and Content of Premarket Submissions* (Feb 3, 2026) | https://www.fda.gov/media/119933/download | 150 | 73 |

## Overlap with SolvrenAI/bge-base-en-v1.5-hipaa

The companion embedding model [SolvrenAI/bge-base-en-v1.5-hipaa](https://huggingface.co/SolvrenAI/bge-base-en-v1.5-hipaa) was trained on 45 CFR Part 160 and Part 164 Subpart E (Privacy Rule). Rows from Subpart E have `in_embedding_train = true` (46 of 133 QA rows, 156 of 345 corpus chunks). **When evaluating that model, filter these rows out** to avoid train/test leakage:

```python
qa_heldout = qa.filter(lambda r: not r["in_embedding_train"])   # 87 rows
```

## Generation method

This is a local port of the AWS sample [generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems](https://github.com/aws-samples/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems) (MIT-0). The code is at [solvren-ai/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems](https://github.com/solvren-ai/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems/tree/mac-local) (`local/run_local.py`). Everything ran locally on an Apple M5 (16 GB) with `mlx-lm` 0.32.2 and greedy decoding. No cloud APIs were used.

1. **Chunking.** The CFR is split one document per section, the FDA PDF one document per page. Each document is chunked with `RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)`, dropping chunks under 300 characters, giving 345 chunks. Every chunk is used.
2. **Generation.** [`mlx-community/Qwen3-4B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) (Apache-2.0) produces, for each chunk: a question, a compressed rewrite, a reference answer grounded only in the chunk, and the supporting sentences (`context`). It also self-scores the question (`self_*_score`).
3. **Independent judge.** [`mlx-community/Qwen3-8B-4bit`](https://huggingface.co/mlx-community/Qwen3-8B-4bit) (Apache-2.0, thinking disabled) is a different and larger model, loaded only after the generator was unloaded. It scores every question from 1 to 5:
   - **groundedness**: whether the chunk fully answers the question (original notebook prompt)
   - **relevance**: usefulness to a compliance officer or healthcare privacy analyst (notebook prompt with the persona rewritten)
   - **standalone**: whether the question is understandable without the passage (added)
4. **Filter.** A row is kept only when all three **judge** scores are ≥ 4. 345 generated → 133 kept (39%). Filtering on the 4B self-scores instead would have kept 247 rows. The 8B judge is clearly stricter, mainly on standalone and groundedness.

Runtime: generation took 2,674 s (about 45 min) for 345 chunks. Judging took 1,164 s for the last 285 rows, plus roughly 4 min for the first 60 in an earlier, interrupted run.

### Judge score distribution (Qwen3-8B), all 345 → kept 133

| Score | groundedness | relevance | standalone |
|---|---|---|---|
| 5 | 142 → 78 | 221 → 110 | 108 → 70 |
| 4 | 112 → 55 | 60 → 23 | 103 → 63 |
| 3 | 76 | 56 | 134 |
| 2 | 10 | 4 | 0 |
| 1 | 5 | 4 | 0 |
| mean (kept) | 4.59 | 4.83 | 4.53 |

Self-scores (Qwen3-4B, all 345): groundedness 5:113 / 4:185 / 3:20 / 2:25 / 1:2; relevance 5:203 / 4:108 / 3:24 / 2:8 / 1:2; standalone 5:311 / 3:34.

### Spot-check

This was a manual spot-check of 20 random kept rows (`random.Random(42).sample`) by an automated reviewer (LLM agent), not a human. Each answer was checked against its `source_chunk`. **16 of 20 passed (80%).** The failures:

| chunk_id | Failure type |
|---|---|
| `45 CFR 164.528#0` | Polarity error: the chunk excludes treatment disclosures from the accounting, but the answer says they cannot be excluded |
| `45 CFR 164.506#0` | Polarity error: the answer says consent is required for treatment disclosures, but the chunk permits them without consent |
| `45 CFR 164.508#5` | Misread conditional: a clause about conditioning enrollment on an authorization is answered as a permission to disclose |
| `45 CFR 164.524#6` | Self-contradictory answer (says "No", then "yes") |

The main pattern is yes/no questions about rules that come with exceptions. Expect roughly 1 in 5 reference answers to have a substantive error. The questions and `source_chunk` are more reliable than the answers. Among the passes, several answers mention "the context" or bare paragraph numbers, which is harmless but stylistically off.

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
| `groundedness_score`, `relevance_score`, `standalone_score` | Judge (Qwen3-8B) scores, 1–5; used for filtering |
| `groundedness_reasoning`, `relevance_reasoning`, `standalone_reasoning` | Judge's one-sentence rationale |
| `self_groundedness_score`, `self_relevance_score`, `self_standalone_score` | Generator (Qwen3-4B) self-scores, 1–5 |
| `generator_model`, `judge_model` | Model IDs |

`data/corpus.jsonl` (`corpus` config) has the fields `chunk_id`, `source`, `source_url`, `cfr_subpart`, `in_embedding_train` and `text`.

## Limitations

- **Synthetic.** Questions, answers and scores were produced by small open models (4B generator, 8B judge). See the spot-check: about 20% of reference answers may be wrong, especially on yes/no questions with exceptions. Verify answers before using them as hard ground truth. They work better for retrieval evaluation (via `chunk_id`) than for exact-answer grading.
- **Single-hop.** One question per chunk, answerable from that chunk alone. No multi-hop or cross-section questions.
- **Not legal advice.** Answers paraphrase regulatory text and may be incomplete. The CFR is a 2026-10-06 snapshot and the FDA guidance is the Feb 2026 version. Later amendments are not reflected.
- **No PHI.** All inputs are public regulatory text. The dataset contains no real patient data or other personal information.
- **Skew.** Subpart E (Privacy Rule) and the FDA guidance dominate. Subparts A, C and D together contribute 14 kept rows.

## Reproduce

```sh
git clone -b mac-local https://github.com/solvren-ai/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems.git
cd generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems
uv run local/run_local.py gen        # resumable
uv run local/run_local.py judge      # resumable
uv run local/run_local.py finalize   # data/unfiltered.jsonl, data/test.jsonl, data/corpus.jsonl
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
  note         = {Code: https://github.com/solvren-ai/rag-eval-generator}
}
```

Source text: US eCFR and FDA guidance (public domain). Adapted from aws-samples/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems (MIT-0).
