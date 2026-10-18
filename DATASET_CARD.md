---
license: cc0-1.0
language:
- en
pretty_name: Synthetic HIPAA & FDA Medical Device Cybersecurity RAG Evaluation Set
size_categories:
- n<1K
task_categories:
- question-answering
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
---

# Synthetic HIPAA & FDA Medical Device Cybersecurity RAG Evaluation Set

134 synthetic question/answer/context triples for evaluating retrieval-augmented generation (RAG) systems over public US healthcare regulations. Questions are written from the point of view of a compliance officer or healthcare privacy analyst.

## Sources

All source text is published by the US government (public domain):

| Source | URL | Kept rows |
|---|---|---|
| 45 CFR Part 164: HIPAA Security Rule (Subpart C), Breach Notification Rule (Subpart D), Privacy Rule (Subpart E), via the eCFR API (as of 2025-10-01) | https://www.ecfr.gov/api/versioner/v1/full/2025-10-01/title-45.xml?part=164 | 83 |
| FDA guidance: *Cybersecurity in Medical Devices: Quality Management System Considerations and Content of Premarket Submissions* (Feb 3, 2026) | https://www.fda.gov/media/119933/download | 51 |

## Generation method

This is a local port of the AWS sample [generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems](https://github.com/aws-samples/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems) (MIT-0). See `local/run_local.py`.

1. The CFR is split one document per section and the FDA PDF one document per page. Each document is then chunked with `RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)`, giving 345 chunks. They are shuffled with seed 0 and the first 192 are used.
2. For each chunk the LLM generates: a question, a compressed rewrite of the question, a reference answer grounded only in the chunk, and the verbatim supporting sentences (`context`).
3. Three LLM critiques score each question from 1 to 5:
   - **groundedness**: whether the chunk fully answers the question (original notebook prompt)
   - **relevance**: usefulness to a compliance officer or healthcare privacy analyst (notebook prompt with the persona rewritten)
   - **standalone**: whether the question is understandable without the passage (added)
4. Filter: a row is kept only when all three scores are ≥ 4. 192 generated → 134 kept (70%).

**Generator and critic model:** [`mlx-community/Qwen3-4B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) (Apache-2.0), with greedy decoding via `mlx-lm` 0.32.2 on an Apple M5 (16 GB). No cloud APIs were used.

### Score distribution

| Score | groundedness (all / kept) | relevance (all / kept) | standalone (all / kept) |
|---|---|---|---|
| 5 | 65 / 54 | 117 / 96 | 177 / 134 |
| 4 | 97 / 80 | 53 / 38 | 0 / 0 |
| 3 | 13 / – | 15 / – | 15 / – |
| 2 | 16 / – | 7 / – | 0 / – |
| 1 | 1 / – | 0 / – | 0 / – |

## Fields

| Field | Description |
|---|---|
| `question` | Generated question |
| `question_compressed` | Shorter, more indirect rewrite (for robustness testing) |
| `answer` | Reference answer generated from the chunk only |
| `context` | Sentences from the chunk that support the answer, as extracted by the LLM |
| `source_chunk` | Full source chunk the question was generated from |
| `source` | Citation: CFR section (e.g. `45 CFR 164.408`) or FDA guidance page |
| `source_url` | Download URL of the source document |
| `groundedness_score`, `relevance_score`, `standalone_score` | Critique scores, 1–5 |
| `*_reasoning` | One-sentence critique rationale |

`data/unfiltered.jsonl` in the generating repo has all 192 rows, before filtering.

## Limitations

- **Synthetic.** Questions, answers and scores were all produced by a 4B-parameter model, and the same model generated and judged them. Scores are lenient: no question scored below 3 on standalone, and the filter mostly removes weakly grounded or low-relevance items. Spot-check the data before relying on it.
- **Not legal advice.** Answers paraphrase regulatory text and may be incomplete or out of date. The CFR snapshot is from 2025-10-01 and does not include later amendments, such as any finalized HIPAA Security Rule update.
- **No PHI.** All inputs are public regulatory text. The dataset contains no real patient data or other personal information.
- **Coverage.** Only 192 of 345 chunks were used, so some sections of Part 164 and of the FDA guidance have no questions. Each chunk produces a single question, so the questions are single-hop.
- `context` is extracted by the LLM. It is usually verbatim but is not guaranteed to be.

## Reproduce

```sh
git clone -b mac-local https://github.com/solvren-ai/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems.git
cd generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems
uv run local/run_local.py --max-chunks 192
```
