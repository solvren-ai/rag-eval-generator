# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx-lm==0.32.2", "pypdf>=5", "langchain-text-splitters>=0.3"]
# ///
"""Local (Apple Silicon, no AWS) port of the notebook in ../Notebook.

Same pipeline: chunk -> question -> compressed question -> answer -> source sentences
-> critique scores (groundedness, relevance, + standalone) -> filter.
Stages (each resumable; rerun after an OOM and it continues from the last saved chunk):
  gen      Qwen3-4B generates rows (+ self-critique scores)  -> data/work/gen.jsonl
  judge    Qwen3-8B re-scores every row independently       -> data/work/judge.jsonl
  finalize merge -> data/unfiltered.jsonl, filter on judge scores -> data/test.jsonl,
           all chunks -> data/corpus.jsonl (retrieval corpus, joins on chunk_id)
Bedrock Claude 3 Haiku is replaced by a local MLX model; Amazon shareholder letters
are replaced by public US healthcare regulations. Chunks are processed one at a time.
"""
import argparse, gzip, json, re, time, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "source_docs"  # downloaded files = untrusted data, kept apart from code
OUT_DIR = ROOT / "data"

SOURCES = {
    # HIPAA Privacy, Security and Breach Notification Rules (45 CFR Part 164), eCFR
    "ecfr_45cfr164_2026-10-06.xml": ("45 CFR Part 164 (HIPAA Privacy/Security/Breach Rules)",
                                     "https://www.ecfr.gov/api/versioner/v1/full/2026-10-06/title-45.xml?part=164"),
    # FDA guidance: Cybersecurity in Medical Devices (Feb 2026)
    "fda_cybersecurity_medical_devices.pdf": ("FDA Guidance: Cybersecurity in Medical Devices",
                                              "https://www.fda.gov/media/119933/download"),
}


def download():
    SRC_DIR.mkdir(exist_ok=True)
    for name, (_, url) in SOURCES.items():
        path = SRC_DIR / name
        if path.exists():
            continue
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
        path.write_bytes(body)


def load_documents():
    """Return [(text, source_label, url, cfr_subpart)] — one per CFR section / PDF page."""
    docs = []
    for name, (label, url) in SOURCES.items():
        path = SRC_DIR / name
        if name.endswith(".xml"):
            for subpart in ET.parse(path).getroot().iter("DIV6"):
                for sec in subpart.iter("DIV8"):
                    text = re.sub(r"[ \t]+", " ", "\n".join(" ".join(p.itertext()) for p in sec if p.tag in ("HEAD", "P", "FP")))
                    docs.append((text.strip(), f"45 CFR {sec.get('N')}", url, subpart.get("N")))
        else:
            from pypdf import PdfReader
            for i, page in enumerate(PdfReader(path).pages):
                docs.append((page.extract_text() or "", f"{label}, p.{i + 1}", url, None))
    return [d for d in docs if len(d[0]) > 200]


# ---------------------------------------------------------------- prompts
# Generation prompts are the notebook's, unchanged except for the domain wording.
QUESTION = """<Instructions>
Here is some context from a US healthcare regulation or regulatory guidance document:
<context>
{context}
</context>

Your task is to generate 1 question that a compliance officer or healthcare privacy analyst could ask, and that can be answered using the provided context, following these rules:

<rules>
1. The question should make sense to humans even when read without the given context.
2. The question should be fully answered from the given context.
3. The question should be framed from a part of context that contains important information. It can also be from tables, lists, etc.
4. The answer to the question should not contain any links.
5. The question should be of moderate difficulty.
6. The question must be reasonable and must be understood and responded by humans.
7. Do not use phrases like 'provided context', 'this section', 'this document', etc. in the question.
8. Avoid framing questions using the word "and" that can be decomposed into more than one question.
9. The question should not contain more than 20 words, make use of abbreviations (e.g. HIPAA, PHI, CE, BA) wherever possible.
</rules>

To generate the question, first identify the most important or relevant part of the context. Then frame a question around that part that satisfies all the rules above.

Output only the generated question with a "?" at the end, no other text or characters.
</Instructions>"""

ANSWER = """<Instructions>
<role>You are an experienced QA Engineer for building large language model applications.</role>
<task>It is your task to generate an answer to the following question <question>{question}</question> only based on the <context>{context}</context></task>
The output should be only the answer generated from the context.

<rules>
1. Only use the given context as a source for generating the answer.
2. Be as precise as possible with answering the question.
3. Be concise in answering the question and only answer the question at hand rather than adding extra information.
</rules>

Only output the generated answer as a sentence. No extra characters.
</Instructions>"""

SOURCE = """<Instructions>
Here is the context:
<context>
{context}
</context>

Your task is to extract the relevant sentences from the given context that can potentially help answer the following question. You are not allowed to make any changes to the sentences from the context.

<question>
{question}
</question>

Output only the relevant sentences you found, one sentence per line, without any extra characters or explanations.
</Instructions>"""

COMPRESS = """<Instructions>
<role>You are an experienced linguistics expert for building testsets for large language model applications.</role>

<task>It is your task to rewrite the following question in a more indirect and compressed form, following these rules:

<rules>
1. Make the question more indirect
2. Make the question shorter
3. Use abbreviations if possible
</rules>

<question>
{question}
</question>

Your output should only be the rewritten question with a question mark "?" at the end. Do not provide any other explanation or text.
</task>
</Instructions>"""

GROUNDEDNESS = """<Instructions>
You will be given a context and a question related to that context.

Your task is to provide an evaluation of how well the given question can be answered using only the information provided in the context. Rate this on a scale from 1 to 5, where:

1 = The question cannot be answered at all based on the given context
2 = The context provides very little relevant information to answer the question
3 = The context provides some relevant information to partially answer the question
4 = The context provides substantial information to answer most aspects of the question
5 = The context provides all the information needed to fully and unambiguously answer the question

First, read through the provided context carefully:

<context>
{context}
</context>

Then read the question:

<question>
{question}
</question>

Evaluate how well you think the question can be answered using only the context information. Provide your reasoning in an <evaluation> section, explaining what relevant or missing information from the context led you to your evaluation score in only one sentence.

Provide your evaluation in the following format:

<rating>(Your rating from 1 to 5)</rating>

<evaluation>(Your evaluation and reasoning for the rating)</evaluation>
</Instructions>"""

# Persona rewritten: Wall Street analyst / Amazon letters -> compliance officer / healthcare privacy analyst.
RELEVANCE = """<Instructions>
You will be given a question related to US healthcare regulations (e.g. the HIPAA Privacy, Security and Breach Notification Rules in 45 CFR Part 164, or FDA medical device cybersecurity guidance). Your task is to evaluate how useful this question would be for a compliance officer or healthcare privacy analyst working at a covered entity, business associate or medical device manufacturer.

To evaluate the usefulness of the question, consider the following criteria:

1. Relevance: Is the question directly relevant to regulatory compliance, privacy or security work? Questions that are too broad or unrelated to this domain should receive a lower rating.

2. Practicality: Does the question address a practical problem a compliance officer might encounter, such as a disclosure decision, a required safeguard, a deadline, a documentation duty or a submission requirement? Theoretical or overly academic questions may be less useful.

3. Clarity: Is the question clear and well-defined? Ambiguous or vague questions are less useful.

4. Depth: Does the question require a substantive answer that demonstrates understanding of the regulatory requirements? Surface-level questions (e.g. only asking for a definition or a section title) may be less useful.

5. Applicability: Would answering this question help with real-world compliance tasks such as policy writing, risk analysis, audits or incident response? Questions with limited applicability should receive a lower rating.

Provide your evaluation in the following format:

<rating>(Your rating from 1 to 5)</rating>

<evaluation>(Your evaluation and reasoning for the rating in one sentence)</evaluation>

Here is an example:
<evaluation>The question is very relevant to the persona because it asks when a covered entity must notify individuals after a breach of unsecured PHI</evaluation>
<rating>5</rating>

Here is the question:

{question}
</Instructions>"""

# Added (not in the original notebook): the question rules require context-independence; this scores it.
STANDALONE = """<Instructions>
You will be given a question. Your task is to evaluate how context-independent it is, i.e. whether a compliance professional with access to the relevant regulations would understand it without seeing the specific passage it was written from. Rate this on a scale from 1 to 5, where:

1 = The question only makes sense with the original passage (e.g. refers to "this section", "the document", "the above")
3 = The question is mostly understandable but leaves important referents vague
5 = The question is fully self-contained; technical terms and abbreviations (e.g. HIPAA, PHI, CE, BA, FDA) are fine

Provide your evaluation in the following format:

<rating>(Your rating from 1 to 5)</rating>

<evaluation>(Your evaluation and reasoning for the rating in one sentence)</evaluation>

Here is the question:

{question}
</Instructions>"""


def rating(text):
    m = re.search(r"<rating>\s*\(?([1-5])", text)
    return int(m.group(1)) if m else None


def reasoning(text):
    m = re.search(r"<evaluation>(.*?)(?:</evaluation>|$)", text, re.S)
    return m.group(1).strip() if m else None


SCORES = ("groundedness", "relevance", "standalone")
WORK = OUT_DIR / "work"  # resumable per-stage outputs, one JSON line per chunk


def read_jsonl(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def critique_prompts(q, ctx):
    return [GROUNDEDNESS.format(question=q, context=ctx), RELEVANCE.format(question=q), STANDALONE.format(question=q)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "judge", "finalize"])
    ap.add_argument("--model", help="default: Qwen3-4B-Instruct-2507 for gen, Qwen3-8B for judge")
    ap.add_argument("--max-chunks", type=int, help="limit chunks (smoke tests); default all")
    ap.add_argument("--min-score", type=int, default=4, help="keep rows with all three judge scores >= this")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    gen_path, judge_path = WORK / "gen.jsonl", WORK / "judge.jsonl"

    if args.stage == "finalize":
        judged = {r["chunk_id"]: r for r in read_jsonl(judge_path)}
        rows = [{**r, **judged[r["chunk_id"]]} for r in read_jsonl(gen_path) if r["chunk_id"] in judged]
        keep = [r for r in rows if all((r[f"{k}_score"] or 0) >= args.min_score for k in SCORES)]
        corpus = [dict(chunk_id=r["chunk_id"], source=r["source"], source_url=r["source_url"], cfr_subpart=r["cfr_subpart"],
                       in_embedding_train=r["in_embedding_train"], text=r["source_chunk"])
                  for r in read_jsonl(gen_path)]  # every chunk = retrieval corpus incl. distractors
        for name, rs in (("unfiltered.jsonl", rows), ("test.jsonl", keep), ("corpus.jsonl", corpus)):
            with open(OUT_DIR / name, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rs)
        print(f"{len(rows)} generated+judged, {len(keep)} kept (all judge scores >= {args.min_score})")
        return

    from mlx_lm import batch_generate, load
    model_id = args.model or {"gen": "mlx-community/Qwen3-4B-Instruct-2507-4bit", "judge": "mlx-community/Qwen3-8B-4bit"}[args.stage]
    model, tok = load(model_id)

    def llm(prompts, max_tokens):  # greedy decoding; prompts of ONE chunk batched together
        ids = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in prompts]
        return [t.strip() for t in batch_generate(model, tok, ids, max_tokens=max_tokens).texts]

    if args.stage == "gen":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        download()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""])
        todo = [(f"{src}#{k}", c, src, url, sub) for text, src, url, sub in load_documents()
                for k, c in enumerate(c for c in splitter.split_text(text) if len(c) > 300)][: args.max_chunks]
    else:
        todo = read_jsonl(gen_path)[: args.max_chunks]
    done = {r["chunk_id"] for r in read_jsonl(gen_path if args.stage == "gen" else judge_path)}
    print(f"{model_id}: {len(todo)} chunks, {len(done)} already done (resuming)")

    t0 = time.time()
    with open(gen_path if args.stage == "gen" else judge_path, "a") as f:
        for i, item in enumerate(todo):
            if (item[0] if args.stage == "gen" else item["chunk_id"]) in done:
                continue
            if args.stage == "gen":
                cid, ctx, src, url, sub = item
                (q,) = llm([QUESTION.format(context=ctx)], 80)
                comp, ans, sent, *crit = llm([COMPRESS.format(question=q), ANSWER.format(question=q, context=ctx),
                                              SOURCE.format(question=q, context=ctx), *critique_prompts(q, ctx)],
                                             [64, 200, 400, 200, 200, 120])
                row = dict(chunk_id=cid, question=q, question_compressed=comp, answer=ans, context=sent, source_chunk=ctx,
                           source=src, source_url=url, cfr_subpart=sub, in_embedding_train=sub == "E",
                           generator_model=model_id, **{f"self_{k}_score": rating(c) for k, c in zip(SCORES, crit)})
            else:
                crit = llm(critique_prompts(item["question"], item["source_chunk"]), [200, 200, 120])
                row = dict(chunk_id=item["chunk_id"], judge_model=model_id,
                           **{f"{k}_score": rating(c) for k, c in zip(SCORES, crit)},
                           **{f"{k}_reasoning": reasoning(c) for k, c in zip(SCORES, crit)})
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"[{i + 1}/{len(todo)}] {time.time() - t0:.0f}s " + " ".join(f"{k.split('_')[-2][0].upper()}{v}" for k, v in row.items() if k.endswith("_score")))
    print(f"{args.stage} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
