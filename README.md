# Generating Synthetic Datasets for Evaluating Retrieval Augmented Generation Systems

As Retrieval Augmented Generation (RAG) systems become more prevalent, evaluating their performance is essential to ensure quality and performance. However, collecting real-world data for evaluation can be costly and time-consuming, especially in the early stages of a project. This lab provides a hands-on guide to leveraging large language models and knowledge retrieval context to generate synthetic evaluation datasets that mimic real human interactions. It covers setting up an end-to-end workflow using Python and the Amazon Bedrock API.

By leveraging large language models and knowledge retrieval context, the proposed approach ensures that the synthetic datasets are diverse, realistic, and representative of real-world scenarios. This solution is relevant for developers and researchers working on RAG systems, as it streamlines the evaluation process and accelerates the iterative development cycle, ultimately leading to better-performing AI systems.

The notebook guides you through generating a synthetic dataset for a QA-RAG application using the Bedrock API, Python and Langchain. The notebook consists of the following chapters: 

1. Set-up of the environment
2. Loading and preparing context data
3. Initial Question Generation
4. Answer Generation
5. Extracting Relevant Context
6. Evolving Questions to fit end-users behaviour
7. Automated Dataset Generation
8. Assessing the Questions quality


## Getting started

### Choose a notebook environment

This lab is presented as a **Python notebook**, which you can run from the environment of your choice:

- [SageMaker Studio](https://aws.amazon.com/sagemaker/studio/) is a web-based integrated development environment (IDE) for machine learning. To get started quickly, refer to the [instructions for domain quick setup](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html).
- [SageMaker Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html) is a machine learning (ML) compute instance running the Jupyter Notebook App.
- To use your existing (local or other) notebook environment, make sure it has [credentials for calling AWS](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).


### Enable AWS IAM permissions for Bedrock

The AWS identity you assume from your notebook environment (which is the [*Studio/notebook Execution Role*](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) from SageMaker, or can be a role or IAM User for self-managed notebooks), must have sufficient [AWS IAM permissions](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html) to call the Amazon Bedrock service.

To grant Bedrock access to your identity:

- Open the [AWS IAM Console](https://us-east-1.console.aws.amazon.com/iam/home?#)
- Find your [Role](https://us-east-1.console.aws.amazon.com/iamv2/home?#/roles) (if using SageMaker or otherwise assuming an IAM Role), or else [User](https://us-east-1.console.aws.amazon.com/iamv2/home?#/users)
- Select *Add Permissions > Create Inline Policy* to attach new inline permissions, open the *JSON* editor and paste in the below example policy:

```
{
    "Version": "2012-10-17",
    "Statement": {
        "Sid": "AllowInference",
        "Effect": "Allow",
        "Action": [
            "bedrock:InvokeModel"
        ],
        "Resource": "arn:aws:bedrock:*::foundation-model/*"
    }
}
```

> ℹ️  **Note:** With Amazon SageMaker, your notebook execution role is typically be *separate* from the user or role that you log in to the AWS Console with. If you want to explore the AWS Console for Amazon Bedrock, you need to grant permissions to your Console user/role too. You can run the notebooks anywhere as long as you have access to the AWS Bedrock service and have appropriate credentials

For more information on the fine-grained action and resource permissions in Bedrock, check out the Bedrock [Developer Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html).

### Clone and use the notebooks

> ℹ️ **Note:** In SageMaker Studio, you can open a "System Terminal" to run these commands by clicking *File > New > Terminal*

Once your notebook environment is set up, clone this workshop repository into it.

```sh
git clone https://github.com/aws-samples/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems.git
cd generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems/Notebook 
```


You're now ready to explore the lab notebook! You will be guided through connection the notebook to Amazon Bedrock for large language model access.


## Run locally on Mac

`local/run_local.py` is a script port of the notebook that runs fully on Apple Silicon. It uses no AWS: a local [MLX](https://github.com/ml-explore/mlx-lm) model replaces Bedrock. It builds a RAG evaluation set over public US healthcare regulations instead of Amazon shareholder letters:

- [45 CFR Part 164](https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164) (HIPAA Security, Breach Notification and Privacy Rules), eCFR snapshot of 2026-10-06
- [FDA guidance: Cybersecurity in Medical Devices](https://www.fda.gov/media/119933/download) (Feb 2026)

The pipeline is the same as the notebook's: chunk (1500/100) → question → compressed question → answer → supporting sentences → critique scores → filter. The relevance critique uses a compliance officer / healthcare privacy analyst persona. A third critique, standalone, was added.

It runs in three stages, each in its own process so only one model is in memory at a time:

1. `gen`: `mlx-community/Qwen3-4B-Instruct-2507-4bit` generates one row per chunk and scores it (self-scores).
2. `judge`: `mlx-community/Qwen3-8B-4bit`, a different and larger model, re-scores every row.
3. `finalize`: merges the outputs, keeps only rows where all three **judge** scores are ≥ 4, and exports the full chunk corpus.

`gen` and `judge` append one line per chunk to `data/work/`. If a run dies (for example from a Metal out-of-memory error), rerun the same command and it continues from the last saved chunk.

Requirements: an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/), about 7 GB of disk for the two models and about 6 GB of free unified memory.

```sh
git clone -b mac-local https://github.com/solvren-ai/generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems.git
cd generating-synthetic-datasets-for-evaluating-retrieval-augmented-generation-systems
uv run local/run_local.py gen        # all chunks; rerun to resume
uv run local/run_local.py judge      # rerun to resume
uv run local/run_local.py finalize   # writes data/unfiltered.jsonl, data/test.jsonl, data/corpus.jsonl
# smoke test: add --max-chunks 3 to gen and judge
```

Options: `--model` (overrides the stage's default model), `--max-chunks`, `--min-score` (finalize, default 4).

Outputs:
- `source_docs/`: downloaded source files (untrusted data, gitignored)
- `data/work/`: resumable per-stage outputs (gitignored)
- `data/unfiltered.jsonl`: every row with judge scores, reasoning and self-scores
- `data/corpus.jsonl`: all chunks (retrieval corpus with distractors; joins on `chunk_id`)
- `data/test.jsonl`: filtered rows, Hugging Face-ready (see `DATASET_CARD.md`). Published privately as `SolvrenAI/hipaa-fda-rag-eval`.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License 
This lab is licensed under the MIT-0 License. 

