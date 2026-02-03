# ExpSum: Code Summarization Aligned with Developer Expectations on Industrial Documentation

This repository contains the implementation and experimental resources for **ExpSum**, an expectations-aware approach for automated code summarization designed to generate *ready-to-use* summaries with high acceptance in industrial settings.

---

## Environment

Python Version: Python 3.9

You may install dependencies using: 

```bash
pip install -r requirements.txt
```

## Datasets

### Industrial Dataset (HarmonyOS)

We release the two HMSum benchmarks used in our work in the directory ./benchmarks. 

See ./benchmarks/HMSum-12 (13)/README.md for details. 

We provide benchmarks in **.xlsx format** so that the data can be read directly.

- Description:
  - Function–code summary pairs collected from HarmonyOS.
  - All reference summaries have passed an **industry-grade review process**.
- Size:
  - HMSum-12: 22,138 function-summary pairs.
  -  HMSum-13: 23,003 function-summary pairs.

------

### Open-Source / Community Projects

We incorporate two open-source benchmarks: CodeSearchNet and a C\C++ benchmark, SFT dataset released by the ProConSuL project.

These datasets are publicly available on GitHub and can be downloaded automatically using the provided scripts.

```bash
cd ./dataset
python download_codesearchnet.py
python download_proconsul_dataset.py --output_dir <TARGET_DIRECTORY>
```

------

## Data Sampling for Open-Source / Industrial Projects

To ensure representativeness and comparability, we apply a controlled sampling strategy when selecting data from open-source and industrial projects.

- We sample from ten widely used, large-scale open-source industrial projects, including: 

  ```
  RxJava、Airflow、Django、Flask、Spring Framework、TensorFlow、Kubernetes、React、Vue and FastAPI
  ```

- Code for sampling:

  ```bash
  cd ./expectation_and_generalizability_verify
  export GITHUB_TOKEN=your_personal_access_token
  pip install PyGithub
  python collect_open_source_functions.py
  ```

------

## Generation Pipeline

ExpSum generates code summaries through a four-stage pipeline that incorporates code modeling, domain knowledge retrieval, and constraint-driven self-refinement.

We also provide the **domain knowledge base** extracted from HarmonyOS 12, in the directory ./generation/knowledge_base

### Running Code Summary Generation

The main entry point for ExpSum is:

```bash
cd ./generation
python generation_pipeline.py \
  --input_file path/to/input.json \
  --output_file path/to/output.txt
```

#### Arguments

- `--input_file`
   Path to the input JSON file containing code functions and related metadata.
- `--output_file`
   Path to the output file where generated code summaries will be saved.

------

## Evaluation

We evaluate ExpSum from both **automatic** and **human-centered** perspectives.

### Automatic Evaluation

Metrics used include:

- BLEU-4 (used for RQ1-RQ4)

  ```bash
  python bleu.py \
    --reference_file reference.txt \
    --prediction_file prediction.txt
  ```

- ROUGE-L (used for RQ1-RQ4)

  ```bash
  python rouge.py \
    --prediction_file reference.txt \
    --reference_file prediction.txt
  ```

- SentenceBERT cosine similarity (used for RQ1-RQ4)

  ```bash
  python bert_cosine.py \
    --prediction_file reference.txt \
    --reference_file prediction.txt
  ```

- LLM-as-a-judge with GPT-4 (used for RQ4)

  ```bash
  python llm-eval.py
  ```

------

### Expert Evaluation on HarmonyOS 21:

##### Review Setup

- Number of functions: 580
- Number of experts: 4, all from Documentation Department of HarmonyOS project.

##### Released Files

- human_eval_records.xlsx: aggregated statistics and examples for each type of code summaries.

  ------

## Figures

The directory `./figures` contains the PNG and PDF source files of the images used in the paper.
