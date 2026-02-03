# Dataset Preparation

This repository provides scripts for downloading and preparing datasets used in our evaluation.

---

## 1. Downloading the ProConSuL Dataset

We use the SFT dataset released by the ProConSuL project.  
The dataset is publicly available on GitHub and can be downloaded automatically using the provided script.

### Dataset Source

- File: `SFT_dataset_before_obf.json`
- Repository: https://github.com/trinity4ai/ProConSuL
- Original location: `datasets/SFT_dataset_before_obf.json`

### Download Script

We provide a Python script to download the dataset and save it to a user-specified directory.

#### Usage

```bash
python download_proconsul_dataset.py --output_dir <TARGET_DIRECTORY>
```

## 2. Preparing Evaluation References

After downloading the dataset, we preprocess it to extract:

- function-level code snippets, and
- corresponding reference summaries

These processed artifacts are used as inputs for automated and human evaluation.

### Preparation Script

The preprocessing step is handled by:

```
python prepare_reference_data.py \
  --input_json <input_file> \
  --output_refs <output_reference_file> \
  --output_codes <output_codes_file> \
  --max_samples <max_extract_numbers>
```

## 3. Directory Structure

```
project_root/
├── download_proconsul_dataset.py
├── prepare_eval_references.py
├── data/
│   └── proconsul/
│       └── SFT_dataset_before_obf.json
└── README.md

```

If you encounter any issues during dataset download or preparation, please ensure that:

- your network connection allows access to GitHub raw content, and
- you are using Python 3.7 or later.