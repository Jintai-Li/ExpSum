# CodeSearchNet v2 Dataset Downloader

This script automatically downloads and extracts the **CodeSearchNet v2** datasets for **Python** and **Java**.

## Supported Datasets

We used two datasets in our work, including:

- Python
- Java

**Source URL:**
https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/

---

## Usage

### 1. Requirements

- Python ≥ 3.7
- Internet connection

No additional dependencies are required.

---

### 2. Download and Extract

Run the following command from the project root:

```bash
python download_codesearchnet.py
```

By default, the datasets will be downloaded and extracted to:

```
Benchmarks/CodeSearchNet/
├── python/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── java/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
```

### 3. Specify Output Directory

```
python download_codesearchnet.py --output_dir /path/to/your/data
```

### 4. Remove Zip Files After Extraction (Optional)

```
python download_codesearchnet.py --remove_zip
```

### Reproducibility Note

This script is intended to support reproducible experiments by automatically retrieving the official CodeSearchNet-v2 datasets used in prior work.