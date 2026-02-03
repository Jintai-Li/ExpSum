# Open-Source Function Sampling for ExpSum

This script collects documented functions from large open-source software projects to support generalization studies on the collected expectations.

## Target Projects
We sample from ten widely used, large-scale open-source industrial projects, including:

- RxJava
- Airflow
- Django
- Flask
- Spring Framework
- TensorFlow
- Kubernetes
- React
- Vue
- FastAPI

## Sampling Strategy
For each project, the script:
- Traverses core source directories only
- Extracts functions that contain documentation comments
- Collects up to 20 functions per project
- Stops traversal immediately once the quota is reached

This design ensures:
- Diversity across projects
- Reproducibility
- Efficient use of GitHub API resources

## GitHub API Authentication
GitHub imposes strict rate limits on unauthenticated API requests.
To ensure stable execution, **a GitHub personal access token is required**.

Please export your token before running the script:

```bash
export GITHUB_TOKEN=your_personal_access_token
```

A classic (read-only) token is sufficient.

## Installation

```json
pip install PyGithub
```

## Usage

```
python collect_open_source_functions.py
```

## Output

The script generates a file:

- `open_source_function_samples.json`

Each entry contains:

- Project name
- Project URL
- Source file path
- Function code with documentation comments

## Notes

We intentionally limit the number of sampled functions per project to avoid exhausting GitHub API resources and to maintain a balanced dataset suitable for manual inspection and expert evaluation.