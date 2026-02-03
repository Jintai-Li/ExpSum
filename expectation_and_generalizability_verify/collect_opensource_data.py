import os
import re
import json
from github import Github

MAX_FUNCTIONS_PER_REPO = 20
SUPPORTED_EXTENSIONS = (".py", ".java")

PROJECTS = [
    ("RxJava", "ReactiveX/RxJava"),
    ("Airflow", "apache/airflow"),
    ("Django", "django/django"),
    ("Flask", "pallets/flask"),
    ("Spring", "spring-projects/spring-framework"),
    ("TensorFlow", "tensorflow/tensorflow"),
    ("Kubernetes", "kubernetes/kubernetes"),
    ("React", "facebook/react"),
    ("Vue", "vuejs/vue"),
    ("FastAPI", "tiangolo/fastapi"),
]

CORE_DIR_KEYWORDS = [
    "src", "main", "core", "airflow", "django", "fastapi"
]

def is_core_dir(path: str) -> bool:
    return any(k in path.lower() for k in CORE_DIR_KEYWORDS)


def extract_functions_with_comments(code: str, lang: str):
    """
    Extract functions that contain documentation comments.
    Designed for sampling and manual inspection rather than full parsing.
    """
    results = []

    if lang == "python":
        pattern = re.compile(
            r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')\s*def\s+[\s\S]*?:',
            re.MULTILINE
        )
    else:  # Java
        pattern = re.compile(
            r'/\*\*[\s\S]*?\*/\s*(public|protected|private)?[\s\S]*?\{',
            re.MULTILINE
        )

    for match in pattern.finditer(code):
        results.append(match.group(0).strip())

    return results


def collect_from_repo(repo, project_name):
    collected = []

    def dfs(path=""):
        nonlocal collected
        if len(collected) >= MAX_FUNCTIONS_PER_REPO:
            return

        try:
            contents = repo.get_contents(path)
        except Exception:
            return

        if not isinstance(contents, list):
            contents = [contents]

        for item in contents:
            if len(collected) >= MAX_FUNCTIONS_PER_REPO:
                return

            if item.type == "dir":
                if is_core_dir(item.path):
                    dfs(item.path)

            elif item.type == "file" and item.path.endswith(SUPPORTED_EXTENSIONS):
                try:
                    code = item.decoded_content.decode("utf-8", errors="ignore")
                except Exception:
                    continue

                lang = "python" if item.path.endswith(".py") else "java"
                functions = extract_functions_with_comments(code, lang)

                for func in functions:
                    if len(collected) >= MAX_FUNCTIONS_PER_REPO:
                        return
                    collected.append({
                        "project": project_name,
                        "project_url": repo.html_url,
                        "file_path": item.path,
                        "function_with_comment": func
                    })

    dfs()
    return collected


def main():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GitHub API token not found. Please set GITHUB_TOKEN."
        )

    github = Github(token)
    all_samples = []

    for project_name, repo_name in PROJECTS:
        print(f"[INFO] Sampling from {project_name} ({repo_name})")
        repo = github.get_repo(repo_name)
        samples = collect_from_repo(repo, project_name)
        print(f"[INFO] Collected {len(samples)} functions from {project_name}")
        all_samples.extend(samples)

    with open("open_source_function_samples.json", "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print("[DONE] Dataset saved to open_source_function_samples.json")


if __name__ == "__main__":
    main()
