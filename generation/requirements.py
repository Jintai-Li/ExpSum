"""
Project dependencies for the comment generation pipeline.

This file lists all third-party Python packages required to run the codebase.
It is intended for documentation, dependency checking, or programmatic export
to requirements.txt.
"""

REQUIREMENTS = [
    # Core runtime
    "python>=3.8",

    # LLM client
    "openai>=1.0.0",

    # Progress bar
    "tqdm>=4.60.0",

    # Concurrency (standard library, listed for clarity)
    # concurrent.futures

    # Knowledge base dependencies
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",

    # Optional (used implicitly by some environments)
    "pathlib; python_version<'3.4'"
]


def export_requirements_txt(path: str = "requirements.txt"):
    """
    Export dependencies to a requirements.txt file.
    """
    with open(path, "w", encoding="utf-8") as f:
        for req in REQUIREMENTS:
            if not req.startswith("#"):
                f.write(req + "\n")
