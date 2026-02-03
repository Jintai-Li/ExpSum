import argparse
import json
import re
from typing import Dict, List


def extract_doc_and_code(
    input_json_path: str,
    output_txt_path: str,
    output_json_path: str,
    max_samples: int = 200
) -> None:
    """
    Extract reference summaries and corresponding function code from a dataset.

    For each entry in the input JSON:
      - If both `doc` and `code` fields are non-empty:
          * The documentation string is normalized into a single line and
            written to a TXT file (one summary per line).
          * The function code is stored as a JSON object in a list.

    Parameters
    ----------
    input_json_path : str
        Path to the input JSON dataset.
    output_txt_path : str
        Path to the output TXT file containing reference summaries.
    output_json_path : str
        Path to the output JSON file containing function code entries.
    max_samples : int, optional
        Maximum number of (doc, code) pairs to extract (default: 200).
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data: Dict = json.load(f)

    docs: List[str] = []
    codes: List[Dict[str, str]] = []

    # The dataset is expected to have a top-level "test" split
    test_data = data.get("test", {})

    for _, entry in test_data.items():
        doc = entry.get("doc", "").strip()
        code = entry.get("code", "").strip()

        if not doc or not code:
            continue

        # Normalize documentation to a single line
        doc_one_line = re.sub(r"[\r\n]+", " ", doc).strip()

        docs.append(doc_one_line)
        codes.append({"code": code})

        if len(docs) >= max_samples:
            break

    # Write reference summaries
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d + "\n")

    # Write function code as a valid JSON array
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(codes, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare reference summaries and function code for evaluation."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to the input dataset JSON file."
    )
    parser.add_argument(
        "--output_refs",
        type=str,
        required=True,
        help="Path to the output TXT file for reference summaries."
    )
    parser.add_argument(
        "--output_codes",
        type=str,
        required=True,
        help="Path to the output JSON file for function code."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum number of samples to extract (default: 200)."
    )

    args = parser.parse_args()

    extract_doc_and_code(
        input_json_path=args.input_json,
        output_txt_path=args.output_refs,
        output_json_path=args.output_codes,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
