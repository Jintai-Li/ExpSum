import argparse
import os
import zipfile
import urllib.request
from pathlib import Path

BASE_URL = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2"
LANGUAGES = ["python", "java"]


def download_file(url: str, dest_path: Path):
    print(f"[INFO] Downloading {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"[INFO] Saved to {dest_path}")


def extract_zip(zip_path: Path, extract_to: Path):
    print(f"[INFO] Extracting {zip_path}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[INFO] Extracted to {extract_to}")


def main(output_dir: Path, remove_zip: bool):
    for lang in LANGUAGES:
        zip_name = f"{lang}.zip"
        url = f"{BASE_URL}/{zip_name}"

        zip_path = output_dir / zip_name
        extract_path = output_dir / lang

        if not zip_path.exists():
            download_file(url, zip_path)
        else:
            print(f"[INFO] {zip_path} already exists, skipping download.")

        extract_zip(zip_path, extract_path)

        if remove_zip:
            zip_path.unlink()
            print(f"[INFO] Removed zip file {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract CodeSearchNet v2 datasets (Python & Java)."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("resources/data"),
        help="Directory to store downloaded and extracted datasets (default: resources/data)",
    )
    parser.add_argument(
        "--remove_zip",
        action="store_true",
        help="Remove zip files after extraction",
    )

    args = parser.parse_args()
    main(args.output_dir, args.remove_zip)
