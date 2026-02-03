import argparse
import os
import urllib.request


RAW_URL = (
    "https://raw.githubusercontent.com/"
    "trinity4ai/ProConSuL/main/datasets/SFT_dataset_before_obf.json"
)


def download_dataset(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "SFT_dataset_before_obf.json")
    print(f"Downloading dataset to: {output_path}")

    urllib.request.urlretrieve(RAW_URL, output_path)

    print("Download completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Download the ProConSuL SFT dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/proconsul",
        help="Directory to save the downloaded dataset (default: data/proconsul)"
    )
    args = parser.parse_args()

    download_dataset(args.output_dir)


if __name__ == "__main__":
    main()
