import argparse

from naat.data import ROOT_PATH
from naat.data import preprocess_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="File Preparation CLI")

    parser.add_argument("-c", "--copy-to", help="Copy to")
    args = parser.parse_args()

    if args.copy_to:
        preprocess_dataset(ROOT_PATH, copy_to=args.copy_to)
