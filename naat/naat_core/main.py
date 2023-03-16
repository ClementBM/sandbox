import argparse

from naat_core.data import CORPUS_PATH
from naat_core.data import preprocess_dataset


if __name__ == "__main__":
    # python naat/main.py -c test
    parser = argparse.ArgumentParser(description="File Preparation CLI")

    parser.add_argument("-c", "--copy-to", help="Copy to")
    args = parser.parse_args()

    if args.copy_to:
        preprocess_dataset(CORPUS_PATH, copy_to=args.copy_to)
