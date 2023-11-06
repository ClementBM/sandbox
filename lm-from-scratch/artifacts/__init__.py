from pathlib import Path

DATASETS_PATH = Path(__file__).parent.absolute()

TINY_SHAKESPEARE_CORPUS = DATASETS_PATH / "tinyshakespeare.txt"

DECISION_CORPUS = DATASETS_PATH / "decision-corpus.jsonl"

TOKENIZER_PATH = DATASETS_PATH / "decision-tokenizer.json"

DECISION_CORPUS_RAW = DATASETS_PATH / "decision-raw.txt"
