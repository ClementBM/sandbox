from pathlib import Path

ROOT = Path(__file__).parent

EPUB_FILE = ROOT / "dice-10943.epub"
EPUB_TRANSFORMER = ROOT / "ebook-transformer.xslt"

RAW_CHAPTER_FILE = lambda x: ROOT / f"raw-chapter-{x}.xml"
FORMATED_CHAPTER_FILE = lambda x: ROOT / f"formated-chapter-{x}.xml"
