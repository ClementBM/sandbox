from naat.corpus import DocxCorpusReader
from nltk.text import Text


def test_docx_corpusreader():
    corpus = DocxCorpusReader()
    assert len(corpus._fileids) > 0

    text = Text(corpus.words(corpus._fileids[0]))
    assert text
