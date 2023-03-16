import os

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, ZipFilePathPointer, concat

from naat_core.tokenizers import DocxTokenizer
from dstb.language.corpus import CorpusReaderBase

from naat_core.data import CORPUS_PATH


class DocxCorpusReader(CorpusReaderBase):
    corpus_view = StreamBackedCorpusView
    _summaries = None

    def __init__(self, word_tokenizer=DocxTokenizer(), encoding="utf8"):
        CorpusReader.__init__(self, str(CORPUS_PATH), r".*\.txt", encoding)

        for path in self.abspaths(self._fileids):
            if isinstance(path, ZipFilePathPointer):
                pass
            elif os.path.getsize(path) == 0:
                # Check that all user-created corpus files are non-empty.
                raise ValueError(f"File {path} is empty")

        self._word_tokenizer = word_tokenizer

    def words(self, fileids=None):
        """
        :return: the given file(s) as a list of words
            and punctuation symbols.
        :rtype: list(str)
        """
        return concat(
            [
                self.corpus_view(path, self._read_word_block, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

    def _read_word_block(self, stream):
        words = []
        for i in range(20):  # Read 20 lines at a time.
            words.extend(self._word_tokenizer.tokenize(stream.readline()))
        return words
