from tqdm import tqdm
from artifacts import DECISION_CORPUS
import pandas as pd
import re
import fr_core_news_sm

SENTENCE_DELIMITER = (
    r"(?<!Mme)(?<!cf)(?<!Bull)(?<!concl)(?<!\.\.)(?<=[^A-Z])(?:\.|;)(?=\s[^\d]|\\n)"
)


class DecisionCorpus:
    def __init__(self) -> None:
        self.df = pd.read_json(DECISION_CORPUS, lines=True)
        self.nlp = fr_core_news_sm.load()

        # fast sentence segmentation without dependency parses
        self.nlp.disable_pipe("parser")
        self.nlp.enable_pipe("senter")

    def get_text(self):
        return self.df["text"]

    def get_sentence_pairs(self):
        sentences_pairs = []
        for text in self.get_text():
            sentences_pairs += self._sentence_pair_parser(text)
        return sentences_pairs

    def get_sentences(self):
        sentences = []
        for text in tqdm(self.get_text()):
            sentences += list(self._sentence_parser(text))
        return sentences

    def get_spacy_sentences(self):
        sentences = []
        for text in tqdm(self.get_text()):
            sentences += list(self._spacy_sentence_parser(text))
        return sentences

    def _sentence_parser(self, text):
        cuts = [0]
        for match in re.finditer(SENTENCE_DELIMITER, text):
            cut = match.regs[0][1]
            if cut - cuts[-1] > 20:
                cuts.append(cut)

        cuts += [len(text)]

        sentence_pairs = []
        for i in range(1, len(cuts)):
            sentence = text[cuts[i - 1] : cuts[i]]
            sentence_pairs.append(sentence)
        return sentence_pairs

    def _sentence_pair_parser(self, text):
        cuts = [0]
        for match in re.finditer(SENTENCE_DELIMITER, text):
            cut = match.regs[0][1]
            if cut - cuts[-1] > 20:
                cuts.append(cut)

        cuts += [len(text)]

        sentence_pairs = []
        for i in range(2, len(cuts)):
            first_sentence = text[cuts[i - 2] : cuts[i - 1]]
            second_sentence = text[cuts[i - 1] : cuts[i]]

            sentence_pairs.append((first_sentence, second_sentence))

        return sentence_pairs

    def _spacy_sentence_parser(self, text):
        parsed_text = self.nlp(text)
        for sentence in parsed_text.sents:
            if len(sentence) > 10:
                yield sentence.text
