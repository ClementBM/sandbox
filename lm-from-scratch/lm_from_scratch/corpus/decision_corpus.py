from artefacts import DECISION_CORPUS
import pandas as pd
import re

SENTENCE_DELIMITER = (
    r"(?<!Mme)(?<!cf)(?<!Bull)(?<!concl)(?<!\.\.)(?<=[^A-Z])(?:\.|;)(?=\s[^\d]|\\n)"
)


class DecisionCorpus:
    def __init__(self) -> None:
        self.df = pd.read_json(DECISION_CORPUS, lines=True)

    def get_text(self):
        return self.df["text"]

    def get_sentence_pairs(self):
        sentences_pairs = []
        for text in self.get_text():
            sentences_pairs += self._sentence_parser(text)
        return sentences_pairs

    def _sentence_parser(self, text):
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
