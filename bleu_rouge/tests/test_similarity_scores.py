from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import math
import typing
from sacrebleu.metrics import BLEU
import numpy as np


class EvaluationSet:
    hypothesis: str
    references: typing.List[str]
    rouge_score: float

    def __init__(self, hypothesis, references, rouge_score: float, bleu_score: float):
        self.references = references
        self.hypothesis = hypothesis

        self.rouge_score = rouge_score
        self.bleu_score = bleu_score

    def tokenized_hypothesis(self):
        return word_tokenize(self.hypothesis)

    def tokenized_references(self):
        return [word_tokenize(reference) for reference in self.references]


evaluation_sets = [
    EvaluationSet(
        hypothesis="To make people trustworthy, you need to trust them.",
        references=["The way to make people trustworthy is to trust them."],
        rouge_score=0.55555,
        bleu_score=0.339325,
    ),
    # EvaluationSet(
    #     hypothesis="he began by starting a five person war cabinet and included chamberlain as lord president of the council",
    #     references=[
    #         "he began his premiership by forming a five-man war cabinet which included chamberlain as lord president of the council"
    #     ],
    #     rouge_score=0.75675,
    #     bleu_score=0.45057,
    # ),
    # EvaluationSet(
    #     hypothesis="the hello a cat dog fox jumps",
    #     references=["the fox jumps"],
    #     rouge_score=0.59999,
    #     bleu_score=0.145357,
    # ),
]


def test():
    tokenizer = RegexpTokenizer(r"\w+")

    w_n = 1 / 4
    hypothesis_tokens = tokenizer.tokenize(
        "To make people trustworthy, you need to trust them.".lower()
    )
    l_hyp = len(hypothesis_tokens)

    reference_tokens = tokenizer.tokenize(
        "The way to make people trustworthy is to trust them.".lower()
    )
    l_ref = len(reference_tokens)

    p_n = [
        7 / 9,
        5 / 8,
        3 / 7,
        1 / 6,
    ]

    bp = math.exp(1 - (l_ref / l_hyp))

    sum_pn = np.sum([math.log(pi) for pi in p_n])
    sum_pn *= w_n
    bleu = bp * math.exp(sum_pn)
    assert bleu

    hypothesis = " ".join(hypothesis_tokens)
    references = [" ".join(reference_tokens)]

    bleu_scorer = BLEU(
        # max_ngram_order=4,
        # tokenize="none",
    )
    score = bleu_scorer.sentence_score(
        hypothesis=hypothesis,
        references=references,
    )

    assert score.score


def test_rouge_score():
    rouge_scorer = Rouge()

    for evaluation_set in evaluation_sets:
        for reference in evaluation_set.references:
            score = rouge_scorer.get_scores(
                hyps=evaluation_set.hypothesis,
                refs=reference,
            )
            assert math.isclose(
                score[0]["rouge-l"]["f"], evaluation_set.rouge_score, rel_tol=1e-5
            )


def test_bleu_score():
    for evaluation_set in evaluation_sets:
        score = sentence_bleu(
            references=evaluation_set.tokenized_references(),
            hypothesis=evaluation_set.tokenized_hypothesis(),
        )
        assert math.isclose(score, evaluation_set.bleu_score, rel_tol=1e-5)


def test_sacrebleu_score():
    bleu = BLEU()
    for evaluation_set in evaluation_sets:
        score = bleu.sentence_score(
            hypothesis=evaluation_set.hypothesis,
            references=evaluation_set.references,
        )

        assert math.isclose(score.score / 100, evaluation_set.bleu_score, rel_tol=1e-5)
