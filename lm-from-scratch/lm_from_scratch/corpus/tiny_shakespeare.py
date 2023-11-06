from artifacts import TINY_SHAKESPEARE_CORPUS


class TinyShakespeareCorpus:
    def __init__(self) -> None:
        with open(TINY_SHAKESPEARE_CORPUS, "r", encoding="utf-8") as f:
            self.text = f.read()
