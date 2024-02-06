class CharLevelTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(list(set(text)))

        self.vocab_size = len(chars)

        self.string_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_string = {i: ch for i, ch in enumerate(chars)}

    def add_token(self, token):
        if token not in self.string_to_idx:
            self.idx_to_string[len(self.idx_to_string)] = token
            self.string_to_idx[token] = len(self.idx_to_string) - 1
        return self.string_to_idx[token]

    def encode(self, string_value):
        """
        Encoder: take a string, output a list of integers
        """
        return [self.string_to_idx[c] for c in string_value]

    def decode(self, idx_list):
        """
        decoder: take a list of integers, output a string
        """
        return "".join([self.idx_to_string[i] for i in idx_list])
