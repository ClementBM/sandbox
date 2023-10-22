import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_index, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # (B,T,C) (batch (4), time (8), channel (65:vocab size))
        # logits probability of next tokens
        logits = self.token_embedding_table(token_index)

        if targets is None:
            loss = None
        else:
            # reshaping the logits
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            # calculate the cross entropy loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, token_index, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(token_index)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            next_token_index = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            token_index = torch.cat((token_index, next_token_index), dim=1)  # (B, T+1)

        return token_index
