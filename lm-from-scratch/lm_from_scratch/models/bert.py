import math
import torch
import torch.nn as nn
from torch.nn import functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, n_segments, max_len, embed_dim, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.segment_embedding_table = nn.Embedding(n_segments, embed_dim)
        self.position_embedding_table = nn.Embedding(max_len, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, segment):
        # T should be max_len
        B, T = x.shape

        embedding = (
            self.token_embedding_table(x)
            + self.segment_embedding_table(segment)
            + self.position_embedding_table(torch.arange(T, device=DEVICE))
        )
        # B, T, embed_dim
        embedding = self.drop(embedding)

        return self.norm(embedding)


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x, attn_mask):
        # input of size (batch, time-step, channels=embedding size)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape

        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        # Fills elements of self tensor with value where mask is one.
        wei = wei.masked_fill(attn_mask[:, :, None] == 0, -1e-11)  # (B, T, T)

        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(num_heads)])

    def forward(self, x, attn_mask):
        return torch.cat([h(x, attn_mask) for h in self.heads], dim=-1)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BERT(nn.Module):
    def __init__(
        self, vocab_size, n_segments, max_len, embed_dim, num_heads, dropout, n_layers
    ):
        super().__init__()

        self.embedding = BERTEmbedding(
            vocab_size, n_segments, max_len, embed_dim, dropout
        )

        head_size = embed_dim // num_heads

        self.multi_head = MultiHeadAttention(num_heads, head_size, embed_dim)
        self.lm_head = nn.Linear(head_size * num_heads, vocab_size)
        self.sentence_classifier = nn.Linear(head_size * num_heads, 2)

    def forward(self, sequence, segment, attn_mask, masked_pos):
        x = self.embedding(sequence, segment)  # B, T, embedding_size
        x = self.multi_head(x, attn_mask)  # B, T, head_size * num_heads

        # masked token prediction
        masked_pos = masked_pos[:, :, None].expand(-1, -1, x.size(-1))
        x_masked = torch.gather(x, dim=1, index=masked_pos)
        logits_lm = self.lm_head(x_masked)  # B, T, vocab_size

        # next sentence prediction, by first token(CLS)
        logits_clsf = self.sentence_classifier(x[:, 0, :])  # B, 2

        return logits_lm, logits_clsf


# if __name__ == "__main__":
#     VOCAB_SIZE = 30000
#     N_SEGMENTS = 3
#     MAX_LEN = 512
#     EMBED_DIM = 768
#     N_LAYERS = 12
#     ATTN_HEADS = 12
#     DROPOUT = 0.1

#     sample_sequence = torch.randint(
#         high=VOCAB_SIZE,
#         size=[
#             MAX_LEN,
#         ],
#     )
#     sample_segment = torch.randint(
#         high=N_SEGMENTS,
#         size=[
#             MAX_LEN,
#         ],
#     )

#     embedding = BERTEmbedding(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, DROPOUT)
#     embedding_tensor = embedding(sample_sequence, sample_segment)

#     print("sequence", sample_sequence.size())
#     print("segment", sample_segment.size())
#     print(embedding_tensor.size())

#     bert = BERT(
#         VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT
#     )
#     out = bert(sample_seq, sample_seg)
#     print(out.size())
