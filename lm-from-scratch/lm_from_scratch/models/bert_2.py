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

    def __init__(self, head_size, embed_dim):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

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
        wei = wei.masked_fill(attn_mask[:, :, None] == 0, -1e11)  # (B, T, T)

        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_dim):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, embed_dim) for _ in range(num_heads)]
        )
        self.projection = nn.Linear(
            embed_dim, embed_dim
        )  # projection for residual connection

    def forward(self, x, attn_mask):
        out = torch.cat([h(x, attn_mask) for h in self.heads], dim=-1)
        out = self.projection(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),  # projection for residual connection
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, head_size, num_heads) -> None:
        super().__init__()

        self.multihead_attention = MultiHeadAttention(num_heads, head_size, embed_dim)
        self.ffwd = FeedForward(embed_dim)

    def forward(self, x, attn_mask):
        x = x + self.multihead_attention(x, attn_mask)
        x = x + self.ffwd(x)
        return x


class BERT(nn.Module):
    def __init__(
        self, vocab_size, n_segments, max_len, embed_dim, num_heads, dropout, n_layers
    ):
        super().__init__()

        self.embedding = BERTEmbedding(
            vocab_size, n_segments, max_len, embed_dim, dropout
        )

        head_size = embed_dim // num_heads

        self.multihead_attention = TransformerBlock(num_heads, head_size, embed_dim)
        self.lm_head = nn.Linear(head_size * num_heads, vocab_size)
        self.sentence_classifier = nn.Linear(head_size * num_heads, 2)

    def forward(self, sequence, segment, attn_mask, masked_pos):
        x = self.embedding(sequence, segment)  # B, T, embedding_size
        x = self.multihead_attention(x, attn_mask)  # B, T, head_size * num_heads

        # masked token prediction
        masked_pos = masked_pos[:, :, None].expand(-1, -1, x.size(-1))
        x_masked = torch.gather(x, dim=1, index=masked_pos)
        logits_lm = self.lm_head(x_masked)  # B, T, vocab_size

        # next sentence prediction, by first token(CLS)
        logits_clsf = self.sentence_classifier(x[:, 0, :])  # B, 2

        return logits_lm, logits_clsf
