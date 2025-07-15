import torch
import torch.nn as nn
from torch.nn import functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, embed_dim, max_len):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        # buffer != not a parameter of the model
        self.register_buffer("tril", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x):
        # input of size (batch, time-step, channels=embedding size)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_dim, max_len):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, embed_dim, max_len) for _ in range(num_heads)]
        )
        self.projection = nn.Linear(
            embed_dim, embed_dim
        )  # projection for residual connection

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
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
    def __init__(self, embed_dim, head_size, num_heads, max_len) -> None:
        super().__init__()

        self.multihead_attention = MultiHeadAttention(
            num_heads, head_size, embed_dim, max_len
        )
        self.ffwd = FeedForward(embed_dim)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out = self.layer_norm_1(x)
        out = out + self.multihead_attention(out)

        out = self.layer_norm_2(out)
        out = out + self.ffwd(out)
        return out


class EmbeddingStem(nn.Module):

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(max_len, embed_dim)

    def forward(self, idx):
        B, T = idx.shape

        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=DEVICE)
        )

        return token_embedding + position_embedding  # B, T, embedding_size


class Gpt(nn.Module):
    def __init__(self, vocab_size, num_heads, head_size, embed_dim, max_len):
        super().__init__()

        self.embedding = EmbeddingStem(
            vocab_size=vocab_size, embed_dim=embed_dim, max_len=max_len
        )

        head_size = embed_dim // num_heads

        self.transformer_block = TransformerBlock(
            embed_dim=embed_dim,
            head_size=head_size,
            num_heads=num_heads,
            max_len=max_len,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(head_size * num_heads, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape

        x = self.embedding(idx)  # B, T, embedding_size
        x = self.transformer_block(x)  # B, T, head_size * num_heads
        x = self.layer_norm(x)
        logits = self.lm_head(x)  # B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
