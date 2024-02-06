import torch
from torch import nn
import torch.nn.functional as F

import math

from einops import rearrange

# Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
# https://arxiv.org/abs/1910.10683

MAX_LEN = 64


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class T5LayerNorm(nn.Module):
    """
    Pre-normalization wrapper, they use LayerNorm without bias
    """

    def __init__(self, dim):
        super().__init__()
        # weight
        self.gamma = nn.Parameter(torch.ones(dim))  # a module parameter
        # bias
        self.register_buffer("beta", torch.zeros(dim))  # not a module parameter

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = T5LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # optional dropout
            nn.Linear(inner_dim, dim),  # projection for residual connection
        )

    def forward(self, x):
        return self.net(x)


class T5RelativePositionBias(nn.Module):
    """
    Relative position embeddings produce a different learned embedding according to
    the offset between the “key” and “query” being compared in the self-attention mechanism.

    We use a simplified form of position embeddings where each “embedding” is simply a
    scalar that is added to the corresponding logit used for computing the attention weights.

    For efficiency, we also share the position embedding parameters across all layers in our model,
    though within a given layer each attention head uses a different learned position embedding

    Typically, a fixed number of embeddings are learned, each corresponding to a range of
    possible key-query offsets. In this work, we use 32 embeddings for all of our models
    with ranges that increase in size logarithmically up to an offset of 128 beyond which
    we assign all relative positions to the same embedding.
    """

    def __init__(
        self, scale, causal=False, num_buckets=32, max_distance=MAX_LEN, heads=12
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, causal=True, num_buckets=32, max_distance=MAX_LEN
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> h i j")
        return qk_dots + (bias * self.scale)


class T5SelfAttention(nn.Module):
    def __init__(self, *, dim, heads=12, dim_head=64, causal=False, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.causal = causal

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.relative_position_bias = T5RelativePositionBias(
            scale=dim_head**-0.5, causal=causal, heads=heads
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # batch, seq_len, head_num, head_dim
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = q * self.scale

        sim = torch.einsum(
            "b h i d, b h j d -> b h i j", q, k
        )  # [Batch, Headnum, Nmaxlength, Nmaxlength]

        sim = self.relative_position_bias(sim)

        # mask
        mask_value = -torch.finfo(sim.dtype).max

        if attn_mask is not None:
            sim = sim.masked_fill_(attn_mask[:, None, :, None] == 0, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(
                j - i + 1
            )
            sim = sim.masked_fill(causal_mask, mask_value)

        # attention
        attn_weight = sim.softmax(dim=-1)
        attn_weight = self.dropout(attn_weight)

        # aggregate
        out = torch.einsum("b h i j, b h j d -> b h i d", attn_weight, v)

        # merge heads
        out = rearrange(out, "b h n d -> b n (h d)")

        # combine heads and linear output
        return self.to_out(out)


class T5CrossAttention(nn.Module):
    def __init__(self, *, dim, context_dim=None, heads=12, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, attn_mask=None, context_attn_mask=None):
        b, n, _, h = *x.shape, self.heads

        kv_input = default(context, x)

        q, k, v = self.to_q(x), self.to_k(kv_input), self.to_v(kv_input)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        # [Batch, HeadNum, N_target_maxlength, N_source_maxlength]

        # mask
        mask_value = -torch.finfo(sim.dtype).max  # -infinity

        if attn_mask is not None:
            sim = sim.masked_fill_(attn_mask[:, None, :, None] == 0, mask_value)

        if context_attn_mask is not None:
            sim = sim.masked_fill_(context_attn_mask[:, None, None, :] == 0, mask_value)

        # attention
        attn_weight = sim.softmax(dim=-1)
        attn_weight = self.dropout(attn_weight)

        # aggregate
        out = torch.einsum("b h i j, b h j d -> b h i d", attn_weight, v)
        # BatchSize, HeadNum, Nmaxlength, EmbeddingSize

        # merge heads
        out = rearrange(out, "b h n d -> b n (h d)")
        # BatchSize, Nmaxlength, HeadNum * EmbeddingSize

        # combine heads and linear output
        return self.to_out(out)


class T5Encoder(nn.Module):
    def __init__(
        self, *, dim, vocab_size, depth, heads=12, dim_head=64, mlp_mult=4, dropout=0.0
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)

        self.layer = nn.ModuleList([])
        self.self_attentions = []

        for _ in range(depth):
            self_attention = T5SelfAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                causal=False,
                dropout=dropout,
            )
            self.self_attentions.append(self_attention)

            self.layer.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Residual(self_attention)),
                        PreNorm(
                            dim,
                            Residual(
                                FeedForward(dim=dim, mult=mlp_mult, dropout=dropout),
                            ),
                        ),
                    ]
                )
            )

        self.final_norm = T5LayerNorm(dim)

    def forward(self, x, attn_mask=None):
        x = self.token_emb(x)

        for attn, mlp in self.layer:
            x = attn(x, attn_mask=attn_mask)
            x = mlp(x)

        x = self.final_norm(x)

        return x


class T5Decoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vocab_size,
        depth,
        heads=12,
        dim_head=64,
        mlp_mult=4,
        dropout=0.0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)

        self.layer = nn.ModuleList([])
        self.self_attentions = []
        for _ in range(depth):
            self_attention = T5SelfAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                causal=True,
                dropout=dropout,
            )
            self.self_attentions.append(self_attention)

            self.layer.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Residual(self_attention)),
                        PreNorm(
                            dim,
                            Residual(
                                T5CrossAttention(
                                    dim=dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=dropout,
                                ),
                            ),
                        ),
                        PreNorm(
                            dim,
                            Residual(
                                FeedForward(dim=dim, mult=mlp_mult, dropout=dropout),
                            ),
                        ),
                    ]
                )
            )

        self.final_norm = T5LayerNorm(dim)

    def forward(self, x, context, attn_mask=None, context_attn_mask=None):
        x = self.token_emb(x)

        for self_attn, cross_attn, mlp in self.layer:
            x = self_attn(x, attn_mask=attn_mask)
            x = cross_attn(
                x,
                context=context,
                attn_mask=attn_mask,
                context_attn_mask=context_attn_mask,
            )

            x = mlp(x)

        x = self.final_norm(x)

        return x


class T5(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vocab_size,
        enc_depth,
        enc_heads,
        enc_dim_head,
        enc_mlp_mult,
        dec_depth,
        dec_heads,
        dec_dim_head,
        dec_mlp_mult,
        dropout=0.0,
    ):
        super().__init__()

        self.encoder = T5Encoder(
            dim=dim,
            vocab_size=vocab_size,
            depth=enc_depth,
            heads=enc_heads,
            dim_head=enc_dim_head,
            mlp_mult=enc_mlp_mult,
            dropout=dropout,
        )

        self.decoder = T5Decoder(
            dim=dim,
            vocab_size=vocab_size,
            depth=dec_depth,
            heads=dec_heads,
            dim_head=dec_dim_head,
            mlp_mult=dec_mlp_mult,
            dropout=dropout,
        )

        self.to_logits = nn.Linear(dim, vocab_size)

        # tie weights
        self.encoder.token_emb.weight = self.decoder.token_emb.weight

        # tie relative positional bias
        self.encoder.self_attentions[
            0
        ].relative_position_bias.relative_attention_bias.weight = self.decoder.self_attentions[
            0
        ].relative_position_bias.relative_attention_bias.weight

    def forward(self, src, target, src_attn_mask=None, target_attn_mask=None):
        x = self.encoder(src, attn_mask=src_attn_mask)

        x = self.decoder(
            x=target,
            context=x,
            attn_mask=target_attn_mask,
            context_attn_mask=src_attn_mask,
        )
        x = self.to_logits(x)

        return x
