import torch
from lm_from_scratch.models.t5 import (
    FeedForward,
    Residual,
    T5LayerNorm,
    PreNorm,
    T5RelativePositionBias,
    T5SelfAttention,
    T5CrossAttention,
    T5Encoder,
    T5Decoder,
    T5,
)

EMBEDDING_SIZE = 126
BATCH_SIZE = 32
MAX_LENGTH = 8
NUM_HEADS = 6
VOCAB_SIZE = 1000


def test_residual():
    residual_layer = Residual(lambda x: x)

    sample_sequence = torch.randint(
        high=EMBEDDING_SIZE,
        size=[BATCH_SIZE, MAX_LENGTH],
    )

    out = residual_layer(sample_sequence)
    assert out.shape == sample_sequence.shape


def test_t5layernorm():
    layernorm_layer = T5LayerNorm(dim=EMBEDDING_SIZE)

    sample_sequence = torch.rand(
        size=[BATCH_SIZE, MAX_LENGTH, EMBEDDING_SIZE],
    )

    out = layernorm_layer(sample_sequence)

    assert out.shape == sample_sequence.shape
    assert next(layernorm_layer.buffers("beta")).shape[0] == EMBEDDING_SIZE
    assert next(layernorm_layer.parameters("gamma")).shape[0] == EMBEDDING_SIZE


def test_prenorm():
    prenorm_layer = PreNorm(dim=EMBEDDING_SIZE, fn=lambda x: x)

    sample_sequence = torch.rand(
        size=[BATCH_SIZE, MAX_LENGTH, EMBEDDING_SIZE],
    )

    out = prenorm_layer(sample_sequence)

    assert out.shape == sample_sequence.shape


def test_feedforward():
    feedforward_layer = FeedForward(EMBEDDING_SIZE)

    sample_sequence = torch.rand(
        size=[BATCH_SIZE, MAX_LENGTH, EMBEDDING_SIZE],
    )

    out = feedforward_layer(sample_sequence)

    assert out.shape == sample_sequence.shape


def test_relativepositionembedding():
    relativeposition_layer = T5RelativePositionBias(
        scale=1, causal=False, heads=NUM_HEADS
    )

    sample_sequence = torch.rand(
        size=[BATCH_SIZE, NUM_HEADS, EMBEDDING_SIZE, EMBEDDING_SIZE],
    )

    out = relativeposition_layer(sample_sequence)

    assert out.shape == sample_sequence.shape


def test_selfattention():
    dim_head = EMBEDDING_SIZE // NUM_HEADS
    selfattention_layer = T5SelfAttention(
        dim=EMBEDDING_SIZE,
        heads=NUM_HEADS,
        dim_head=dim_head,
        causal=False,
        dropout=0.0,
    )

    sample_sequence = torch.rand(
        size=[BATCH_SIZE, MAX_LENGTH, EMBEDDING_SIZE],
    )

    out = selfattention_layer(sample_sequence)

    assert out.shape == sample_sequence.shape


def test_crossattention():
    dim_head = EMBEDDING_SIZE // NUM_HEADS

    crossattention_layer = T5CrossAttention(
        dim=EMBEDDING_SIZE,
        context_dim=None,  # If None Then embedding_size
        heads=NUM_HEADS,
        dim_head=dim_head,
        dropout=0.0,
    )

    sample_sequence = torch.rand(
        size=[BATCH_SIZE, MAX_LENGTH, EMBEDDING_SIZE],
    )

    out = crossattention_layer(sample_sequence, None)

    assert out.shape == sample_sequence.shape


def test_t5encoder():
    encoder = T5Encoder(
        dim=EMBEDDING_SIZE,
        vocab_size=VOCAB_SIZE,
        depth=1,
        heads=NUM_HEADS,
        dim_head=EMBEDDING_SIZE // NUM_HEADS,
        mlp_mult=2,
        dropout=0.1,
    )

    sample_sequence = torch.randint(
        high=VOCAB_SIZE,
        size=[BATCH_SIZE, MAX_LENGTH],
    )

    out = encoder(sample_sequence)
    assert out.shape == torch.Size([BATCH_SIZE, MAX_LENGTH, EMBEDDING_SIZE])


def test_t5decoder():
    decoder = T5Decoder(
        dim=EMBEDDING_SIZE,
        vocab_size=VOCAB_SIZE,
        depth=1,
        heads=NUM_HEADS,
        dim_head=EMBEDDING_SIZE // NUM_HEADS,
        mlp_mult=2,
        dropout=0.1,
    )

    sample_sequence = torch.randint(
        high=VOCAB_SIZE,
        size=[BATCH_SIZE, MAX_LENGTH],
    )

    context = torch.rand([BATCH_SIZE, MAX_LENGTH, EMBEDDING_SIZE])

    out = decoder(sample_sequence, context)
    assert out.shape == torch.Size([BATCH_SIZE, MAX_LENGTH, EMBEDDING_SIZE])


def test_t5model():
    model = T5(
        dim=EMBEDDING_SIZE,
        vocab_size=VOCAB_SIZE,
        enc_depth=1,
        enc_heads=NUM_HEADS,
        enc_dim_head=EMBEDDING_SIZE // NUM_HEADS,
        enc_mlp_mult=4,
        dec_depth=1,
        dec_heads=NUM_HEADS,
        dec_dim_head=EMBEDDING_SIZE // NUM_HEADS,
        dec_mlp_mult=4,
        dropout=0.0,
    )

    src = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_LENGTH))
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_LENGTH))

    logits = model(src, target)
    logits.shape == torch.Size([BATCH_SIZE, MAX_LENGTH, VOCAB_SIZE])

    src_mask = torch.ones((BATCH_SIZE, MAX_LENGTH)).bool()
    logits = model(src, target, src_attn_mask=src_mask)
    logits.shape == torch.Size([BATCH_SIZE, MAX_LENGTH, VOCAB_SIZE])

    context_mask = torch.ones((BATCH_SIZE, MAX_LENGTH)).bool()
    logits = model(src, target, src_attn_mask=src_mask, target_attn_mask=context_mask)
    logits.shape == torch.Size([BATCH_SIZE, MAX_LENGTH, VOCAB_SIZE])
