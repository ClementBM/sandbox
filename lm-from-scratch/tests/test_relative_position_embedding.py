import torch
from lm_from_scratch.models.t5 import T5RelativePositionBias


def test_relative_position():
    max_len = 10

    q_pos = torch.arange(max_len, dtype=torch.long)
    k_pos = torch.arange(max_len, dtype=torch.long)

    rel_pos = k_pos[None, :] - q_pos[:, None]

    re_biases = T5RelativePositionBias._relative_position_bucket(
        rel_pos, causal=False, num_buckets=6, max_distance=6
    )

    assert re_biases.shape == torch.Size((max_len, max_len))


def test_relative_position_embedding():
    max_len = 10
    i = max_len
    j = max_len

    q_pos = torch.arange(j - i, j, dtype=torch.long)
    k_pos = torch.arange(j, dtype=torch.long)

    rel_pos = k_pos[None, :] - q_pos[:, None]

    re_biases = T5RelativePositionBias._relative_position_bucket(
        rel_pos, causal=False, num_buckets=6, max_distance=6
    )

    embedding_table = torch.nn.Embedding(6, 2)
    output = embedding_table(re_biases)

    assert output.shape == torch.Size((max_len, max_len, 2))
