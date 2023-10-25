import torch
import numpy as np
from lm_from_scratch.models.bert import BERT
from lm_from_scratch.corpus.decision_corpus import DecisionCorpus
import pandas as pd
from artefacts import TOKENIZER_PATH
from random import randint, random

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

VOCAB_SIZE = 1000
N_SEGMENTS = 2
MAX_LEN = 256  # 512 # what is the maximum context length for predictions?
EMBED_DIM = 128  # 768
N_LAYERS = 3
ATTN_HEADS = 4  # 32 * 4 = 128
DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_ITERS = 200
MAX_ITERS = 100
EVAL_INTERVAL = 10
LEARNING_RATE = 1e-3

BATCH_SIZE = 32  # how many independent sequences will we process in parallel?

MAX_SENTENCE_LEN = MAX_LEN // 2
MIN_SENTENCE_LEN = 10

# Corpus and tokenizer setup
corpus = DecisionCorpus()

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
tokenizer.pre_tokenizer = Whitespace()

tokenizer.train_from_iterator(corpus.get_text(), trainer)

# post-processing to traditional BERT inputs
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# pad the outputs to the longest sentence present
tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=MAX_LEN)

tokenizer.save(str(TOKENIZER_PATH))


# Load sentence pairs
sentences_pairs = corpus.get_sentence_pairs()

df = pd.DataFrame(sentences_pairs, columns=["sentence_1", "sentence_2_isnext"])


# Split dataset
# Train and test splits
sentence_pair_split = int(0.9 * len(df))

df_train = df[:sentence_pair_split]
df_eval = df[sentence_pair_split:].reset_index(drop=True)

train_col_1_shuffled = (
    df_train["sentence_2_isnext"]
    .sample(frac=1, random_state=212, replace=True)
    .reset_index(drop=True)
)
df_train["sentence_2_notnext"] = train_col_1_shuffled.values

sum(train_col_1_shuffled == df_train["sentence_2_isnext"]) == 0

eval_col_1_shuffled = (
    df_eval["sentence_2_isnext"]
    .sample(frac=1, random_state=23, replace=True)
    .reset_index(drop=True)
)
df_eval["sentence_2_notnext"] = eval_col_1_shuffled.values

sum(eval_col_1_shuffled == df_eval["sentence_2_isnext"]) == 0

sentence_pair_train_isnext = tokenizer.encode_batch(
    df_train[["sentence_1", "sentence_2_isnext"]].values
)

sentence_pair_train_notnext = tokenizer.encode_batch(
    df_train[["sentence_1", "sentence_2_notnext"]].values
)

sentence_pair_val_isnext = tokenizer.encode_batch(
    df_eval[["sentence_1", "sentence_2_isnext"]].values
)

sentence_pair_val_notnext = tokenizer.encode_batch(
    df_eval[["sentence_1", "sentence_2_isnext"]].values
)


def filter_by_sentence_length(sentence_pairs_isnext, sentence_pairs_notnext, max_len):
    pairs_isnext = []
    pairs_notnext = []

    for pair_isnext, pair_notnext in zip(sentence_pairs_isnext, sentence_pairs_notnext):
        if len(pair_isnext) > max_len or len(pair_isnext) < MIN_SENTENCE_LEN:
            continue
        if len(pair_notnext) > max_len or len(pair_notnext) < MIN_SENTENCE_LEN:
            continue
        pairs_isnext.append(pair_isnext)
        pairs_notnext.append(pair_notnext)

    return pairs_isnext, pairs_notnext


sentence_pair_train_data = filter_by_sentence_length(
    sentence_pair_train_isnext, sentence_pair_train_notnext, MAX_LEN
)

sentence_pair_val_data = filter_by_sentence_length(
    sentence_pair_val_isnext, sentence_pair_val_notnext, MAX_LEN
)


CLS_TOKEN_ID = 0
MASK_TOKEN_ID = 4


def get_sentence_pair_batch(split, batch_size=BATCH_SIZE):
    data_isnext, data_notnext = (
        sentence_pair_train_data if split == "train" else sentence_pair_val_data
    )
    pair_ix = torch.randint(len(data_isnext), (batch_size,))
    max_pred_count = len(data_isnext[0])

    for i, ix in enumerate(pair_ix):
        is_next = i % 2 == 0

        if ix > len(data_isnext) or ix > len(data_notnext):
            print(ix)
            print(len(data_isnext))
            print(len(data_notnext))

        sentence_pair = data_isnext[ix] if is_next else data_notnext[ix]

        available_mask = np.where(np.array(sentence_pair.special_tokens_mask) == 0)[0]
        pred_count = min(max_pred_count, max(1, round(len(available_mask) * 0.15)))

        masked_positions = np.random.choice(available_mask, pred_count, replace=False)
        masked_positions.sort()

        masked_tokens = np.array(sentence_pair.ids)[masked_positions]

        masked_token_ids = sentence_pair.ids.copy()
        for masked_position in masked_positions:
            if random() < 0.8:  # 80%
                masked_token_ids[masked_position] = MASK_TOKEN_ID
            elif random() < 0.5:  # 10%
                index = randint(5, VOCAB_SIZE - 1)  # random index in vocabulary
                masked_token_ids[masked_position] = index

        mask_padding = max_pred_count - len(masked_positions)

        yield [
            sentence_pair.ids,
            masked_token_ids,
            sentence_pair.type_ids,
            np.concatenate([masked_tokens, [CLS_TOKEN_ID] * mask_padding]),
            np.concatenate([masked_positions, [CLS_TOKEN_ID] * mask_padding]),
            sentence_pair.attention_mask,
            is_next,
        ]


model = BERT(
    vocab_size=VOCAB_SIZE,
    n_segments=N_SEGMENTS,
    max_len=MAX_LEN,
    embed_dim=EMBED_DIM,
    num_heads=ATTN_HEADS,
    dropout=DROPOUT,
    n_layers=N_LAYERS,
)
m = model.to(DEVICE)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        clsf_losses = torch.zeros(EVAL_ITERS)
        lm_losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            batch = map(
                lambda x: torch.tensor(x, device=DEVICE, dtype=torch.long),
                zip(*get_sentence_pair_batch("train", BATCH_SIZE)),
            )
            (
                _,
                token_ids,
                segment_ids,
                masked_tokens,
                masked_positions,
                attention_masks,
                is_next,
            ) = batch
            logits_lm, logits_clsf = model(
                token_ids, segment_ids, attention_masks, masked_positions
            )

            loss_lm = criterion(
                logits_lm.transpose(1, 2), masked_tokens
            )  # for masked LM
            loss_lm = (loss_lm.float()).mean()

            loss_clsf = criterion(logits_clsf, is_next)  # for sentence classification

            clsf_losses[k] = loss_clsf.item()
            lm_losses[k] = loss_lm.item()

        out[split] = (clsf_losses.mean(), lm_losses.mean())
    model.train()
    return out


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

for iter in range(MAX_ITERS):
    # every once in a while evaluate the loss on train and val sets
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train'][0]:.4f}|{losses['train'][1]:.4f},"
            + f"val loss {losses['val'][0]:.4f}|{losses['val'][1]:.4f}"
        )

    # sample a batch of data
    batch = map(
        lambda x: torch.tensor(x, device=DEVICE, dtype=torch.long),
        zip(*get_sentence_pair_batch("train", BATCH_SIZE)),
    )
    (
        _,
        token_ids,
        segment_ids,
        masked_tokens,
        masked_positions,
        attention_masks,
        is_next,
    ) = batch

    # evaluate the loss
    logits_lm, logits_clsf = model(
        token_ids, segment_ids, attention_masks, masked_positions
    )

    loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
    loss_lm = (loss_lm.float()).mean()
    loss_clsf = criterion(logits_clsf, is_next)  # for sentence classification
    loss = loss_lm + loss_clsf

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
