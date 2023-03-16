# https://github.com/stelladk/PretrainingBERT
# http://nlp.polytechnique.fr/resources
import pandas as pd
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertModel,
    DataCollatorForLanguageModeling,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

SEQUENCE_LEN = 512  # maximum length of embeddings
TOKENIZER_PATH = None
TOKENIZER_FILES = None  # files of raw text
VOCAB_SIZE = 32000  # vocabulary size for tokenizer
MIN_FREQ = 2  # minimum term frequency for the vocabulary

MODEL_PATH = None  # path to save the pretrained model

HIDDEN_LAYERS = 12  # number of hidden layers of BERT model
HIDDEN_SIZE = 768  # hidden size of BERT model
ATTENTION_HEADS = 12  # number of attention heads of BERT model

# /media/clem/Back DD/test


class LegalDataset(Dataset):
    def __init__(self, text):
        self.encodings = text

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        item = {"input_ids": torch.tensor(self.encodings.iloc[index])}
        return item


def create_tokenizer(files, vocab_size, min_freq, max_len, save_path):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
    )
    tokenizer.save_model(save_path)
    tokenizer = ByteLevelBPETokenizer(
        save_path + "vocab.json",
        save_path + "merges.txt",
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=max_len)

    tokenizer.save(save_path + "tokenizer.json")


def process_text(filename, name, map_tokenize, encoding):
    with open(filename, "r", encoding=encoding) as file:
        text = file.readlines()  # list

    text = pd.Series(text)
    tqdm.pandas(desc="Tokenizing")
    text = text.progress_map(map_tokenize)
    dataset = LegalDataset(text)
    text = None
    occ = filename.rfind("/") + 1
    path = filename[:occ]
    torch.save(dataset, path + name + ".pt")
    return path + name + ".pt"


def train_bert():
    create_tokenizer(
        TOKENIZER_FILES, VOCAB_SIZE, MIN_FREQ, SEQUENCE_LEN, TOKENIZER_PATH
    )

    # Load Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH, max_len=SEQUENCE_LEN)

    # Create lamda tokenizing function
    def map_tokenize(text):
        return tokenizer.encode(text, max_length=SEQUENCE_LEN, truncation=True)

    # Process Text
    dataset_path = None
    # Load Dataset
    dataset = torch.load(dataset_path)

    # Create Masked Language Model
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )

    bert_config = BertConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=SEQUENCE_LEN,
        num_hidden_layers=HIDDEN_LAYERS,  # L
        hidden_size=HIDDEN_SIZE,  # H
        num_attention_heads=ATTENTION_HEADS,  # A
        type_vocab_size=1,
    )

    model = BertForMaskedLM(config=bert_config)

    training_args = TrainingArguments(
        output_dir=args.checkpoint,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_limit,
        prediction_loss_only=True,
        max_steps=args.max_steps,
        learning_rate=args.lrate,
        adam_beta1=args.b1,
        adam_beta2=args.b2,
        weight_decay=args.wdecay,
        lr_scheduler_type=args.scheduler,
        warmup_steps=args.warmup_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(MODEL_PATH)


def main():
    juribert_tokenizer = RobertaTokenizer.from_pretrained("path/to/extracted_tokenizer")

    juribert_model = BertModel.from_pretrained(
        "path/to/extracted_model", local_files_only=True
    )
