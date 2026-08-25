"""
Data loading, cleaning, tokenization and vocabulary building for the
GRU vs Transformer sentiment classification comparison.

Dataset: IMDB 50k Movie Reviews (balanced, binary sentiment).
"""
import re
import random
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"


def clean_text(text: str) -> str:
    """Lowercase, strip HTML break tags left over from the IMDB scrape, and
    keep only letters/basic punctuation before tokenizing."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str):
    return clean_text(text).split()


def load_imdb(csv_path: str, max_samples: int = None):
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.sample(n=max_samples, random_state=SEED).reset_index(drop=True)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    df["tokens"] = df["review"].apply(tokenize)
    return df[["tokens", "label"]]


def build_vocab(token_lists, max_vocab_size=25000, min_freq=2):
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.most_common(max_vocab_size):
        if freq < min_freq:
            continue
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def numericalize(tokens, vocab, max_len):
    unk = vocab[UNK_TOKEN]
    ids = [vocab.get(t, unk) for t in tokens[:max_len]]
    return ids


class IMDBDataset(Dataset):
    """Returns (token_id_tensor, label) pairs. Sequences are left un-padded
    here; padding is done per-batch by the collate function so we don't
    waste memory padding every sequence to the global max length."""

    def __init__(self, df, vocab, max_len=300):
        self.vocab = vocab
        self.max_len = max_len
        self.samples = [
            (numericalize(toks, vocab, max_len), label)
            for toks, label in zip(df["tokens"], df["label"])
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, label = self.samples[idx]
        if len(ids) == 0:
            ids = [self.vocab[UNK_TOKEN]]
        return torch.tensor(ids, dtype=torch.long), label


def make_collate_fn(pad_idx):
    def collate(batch):
        seqs, labels = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        padded = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
        labels = torch.tensor(labels, dtype=torch.float)
        return padded, lengths, labels
    return collate


def split_df(df, train_frac=0.7, val_frac=0.15, seed=SEED):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    return train_df, val_df, test_df
