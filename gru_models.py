"""
Two architectures for the same binary sentiment classification task,
trained from scratch on identical data/vocab so the comparison is fair:

  1. GRUClassifier      - embedding -> bidirectional GRU -> linear head
  2. TransformerClassifier - embedding + positional encoding ->
                              Transformer encoder stack -> linear head

Both take padded token-id batches and a lengths tensor and return a
single logit per example (use BCEWithLogitsLoss).
"""
import math
import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, dropout=0.3, pad_idx=0, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * directions, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        # hidden: (num_layers * num_directions, batch, hidden_dim)
        if self.gru.bidirectional:
            last_fwd = hidden[-2]
            last_bwd = hidden[-1]
            final = torch.cat([last_fwd, last_bwd], dim=1)
        else:
            final = hidden[-1]
        return self.fc(self.dropout(final)).squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, ff_dim=256,
                 num_layers=2, dropout=0.3, pad_idx=0, max_len=300):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x, lengths):
        pad_mask = (x == self.pad_idx)  # True where padded -> ignored by attention
        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_encoding(embedded)
        encoded = self.encoder(embedded, src_key_padding_mask=pad_mask)

        # Mean-pool over real (non-pad) tokens instead of just taking token 0,
        # since we have no learned [CLS] token in this from-scratch model.
        mask = (~pad_mask).unsqueeze(-1).float()
        summed = (encoded * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / counts
        return self.fc(self.dropout(pooled)).squeeze(1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
