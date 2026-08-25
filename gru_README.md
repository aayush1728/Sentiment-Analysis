# GRU vs. Transformer: A Controlled Comparison for Sentiment Classification

This project extends `baseline_analytics.py` (BlogIntel) — a lexicon-based
sentiment/readability tool — by replacing the word-counting approach with two
sequence models trained **from scratch** on the same data, letting us
directly compare **recurrence (GRU)** against **self-attention (Transformer
encoder)** for binary sentiment classification.

Both architectures are implemented from first principles in PyTorch (no
pretrained weights), trained under **identical conditions** — same
vocabulary, train/val/test split, optimizer, learning rate, batch size and
number of epochs — so any difference in the results reflects the
architecture itself, not confounding factors.

## Dataset

[IMDB 50k Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)
(Maas et al., 2011) — 25,000 positive / 25,000 negative reviews, a standard
sentiment-classification benchmark. See `gru_IMDB-Dataset.csv`.

## Models

| | GRU | Transformer |
|---|---|---|
| Core mechanism | Bidirectional GRU, sequential recurrence | Multi-head self-attention, fully parallel over the sequence |
| Positional info | Implicit via recurrence order | Explicit sinusoidal positional encoding |
| Pooling | Final hidden state (both directions concatenated) | Mean-pooling over non-pad token representations |
| Regularization | Dropout, gradient clipping | Dropout, gradient clipping |

See `gru_models.py` for full implementation details.

## Results

Run on a 6,000-review subset (CPU, 4 epochs — see "Reproducing / scaling up"
below for the full 50k-review, GPU-scale setup):

| Model | Params | Train time (s) | Test Acc | Test F1 | Test AUC | Latency (ms/ex) |
|---|---|---|---|---|---|---|
| GRU | 2,302,801 | 280.1 | 0.7067 | 0.7067 | 0.7838 | 1.166 |
| Transformer | 2,162,501 | 192.8 | **0.7311** | **0.7623** | **0.8104** | 1.512 |

Full numbers: `gru_results_RESULTS.md` / `gru_results.json`.
Plots: `gru_results_val_f1_curve.png`, `gru_results_loss_curves.png`,
`gru_results_epoch_time.png`, `gru_results_confusion_gru.png`,
`gru_results_confusion_transformer.png`.

### Analysis

- **Accuracy/F1/AUC**: the Transformer outperforms the GRU at a comparable
  parameter budget. Self-attention lets every token attend directly to every
  other token in one layer, which appears to capture the long-range cues
  that matter for review sentiment (e.g. a negation or "but" clause far from
  the sentiment-bearing word) more effectively than a recurrent state that
  has to carry that information step-by-step across the sequence.
- **Training time**: despite having a comparable parameter count, the
  Transformer trained faster in wall-clock time. Self-attention is fully
  parallelizable across the sequence dimension, while the GRU must process
  tokens sequentially — this gap would be even larger on a GPU, where the
  Transformer's parallelism is exploited much more effectively than on CPU.
- **Inference latency**: the GRU is faster per example at inference time
  here, since attention has O(n²) cost in sequence length versus the GRU's
  O(n), and this test used relatively short sequences (max 120 tokens). This
  gap flips in the GRU's favor further as sequence length grows — a relevant
  trade-off for latency-sensitive deployment.
- **Overfitting behavior**: the GRU's training loss drops faster than its
  validation loss improves (see `gru_results_loss_curves.png`), consistent
  with a recurrent model with no attention regularization overfitting a
  small dataset faster than the Transformer here.
- **Takeaway**: with no pretraining, at this data scale, the Transformer's
  attention mechanism gives it an edge in classification quality, at the
  cost of more expensive inference. This mirrors the trade-off widely
  reported in the literature between the two architecture families before
  large-scale pretraining is factored in.

## Reproducing / scaling up

```bash
pip install -r gru_requirements.txt

# Quick run (CPU, ~10 min):
python gru_run_comparison.py --max_samples 6000 --epochs 4 --max_len 120

# Full benchmark (GPU recommended, uses the entire 50k dataset):
python gru_run_comparison.py --max_samples 50000 --epochs 10 --max_len 300 \
    --embed_dim 200 --hidden_dim 200 --num_layers 3
```

All metrics, plots and a `results.json` are written to `--out_dir`
(default `gru_results/`, created at run time).

## Files in this study

| File | Role |
|---|---|
| `gru_IMDB-Dataset.csv` | 50k labeled reviews |
| `gru_data_utils.py` | Cleaning, tokenization, vocab, Dataset/collate |
| `gru_models.py` | `GRUClassifier`, `TransformerClassifier` |
| `gru_train.py` | Shared training/eval loop, latency measurement |
| `gru_report.py` | Plots + markdown/JSON report generation |
| `gru_run_comparison.py` | Main entry point |
| `gru_requirements.txt` | Dependencies |
| `gru_results_*` | Generated metrics, plots, results table from the run |

## Possible extensions

- Add a third arm: a pretrained transformer (DistilBERT) fine-tuned on the
  same split, to separate "attention as an inductive bias" from "attention +
  large-scale pretraining."
- Attention-weight visualization for the Transformer to inspect which words
  it attends to for a given prediction (interpretability).
- Statistical significance testing (e.g. bootstrap confidence intervals on
  the accuracy gap) given multiple random seeds.

## Relationship to the original project

The original lexicon project (`baseline_analytics.py`, BlogIntel) scores
sentiment by counting words against static positive/negative lists and
computing readability indices (Fog Index, etc.) — no learning involved.
This project keeps the same underlying task (classify text sentiment) but
replaces the scoring mechanism with two supervised sequence models trained
on labeled data, and reports standard ML classification metrics (accuracy,
F1, precision, recall, AUC) plus a systematic comparison the lexicon
approach had no way to produce.
