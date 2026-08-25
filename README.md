# BlogIntel — Sentiment Analysis: From Lexicon Scoring to Learned Classification

## Purpose

This project classifies text sentiment (positive/negative) and shows two
generations of the same problem solved two different ways:

1. **`baseline_*` files** — the original approach: score sentiment by
   counting words against curated positive/negative dictionaries, no
   machine learning involved.
2. **`gru_*` files** — the upgrade: two neural sequence models (a GRU and
   a Transformer encoder), trained from scratch on 50,000 labeled movie
   reviews, evaluated with standard ML classification metrics.

Keeping both in one place shows the full progression: rule-based baseline
→ supervised learned models → a controlled comparison between two
architecture families (recurrence vs. self-attention).

## How it works

### 1. Lexicon baseline (`baseline_analytics.py`)

- Scrapes blog post text from a list of 100+ URLs (`baseline_Input.xlsx`)
  using `requests` + `BeautifulSoup`, in parallel via Python's
  `multiprocessing`.
- Cleans each post by removing stop words (loaded from a `StopWords/`
  folder of text files).
- Counts occurrences of words from a positive-words list and a
  negative-words list (`MasterDictionary/`) to derive a **Positive
  Score**, **Negative Score**, **Polarity Score**, and **Subjectivity
  Score**.
- Computes standard readability metrics — average sentence length,
  percentage of complex words, **Gunning Fog Index**, syllables per word,
  personal pronoun count, average word length — using a custom syllable
  counter and regex-based sentence/word splitting.
- Saves everything to `baseline_Output.csv`.
- This method needs no training data and is fully transparent (every
  score traces back to a word count), but it can't learn from examples,
  and it has no notion of "correct" labels to measure accuracy against —
  it can only report scores, not validated predictions.

### 2. GRU vs. Transformer comparison (`gru_*`)

- Loads 50,000 labeled IMDB movie reviews (`gru_IMDB-Dataset.csv`),
  cleans HTML artifacts, lowercases, and tokenizes (`gru_data_utils.py`).
- Builds a shared vocabulary (top ~20k tokens) and splits data into
  70% train / 15% validation / 15% test.
- Trains two models **from scratch** (no pretrained weights) under
  **identical conditions** — same vocab, split, batch size, optimizer,
  learning rate, and epoch count — so the comparison isolates the effect
  of the architecture itself (`gru_models.py`):
  - **GRUClassifier**: embedding → bidirectional GRU → linear head.
    Processes tokens sequentially; the final hidden state summarizes the
    whole sequence.
  - **TransformerClassifier**: embedding + sinusoidal positional encoding
    → Transformer encoder stack → mean-pooling → linear head. Every token
    attends to every other token in parallel, with explicit positional
    encoding standing in for the recurrence order a GRU gets for free.
- `gru_train.py` runs the shared training/evaluation loop, tracking loss,
  accuracy, F1, precision, recall, AUC, per-epoch training time, and
  measures per-example inference latency.
- `gru_report.py` generates a results table, JSON metrics, loss/F1
  curves, and confusion matrices; `gru_run_comparison.py` is the single
  entry point that ties it all together.

## Setup

```bash
# Lexicon baseline
# Needs a StopWords/ folder and a MasterDictionary/ folder (positive-words.txt,
# negative-words.txt) alongside baseline_analytics.py, plus internet access
# for scraping the URLs listed in baseline_Input.xlsx.
pip install pandas numpy requests beautifulsoup4 nltk pyenchant lxml
python baseline_analytics.py

# GRU vs Transformer comparison
pip install -r gru_requirements.txt
python gru_run_comparison.py --max_samples 6000 --epochs 4 --max_len 120
```

Key `gru_run_comparison.py` flags: `--max_samples` (subsample size, up to
50000), `--epochs`, `--max_len` (tokens per review), `--batch_size`,
`--embed_dim`, `--hidden_dim`, `--num_layers`, `--num_heads`, `--lr`,
`--out_dir` (default `gru_results/`, created at run time). For a stronger
result on a GPU: `python gru_run_comparison.py --max_samples 50000
--epochs 10 --max_len 300 --embed_dim 200 --hidden_dim 200
--num_layers 3`.

## Results

The lexicon baseline outputs per-post scores (`baseline_Output.csv`) but
has no ground-truth labels to score itself against — there's no accuracy
number to report for it, which is itself part of the motivation for the
upgrade.

GRU vs. Transformer, trained from scratch on a 6,000-review subset (CPU,
4 epochs) — full numbers in `gru_results_RESULTS.md` / `gru_results.json`:

| Model | Params | Train time (s) | Test Acc | Test F1 | Test AUC | Latency (ms/ex) |
|---|---|---|---|---|---|---|
| GRU | 2,302,801 | 280.1 | 0.7067 | 0.7067 | 0.7838 | 1.166 |
| Transformer | 2,162,501 | 192.8 | **0.7311** | **0.7623** | **0.8104** | 1.512 |

Plots: `gru_results_val_f1_curve.png`, `gru_results_loss_curves.png`,
`gru_results_epoch_time.png`, `gru_results_confusion_gru.png`,
`gru_results_confusion_transformer.png`.

**Reading the results:**
- **Accuracy/F1/AUC**: the Transformer wins at a comparable parameter
  budget. Self-attention lets every token attend directly to every other
  token in one layer, which seems to capture long-range cues (e.g. a
  negation far from the sentiment-bearing word) better than a recurrent
  state that has to carry that information step-by-step.
- **Training time**: the Transformer trained faster in wall-clock time
  despite a similar parameter count, since self-attention parallelizes
  across the sequence while the GRU processes tokens one at a time — this
  gap would widen further on a GPU.
- **Inference latency**: the GRU is faster per example here, since
  attention has O(n²) cost in sequence length vs. the GRU's O(n); this
  favors the GRU more as sequence length grows.
- **Overfitting**: the GRU's train loss drops faster than its validation
  loss improves (see the loss curve plot), suggesting it overfits this
  small dataset faster than the Transformer does.
- **Takeaway**: without pretraining, at this data scale, attention gives
  a quality edge at the cost of more expensive inference — a trade-off
  consistent with what's broadly reported in the literature before
  large-scale pretraining is factored in.

## Tech used

- **Language**: Python 3
- **Baseline**: `pandas`, `numpy`, `requests`, `BeautifulSoup4`, `nltk`,
  `pyenchant`, `multiprocessing`
- **GRU/Transformer study**: `PyTorch` (`nn.GRU`, `nn.TransformerEncoder`),
  `pandas`, `scikit-learn` (metrics), `matplotlib` (plots)
- **Dataset**: [IMDB 50k Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)
  (Maas et al., 2011), balanced binary sentiment benchmark

## File guide

| File | Role |
|---|---|
| `baseline_analytics.py` | Scraping, cleaning, lexicon scoring, Fog Index |
| `baseline_Input.xlsx` / `baseline_Output.csv` | Baseline input URL list / output scores |
| `baseline_Text_Analysis.docx` | Original project write-up |
| `gru_data_utils.py` | Cleaning, tokenization, vocab, Dataset/collate |
| `gru_models.py` | `GRUClassifier`, `TransformerClassifier` |
| `gru_train.py` | Shared training/eval loop, latency measurement |
| `gru_report.py` | Plots + markdown/JSON report generation |
| `gru_run_comparison.py` | Main entry point — trains & compares both models |
| `gru_requirements.txt` | Python dependencies for the GRU/Transformer study |
| `gru_IMDB-Dataset.csv` | 50k labeled movie reviews used for training |
| `gru_results_RESULTS.md` / `gru_results.json` | Metrics from the run reported above |
| `gru_results_*.png` | Training curves and confusion matrices from the run |

## Suggested resume line

> Built and compared GRU and Transformer-based sentiment classifiers
> trained from scratch on 50k IMDB reviews, evaluating accuracy, F1, AUC,
> training time and inference latency; extended an earlier lexicon-based
> sentiment pipeline into a full supervised learning benchmark.

## Possible next steps

- Add a pretrained DistilBERT fine-tuning arm to separate "attention as
  architecture" from "attention + large-scale pretraining."
- Deploy a small Streamlit/Gradio demo comparing all three approaches
  side by side on user-entered text.
- Add attention-weight visualization for interpretability.
- Add statistical significance testing (bootstrap CIs) across multiple
  random seeds.
