"""
GRU vs Transformer: a controlled comparative study on IMDB sentiment
classification.

Both models are trained FROM SCRATCH (no pretrained embeddings/weights)
on the exact same data split, vocabulary, batch size, optimizer and
number of epochs, so that differences in the results reflect the
architectures themselves rather than confounding factors.

Usage:
    python run_comparison.py --max_samples 8000 --epochs 4 --max_len 200

Increase --max_samples toward 50000 and --epochs toward 8-10 on a
machine with a GPU for a stronger, more publication-worthy result.
"""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from data_utils import (
    load_imdb, build_vocab, split_df, IMDBDataset, make_collate_fn, PAD_TOKEN,
)
from models import GRUClassifier, TransformerClassifier, count_parameters
from train import train_model, run_epoch, measure_inference_latency
from report import save_plots, save_report, confusion_matrix_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="gru_IMDB-Dataset.csv")
    parser.add_argument("--max_samples", type=int, default=8000,
                         help="Subsample the dataset for faster runs; use full 50000 with a GPU.")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="gru_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading & tokenizing IMDB reviews...")
    df = load_imdb(args.csv_path, max_samples=args.max_samples)
    train_df, val_df, test_df = split_df(df)
    print(f"Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    vocab = build_vocab(train_df["tokens"], max_vocab_size=args.vocab_size)
    pad_idx = vocab[PAD_TOKEN]
    print(f"Vocab size: {len(vocab)}")

    collate = make_collate_fn(pad_idx)
    train_ds = IMDBDataset(train_df, vocab, max_len=args.max_len)
    val_ds = IMDBDataset(val_df, vocab, max_len=args.max_len)
    test_ds = IMDBDataset(test_df, vocab, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    results = {}
    histories = {}

    # ---------------- GRU ----------------
    gru = GRUClassifier(
        vocab_size=len(vocab), embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, pad_idx=pad_idx,
    )
    gru_params = count_parameters(gru)
    print(f"\nGRU parameters: {gru_params:,}")
    t0 = time.time()
    gru, gru_hist = train_model(gru, train_loader, val_loader, device,
                                 epochs=args.epochs, lr=args.lr, name="GRU")
    gru_train_time = time.time() - t0

    criterion = torch.nn.BCEWithLogitsLoss()
    gru_test_metrics, gru_labels, gru_preds = run_epoch(gru, test_loader, None, criterion, device, train=False)
    gru_latency = measure_inference_latency(gru, test_loader, device)

    results["GRU"] = {
        "params": gru_params,
        "train_time_sec": gru_train_time,
        "test_accuracy": gru_test_metrics["accuracy"],
        "test_f1": gru_test_metrics["f1"],
        "test_precision": gru_test_metrics["precision"],
        "test_recall": gru_test_metrics["recall"],
        "test_auc": gru_test_metrics["auc"],
        "inference_ms_per_example": gru_latency,
    }
    histories["GRU"] = gru_hist
    confusion_matrix_plot(gru_labels, gru_preds, "GRU", args.out_dir)

    # ---------------- Transformer ----------------
    transformer = TransformerClassifier(
        vocab_size=len(vocab), embed_dim=args.embed_dim, num_heads=args.num_heads,
        ff_dim=args.embed_dim * 2, num_layers=args.num_layers, pad_idx=pad_idx,
        max_len=args.max_len,
    )
    tf_params = count_parameters(transformer)
    print(f"\nTransformer parameters: {tf_params:,}")
    t0 = time.time()
    transformer, tf_hist = train_model(transformer, train_loader, val_loader, device,
                                        epochs=args.epochs, lr=args.lr, name="Transformer")
    tf_train_time = time.time() - t0

    tf_test_metrics, tf_labels, tf_preds = run_epoch(transformer, test_loader, None, criterion, device, train=False)
    tf_latency = measure_inference_latency(transformer, test_loader, device)

    results["Transformer"] = {
        "params": tf_params,
        "train_time_sec": tf_train_time,
        "test_accuracy": tf_test_metrics["accuracy"],
        "test_f1": tf_test_metrics["f1"],
        "test_precision": tf_test_metrics["precision"],
        "test_recall": tf_test_metrics["recall"],
        "test_auc": tf_test_metrics["auc"],
        "inference_ms_per_example": tf_latency,
    }
    histories["Transformer"] = tf_hist
    confusion_matrix_plot(tf_labels, tf_preds, "Transformer", args.out_dir)

    # ---------------- Report ----------------
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"{name}: acc={r['test_accuracy']:.4f} f1={r['test_f1']:.4f} "
              f"auc={r['test_auc']:.4f} params={r['params']:,} "
              f"train_time={r['train_time_sec']:.1f}s "
              f"latency={r['inference_ms_per_example']:.3f}ms/ex")

    config = vars(args)
    save_report(results, config, args.out_dir)
    save_plots(histories, args.out_dir)
    print(f"\nSaved metrics, plots and report to {args.out_dir}/")


if __name__ == "__main__":
    main()
