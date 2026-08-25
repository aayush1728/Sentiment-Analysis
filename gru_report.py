import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_report(results, config, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)

    lines = ["# GRU vs Transformer — Results\n"]
    lines.append(f"Config: `{json.dumps(config)}`\n")
    lines.append("| Model | Params | Train time (s) | Test Acc | Test F1 | Test AUC | Latency (ms/ex) |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['params']:,} | {r['train_time_sec']:.1f} | "
            f"{r['test_accuracy']:.4f} | {r['test_f1']:.4f} | {r['test_auc']:.4f} | "
            f"{r['inference_ms_per_example']:.3f} |"
        )
    with open(os.path.join(out_dir, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines))


def save_plots(histories, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    for name, hist in histories.items():
        epochs = range(1, len(hist["val_f1"]) + 1)
        plt.plot(epochs, hist["val_f1"], marker="o", label=f"{name} val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Validation F1")
    plt.title("Validation F1 over training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_f1_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        plt.plot(epochs, hist["train_loss"], marker="o", label=f"{name} train loss")
        plt.plot(epochs, hist["val_loss"], marker="x", linestyle="--", label=f"{name} val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name, hist in histories.items():
        epochs = range(1, len(hist["epoch_time"]) + 1)
        plt.bar([f"{name}\nE{e}" for e in epochs], hist["epoch_time"])
    plt.ylabel("Seconds")
    plt.title("Per-epoch training time")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "epoch_time.png"), dpi=150)
    plt.close()


def confusion_matrix_plot(labels, preds, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{name} — Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_{name.lower()}.png"), dpi=150)
    plt.close()
