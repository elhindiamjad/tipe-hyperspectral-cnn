"""Figures saved to disk: learning curves, confusion matrix, classification maps."""
import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def _class_cmap(num_classes):
    colors = ["black"] + list(plt.cm.tab20.colors[:num_classes])  # 0 = unlabeled
    return ListedColormap(colors)


def plot_history(history, path):
    """Loss and accuracy curves for the training and validation sets."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Validation")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, class_names, path):
    """Row-normalized confusion matrix (each row sums to 1 = recall per class)."""
    cm = np.asarray(cm, dtype=float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    n = len(class_names)
    ax.set_xticks(range(n), labels=class_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels=class_names, fontsize=8)
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set(xlabel="Predicted class", ylabel="True class", title="Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_maps(gt, pred_map, class_names, path):
    """Ground truth, prediction on labeled pixels, and prediction on the full scene."""
    n = len(class_names)
    cmap = _class_cmap(n)
    masked = np.where(gt > 0, pred_map, 0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, img, title in zip(
        axes,
        (gt, masked, pred_map),
        ("Ground truth", "Prediction (labeled pixels)", "Prediction (full scene)"),
    ):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=n, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    handles = [mpatches.Patch(color=cmap.colors[i + 1], label=name) for i, name in enumerate(class_names)]
    fig.legend(handles=handles, loc="center right", fontsize=8, title="Classes")
    fig.tight_layout(rect=[0, 0, 0.84, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)
