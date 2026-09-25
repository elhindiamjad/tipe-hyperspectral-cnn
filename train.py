"""Train and evaluate the 2D CNN on Indian Pines.

Usage:
    python train.py                       # default settings
    python train.py --epochs 100 --lr 5e-4
    python train.py --split spatial --out-dir results/spatial
    python train.py --seeds 0 1 2 3 4       # one run per seed + mean / std in summary.json
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hsi_cnn.data import (CLASS_NAMES, all_pixels, labeled_pixels, load_indian_pines,
                          block_split_indices, normalize_bands, pad_cube,
                          spatial_split_indices, split_indices)
from hsi_cnn.dataset import PatchDataset
from hsi_cnn.engine import evaluate, predict, train_one_epoch
from hsi_cnn.metrics import compute_metrics
from hsi_cnn.model import HSI2DCNN
from hsi_cnn.plots import plot_confusion, plot_history, plot_maps
from hsi_cnn.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--patch-size", type=int, default=11, help="odd number")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--split", choices=["random", "blocks", "spatial"], default="random",
                   help="random: stratified pixel split; blocks: disjoint blocks of the image; "
                        "spatial: disjoint regions of each class (stratified)")
    p.add_argument("--block-size", type=int, default=16, help="block side for --split blocks")
    p.add_argument("--buffer", type=int, default=None,
                   help="for spatial splits: drop train/val pixels this close to a test pixel "
                        "(default: patch_size // 2)")
    p.add_argument("--seeds", "--seed", type=int, nargs="+", default=[42],
                   help="one or several seeds; with several, results are averaged")
    return p.parse_args()


def run(args, seed, out):
    """Train and evaluate one model with the given seed. Return the test metrics."""
    set_seed(seed)
    device = get_device()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | seed {seed}")

    # 1. Data
    cube, gt = load_indian_pines(args.data_dir)
    cube = normalize_bands(cube)
    padded = pad_cube(cube, args.patch_size // 2)
    coords, labels = labeled_pixels(gt)
    if args.split == "random":
        idx_train, idx_val, idx_test = split_indices(labels, args.test_size, args.val_size, seed)
    else:
        buffer = args.patch_size // 2 if args.buffer is None else args.buffer
        if args.split == "blocks":
            idx_train, idx_val, idx_test = block_split_indices(
                coords, gt.shape, args.test_size, args.val_size, args.block_size, buffer, seed)
        else:
            idx_train, idx_val, idx_test = spatial_split_indices(
                coords, labels, gt.shape, args.test_size, args.val_size, buffer, seed)
    print(f"Cube {cube.shape} | labeled pixels: {len(labels)} "
          f"(train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)})")
    missing = [CLASS_NAMES[k] for k in range(len(CLASS_NAMES)) if not (labels[idx_train] == k).any()]
    if missing:
        print(f"Classes without training pixels: {', '.join(missing)}")

    def loader(idx, shuffle):
        ds = PatchDataset(padded, coords[idx], args.patch_size, labels[idx])
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader, val_loader, test_loader = loader(idx_train, True), loader(idx_val, False), loader(idx_test, False)

    # 2. Model, loss, optimizer
    num_classes = len(CLASS_NAMES)
    model = HSI2DCNN(cube.shape[2], num_classes, args.patch_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 3. Training, with model selection on the VALIDATION set (never on the test set)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch = -1.0, 0
    ckpt = out / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        for key, val in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[key].append(val)
        flag = ""
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), ckpt)
            flag = "  <- best"
        print(f"Epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}{flag}")

    # 4. Final evaluation of the best model on the test set (used only once)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, _, preds, trues = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(trues, preds, num_classes)
    metrics.update(best_epoch=best_epoch, best_val_acc=best_val_acc, seed=seed, config=vars(args))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 5. Classification map of the whole scene
    scene_ds = PatchDataset(padded, all_pixels(gt), args.patch_size)
    scene_pred = predict(model, DataLoader(scene_ds, batch_size=512), device)
    pred_map = scene_pred.reshape(gt.shape) + 1  # back to classes 1..16

    # 6. Figures
    plot_history(history, out / "learning_curves.png")
    plot_confusion(metrics["confusion_matrix"], CLASS_NAMES, out / "confusion_matrix.png")
    plot_maps(gt, pred_map, CLASS_NAMES, out / "classification_maps.png")

    print(f"\nBest epoch (validation): {best_epoch}")
    print(f"Test OA    : {metrics['overall_accuracy']:.4f}")
    print(f"Test AA    : {metrics['average_accuracy']:.4f}")
    print(f"Test kappa : {metrics['kappa']:.4f}")
    print(f"Results saved in {out}/")
    return metrics


def main():
    args = parse_args()
    out = Path(args.out_dir)
    if len(args.seeds) == 1:
        run(args, args.seeds[0], out)
        return

    runs = [run(args, seed, out / f"seed_{seed}") for seed in args.seeds]
    summary = {"seeds": args.seeds, "config": vars(args)}
    print(f"\nSummary over {len(runs)} seeds ({args.split} split):")
    for key in ("overall_accuracy", "average_accuracy", "kappa"):
        values = [m[key] for m in runs]
        summary[key] = {"mean": statistics.mean(values), "std": statistics.stdev(values),
                        "values": values}
        print(f"  {key:17s}: {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary saved in {out / 'summary.json'}")


if __name__ == "__main__":
    main()
