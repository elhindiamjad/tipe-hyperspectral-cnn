"""Loading, normalization and train/val/test splitting of the Indian Pines scene."""
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
    "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
    "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
    "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
]


def load_indian_pines(data_dir):
    """Return the hyperspectral cube (H, W, B) as float32 and the ground truth (H, W)."""
    data_dir = Path(data_dir)
    cube_file = data_dir / "Indian_pines_corrected.mat"
    gt_file = data_dir / "Indian_pines_gt.mat"
    for f in (cube_file, gt_file):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found. See data/README.md to download the dataset.")
    cube = sio.loadmat(cube_file)["indian_pines_corrected"].astype(np.float32)
    gt = sio.loadmat(gt_file)["indian_pines_gt"].astype(np.int64)
    return cube, gt


def normalize_bands(cube):
    """Min-max scale each spectral band to [0, 1] (same as sklearn's MinMaxScaler per band).

    No labels are used, so fitting on the whole scene does not leak class information.
    """
    mins = cube.min(axis=(0, 1), keepdims=True)
    maxs = cube.max(axis=(0, 1), keepdims=True)
    return (cube - mins) / np.maximum(maxs - mins, 1e-12)


def pad_cube(cube, margin):
    """Zero-pad the spatial dimensions so patches can be cut around border pixels."""
    return np.pad(cube, ((margin, margin), (margin, margin), (0, 0)), mode="constant")


def labeled_pixels(gt):
    """Return the (row, col) coordinates of labeled pixels and their 0-based labels."""
    rows, cols = np.nonzero(gt)
    coords = np.stack([rows, cols], axis=1)
    labels = gt[rows, cols] - 1  # classes 1..16 -> 0..15
    return coords, labels


def all_pixels(gt):
    """Return the coordinates of every pixel of the scene (for full-map prediction)."""
    rows, cols = np.indices(gt.shape)
    return np.stack([rows.ravel(), cols.ravel()], axis=1)


def split_indices(labels, test_size=0.3, val_size=0.1, seed=42):
    """Stratified split into train / validation / test indices.

    `test_size` and `val_size` are fractions of the whole labeled set. The validation
    set is used for model selection; the test set is used only once, at the very end.
    """
    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )
    val_fraction = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_fraction, random_state=seed, stratify=labels[idx_trainval]
    )
    return idx_train, idx_val, idx_test
