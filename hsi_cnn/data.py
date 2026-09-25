"""Loading, normalization and train/val/test splitting of the Indian Pines scene."""
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.ndimage import binary_dilation
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


def spatial_split_indices(coords, labels, shape, test_size=0.3, val_size=0.1, buffer=5, seed=42):
    """Spatially disjoint, class-stratified split.

    Each class is cut along its longest axis (rows or columns) into three contiguous
    regions: the first `test_size` of its pixels go to the test set, the last `val_size`
    to the validation set and the middle to the training set. The direction of the cut
    is drawn at random for each class, so different seeds give different splits.
    Every class is thus present in every set, even the rare ones that occupy a single
    small region of the scene.

    Training and validation pixels closer than `buffer` pixels to a test pixel (Chebyshev
    distance) are then discarded, so that no test patch contains a training pixel when
    `buffer` >= patch_size // 2. The smallest classes (Alfalfa, Grass-pasture-mowed, Oats
    for an 11 x 11 patch) are too small to keep training pixels after this step: they cannot
    be evaluated on disjoint patches, so they are removed from the validation and test sets.
    """
    rng = np.random.default_rng(seed)
    pixel_set = np.zeros(len(labels), dtype=np.int64)  # 0 = train, 1 = val, 2 = test
    for k in np.unique(labels):
        idx = np.flatnonzero(labels == k)
        rows, cols = coords[idx, 0], coords[idx, 1]
        if np.ptp(rows) >= np.ptp(cols):
            order = idx[np.lexsort((cols, rows))]
        else:
            order = idx[np.lexsort((rows, cols))]
        if rng.random() < 0.5:
            order = order[::-1]
        n_test = max(1, round(test_size * len(idx)))
        n_val = max(1, round(val_size * len(idx)))
        pixel_set[order[:n_test]] = 2
        pixel_set[order[len(idx) - n_val:]] = 1

    test_mask = np.zeros(shape, dtype=bool)
    test_mask[tuple(coords[pixel_set == 2].T)] = True
    near_test = binary_dilation(test_mask, np.ones((2 * buffer + 1,) * 2, dtype=bool))
    keep = ~near_test[coords[:, 0], coords[:, 1]]

    train = (pixel_set == 0) & keep
    learnable = np.isin(labels, labels[train])
    idx = np.arange(len(labels))
    return (idx[train], idx[(pixel_set == 1) & keep & learnable],
            idx[(pixel_set == 2) & learnable])


def block_split_indices(coords, shape, test_size=0.3, val_size=0.1, block_size=16,
                        buffer=5, seed=42):
    """Spatially disjoint split by blocks: the scene is cut into square blocks, and whole
    blocks are assigned to the test, validation or training set.

    Blocks are shuffled, then added to the test set until it holds `test_size` of the
    labeled pixels, then to the validation set, the rest going to training. Training and
    validation pixels closer than `buffer` pixels to a test pixel (Chebyshev distance) are
    discarded. Unlike `spatial_split_indices`, the split is not stratified: a class lying in
    a single region of the scene ends up entirely in one set.
    """
    rng = np.random.default_rng(seed)
    n_block_cols = -(-shape[1] // block_size)  # ceil division
    block_id = (coords[:, 0] // block_size) * n_block_cols + coords[:, 1] // block_size
    counts = np.bincount(block_id)

    # 0 = train, 1 = val, 2 = test
    block_set = np.zeros(len(counts), dtype=np.int64)
    n, cum = len(coords), 0
    for b in rng.permutation(np.unique(block_id)):
        if cum < test_size * n:
            block_set[b] = 2
        elif cum < (test_size + val_size) * n:
            block_set[b] = 1
        cum += counts[b]
    pixel_set = block_set[block_id]

    test_mask = np.zeros(shape, dtype=bool)
    test_mask[tuple(coords[pixel_set == 2].T)] = True
    near_test = binary_dilation(test_mask, np.ones((2 * buffer + 1,) * 2, dtype=bool))
    keep = ~near_test[coords[:, 0], coords[:, 1]]

    idx = np.arange(n)
    return idx[(pixel_set == 0) & keep], idx[(pixel_set == 1) & keep], idx[pixel_set == 2]
