"""PyTorch dataset that cuts patches on the fly (no huge patch array in memory)."""
import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Return (patch, label), with patch shaped (C, H, W) as expected by nn.Conv2d.

    `padded_cube` must be padded by patch_size // 2 on each spatial side, so that the
    patch centered on original pixel (r, c) is padded_cube[r:r+p, c:c+p].
    If `labels` is None, only the patch is returned (used for full-scene prediction).
    """

    def __init__(self, padded_cube, coords, patch_size, labels=None):
        self.cube = torch.from_numpy(np.ascontiguousarray(padded_cube, dtype=np.float32))
        self.coords = np.asarray(coords)
        self.p = patch_size
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        r, c = self.coords[i]
        patch = self.cube[r:r + self.p, c:c + self.p, :].permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        if self.labels is None:
            return patch
        return patch, self.labels[i]
