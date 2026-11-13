"""CIFAR-10 loading, normalization and data augmentation.

The whole dataset (50k + 10k images of 3x32x32 uint8, about 180 MB) is moved
to the training device once and augmentation is applied there per batch. On
a laptop GPU this removes the PIL / DataLoader bottleneck. The transforms are
the standard CIFAR-10 recipe:

    train: RandomCrop(32, padding=4) -> RandomHorizontalFlip(p=0.5)
           -> ToTensor -> Normalize(mean, std) -> Cutout(1 hole, 16x16)
    test:  ToTensor -> Normalize(mean, std)

Random numbers for augmentation come from a seeded CPU generator, so a run is
repeatable for a given seed on any device.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torchvision import datasets

# Per-channel mean / std of the CIFAR-10 training set (pixels scaled to [0, 1]).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


@dataclass
class Split:
    images: torch.Tensor  # uint8, (N, 3, 32, 32), on device
    labels: torch.Tensor  # int64, (N,), on device

    def __len__(self):
        return len(self.labels)


def _to_tensors(ds, device):
    x = torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).contiguous()
    y = torch.as_tensor(ds.targets, dtype=torch.long)
    return x.to(device), y.to(device)


def load_cifar10(root: str, device: torch.device, val_size: int = 0, split_seed: int = 0):
    """Return (train, val, test) splits as on-device uint8 tensors.

    By default (`val_size=0`) the network trains on all 50,000 training images
    and `val` is None; the 10,000 test images are used only to report accuracy
    (no early stopping or checkpoint selection is done on them). With
    `val_size > 0` that many training images are held out with a fixed seed.
    """
    train_ds = datasets.CIFAR10(root, train=True, download=True)
    test_ds = datasets.CIFAR10(root, train=False, download=True)
    x, y = _to_tensors(train_ds, device)
    test = Split(*_to_tensors(test_ds, device))
    if val_size == 0:
        return Split(x, y), None, test
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(y), generator=g).to(device)
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    return Split(x[train_idx], y[train_idx]), Split(x[val_idx], y[val_idx]), test


class GpuAugment:
    """Normalization plus optional crop / flip / cutout on a batch of uint8 images."""

    def __init__(self, device, augment=True, pad=4, cutout=16, seed=0):
        self.device = device
        self.augment = augment
        self.pad = pad
        self.cutout = cutout
        self.gen = torch.Generator().manual_seed(seed)
        self.mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)

    def _rand(self, *shape, high):
        return torch.randint(0, high, shape, generator=self.gen).to(self.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().div_(255.0)  # ToTensor
        if self.augment:
            b, _, h, w = x.shape
            # RandomCrop(32, padding=4): zero-pad, then take a random 32x32 window.
            xp = torch.nn.functional.pad(x, [self.pad] * 4)
            oy = self._rand(b, high=2 * self.pad + 1)
            ox = self._rand(b, high=2 * self.pad + 1)
            rows = (oy[:, None] + torch.arange(h, device=self.device))[:, None, :, None]
            cols = (ox[:, None] + torch.arange(w, device=self.device))[:, None, None, :]
            bidx = torch.arange(b, device=self.device)[:, None, None, None]
            cidx = torch.arange(3, device=self.device)[None, :, None, None]
            x = xp[bidx, cidx, rows, cols]
            # RandomHorizontalFlip(p=0.5)
            flip = self._rand(b, high=2).bool()
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        x = (x - self.mean) / self.std  # Normalize
        if self.augment and self.cutout > 0:
            # Cutout: zero a square of side `cutout` centred at a random pixel
            # (clipped at the border). Zero equals the dataset mean after
            # normalization.
            b, _, h, w = x.shape
            cy = self._rand(b, high=h)[:, None, None]
            cx = self._rand(b, high=w)[:, None, None]
            ys = torch.arange(h, device=self.device)[None, :, None]
            xs = torch.arange(w, device=self.device)[None, None, :]
            half = self.cutout // 2
            hole = ((ys >= cy - half) & (ys < cy + half) &
                    (xs >= cx - half) & (xs < cx + half))
            x = x.masked_fill(hole[:, None], 0.0)
        return x


def iterate(split: Split, batch_size: int, transform, shuffle=False, gen=None, drop_last=False):
    """Yield (images, labels) mini-batches from an on-device split."""
    n = len(split)
    if shuffle:
        order = torch.randperm(n, generator=gen).to(split.labels.device)
    else:
        order = torch.arange(n, device=split.labels.device)
    end = n - n % batch_size if drop_last else n
    for i in range(0, end, batch_size):
        idx = order[i:i + batch_size]
        yield transform(split.images[idx]), split.labels[idx]
