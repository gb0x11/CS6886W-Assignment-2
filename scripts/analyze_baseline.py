"""Failure analysis of the full-precision baseline (Question 1c).

    python scripts/analyze_baseline.py [--checkpoint checkpoints/baseline.pt]

Writes results/baseline_analysis.json (per-class accuracy, confusion matrix,
most confused pairs, confidence statistics), report/figures/confusion.png and
report/figures/failures.png (the most confident wrong predictions).
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressor.pipeline import load_baseline  # noqa: E402
from mbv2.data import CLASSES, GpuAugment, iterate, load_cifar10  # noqa: E402
from mbv2.utils import get_device, save_json  # noqa: E402

SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#2a78d6", "#0b2f5c"])


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/baseline.pt")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = get_device(args.device)
    _, _, test = load_cifar10(args.data_dir, device)
    model = load_baseline(args.checkpoint, device)
    tf = GpuAugment(device, augment=False)

    probs = torch.cat([F.softmax(model(x), dim=1) for x, _ in iterate(test, 500, tf)]).cpu()
    labels = test.labels.cpu()
    conf, pred = probs.max(dim=1)
    wrong = pred != labels
    cm = torch.zeros(10, 10, dtype=torch.long)
    cm.index_put_((labels, pred), torch.ones_like(labels), accumulate=True)
    per_class = (cm.diag().float() / cm.sum(1).float() * 100).tolist()
    off = cm.clone()
    off.fill_diagonal_(0)
    pairs = sorted(((int(off[i, j]), CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10)
                    if i != j), reverse=True)[:8]
    animals = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
    is_animal = torch.zeros(10, dtype=torch.bool)
    is_animal[animals] = True
    within_group = (is_animal[labels[wrong]] == is_animal[pred[wrong]]).float().mean().item()
    out = {
        "test_acc": 100 * (1 - wrong.float().mean().item()), "errors": int(wrong.sum()),
        "per_class_acc": dict(zip(CLASSES, per_class)), "confusion": cm.tolist(),
        "top_confusions": [{"count": c, "true": t, "pred": q} for c, t, q in pairs],
        "cat_dog_errors": int(cm[3, 5] + cm[5, 3]),
        "errors_within_animal_or_vehicle_group": within_group,
        "mean_conf_correct": conf[~wrong].mean().item(),
        "mean_conf_wrong": conf[wrong].mean().item(),
        "wrong_with_conf_above_0.9": int((conf[wrong] > 0.9).sum()),
    }
    save_json(out, "results/baseline_analysis.json")
    print({k: v for k, v in out.items() if k != "confusion"})

    os.makedirs("report/figures", exist_ok=True)
    # Confusion matrix (errors only, so that the diagonal does not hide them).
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(off.numpy(), cmap=SEQ)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(10), CLASSES, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(10):
        for j in range(10):
            v = int(off[i, j])
            if v >= 10:
                ax.text(j, i, v, ha="center", va="center", fontsize=6.5,
                        color="white" if v > off.max() * 0.5 else "#0b0b0b")
    ax.set_title("Test errors by class pair (diagonal removed)", loc="left", fontsize=9,
                 fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout()
    fig.savefig("report/figures/confusion.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # The most confident mistakes.
    idx = torch.nonzero(wrong).squeeze(1)
    idx = idx[conf[idx].argsort(descending=True)][:12]
    images = test.images[idx.to(device)].cpu().permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(2, 6, figsize=(9.0, 3.5))
    for ax, img, i in zip(axes.flat, images, idx.tolist()):
        ax.imshow(np.asarray(img), interpolation="nearest")
        ax.set_title(f"true: {CLASSES[labels[i]]}\npred: {CLASSES[pred[i]]} ({conf[i]:.2f})",
                     fontsize=7)
        ax.axis("off")
    fig.suptitle("The 12 most confident wrong predictions on the test set", x=0.01, ha="left",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig("report/figures/failures.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
