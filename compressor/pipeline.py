"""Building blocks shared by `compress.py` (PTQ / QAT runs) and `test.py`."""
import os

import torch

from mbv2.data import GpuAugment
from mbv2.engine import evaluate
from mbv2.model import mobilenet_v2

from .config import CompressionConfig
from .packed import PackedMobileNetV2, export_packed
from .qmodel import QMobileNetV2, recalibrate_bn
from .size import compression_summary, fp32_activation_elements, trace_activations


def load_baseline(path: str, device):
    """Load the trained full-precision MobileNet-v2 (final or resumable checkpoint)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = mobilenet_v2(10, cfg.get("width_mult", 1.0), cfg.get("dropout", 0.2))
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt["model"])
    return model.to(device).eval()


def calibration_batches(train_set, transform, n_images: int, batch_size: int = 256, seed: int = 0):
    """`n_images` un-augmented training images (fixed random subset) as normalised batches."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train_set), generator=g)[:n_images].to(train_set.labels.device)
    return [transform(train_set.images[idx[i:i + batch_size]])
            for i in range(0, n_images, batch_size)]


def bn_batches(train_set, device, n_images: int, batch_size: int = 256, seed: int = 0,
               cutout: int = 16):
    """`n_images` training images with the *training* augmentation, for BatchNorm re-estimation.

    The running statistics of the baseline were accumulated on augmented
    batches. Re-estimating them on un-augmented training images (which the
    network has memorised) shifts them and costs about 2.7 points of test
    accuracy even without any quantization, so the same augmentation is used.
    """
    augment = GpuAugment(device, augment=True, cutout=cutout, seed=seed)
    return calibration_batches(train_set, augment, n_images, batch_size, seed)


@torch.no_grad()
def post_training_quantize(float_model, cfg: CompressionConfig, calib, bn_calib=None):
    """Quantize a trained model without any gradient step.

    Weight steps are set by a per-channel MSE search when the model is built;
    activation steps by one calibration pass over `calib[0]`. If `bn_calib`
    (augmented training batches, see `bn_batches`) is given, the BatchNorm
    statistics are then re-estimated under quantization.
    """
    qmodel = QMobileNetV2.from_float(float_model, cfg).eval()
    qmodel.calibrate(calib[0])
    if bn_calib:
        recalibrate_bn(qmodel, bn_calib)
    return qmodel


def measure(packed: PackedMobileNetV2, float_model, test_set) -> dict:
    """Size and activation-memory numbers of a packed model (see compressor/size.py).

    Activation memory is measured on one test image: the compressed model is
    run with every activation tensor really stored as bit-packed codes, and
    the bytes of those buffers are added up; the FP32 reference is obtained
    with forward hooks on the uncompressed model.
    """
    image = test_set.images[:1]
    packed.pack_acts = True
    tracer = trace_activations(packed, image)
    packed.pack_acts = False
    out = compression_summary(packed, float_model, tracer)
    measured = sum(r["packed_bytes"] if r["packed_bytes"] is not None else r["size_bits"] / 8
                   for r in tracer.records)
    mean = packed.mean.to(image.device).view(1, 3, 1, 1)
    std = packed.std.to(image.device).view(1, 3, 1, 1)
    hooks = fp32_activation_elements(float_model, (image.float() / 255.0 - mean) / std)
    out.update({"act_measured_bytes": measured, "act_fp32_hook_bytes": hooks["bytes"],
                "act_measured_ratio": hooks["bytes"] / measured,
                "act_records": tracer.records})
    return out


def evaluate_packed(packed: PackedMobileNetV2, split):
    """(loss, top-1 %) of a packed model on a split; it consumes raw uint8 images."""
    return evaluate(packed, split, lambda images: images, batch_size=500)


def save_packed(packed: PackedMobileNetV2, path: str, extra: dict = None) -> int:
    """Save the packed state; returns the file size in bytes."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = packed.state()
    state["info"] = extra or {}
    torch.save(state, path)
    return os.path.getsize(path)


def load_packed(path: str) -> PackedMobileNetV2:
    return PackedMobileNetV2.from_state(torch.load(path, map_location="cpu", weights_only=False))
