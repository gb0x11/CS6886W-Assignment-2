"""Sensitivity-guided mixed-precision bit allocation for the weights.

Not every layer of MobileNet-v2 tolerates the same amount of quantization,
and the layers differ enormously in size (the stem has 864 weights, the last
1x1 convolution 409,600). Giving every layer the same bit-width therefore
wastes bits. The allocator below

  1. measures, for every layer and every candidate bit-width, how much the
     network output changes when only that layer's weights are quantized
     (KL divergence between the full-precision and the perturbed predictions
     on a calibration set, after re-estimating the BatchNorm statistics, so
     that only the damage that a cheap correction cannot undo is counted), and
  2. starting from the highest bit-width everywhere, repeatedly lowers the
     bit-width of the layer where doing so costs the least output change per
     bit of memory saved, until the requested average bit-width is reached.

Large, redundant layers end up with few bits and small, sensitive layers
keep many. The result is a {layer name: bits} table for CompressionConfig.
"""
import torch
import torch.nn.functional as F

from .config import CompressionConfig
from .qmodel import QConvBN, QMobileNetV2, recalibrate_bn
from .quantizers import Quantizer


def _weight(layer):
    return (layer.conv if isinstance(layer, QConvBN) else layer.linear).weight


@torch.no_grad()
def layer_sensitivity(float_model, batches, bit_choices=(2, 3, 4, 5, 6, 8),
                      bn_calib=None) -> dict:
    """KL(full precision || one layer quantized) for every layer and bit-width.

    `batches` is a list of normalised image batches. If `bn_calib` (a list of
    augmented training batches) is given, the BatchNorm statistics are
    re-estimated on it after each perturbation, and also for the reference,
    so that the estimation noise is common to both. Returns
    {"sizes": {layer: number of weights}, "kl": {layer: {bits: value}}}.
    """
    # A wrapper with all quantizers off gives named access to every layer.
    model = QMobileNetV2.from_float(float_model, CompressionConfig(w_bits=32, a_bits=32)).eval()
    if bn_calib:
        recalibrate_bn(model, bn_calib)
    reference = [F.softmax(model(x), dim=1) for x in batches]
    sizes, kl = {}, {}
    for layer in model.weight_layers():
        w = _weight(layer)
        original = w.data.clone()
        sizes[layer.name], kl[layer.name] = w.numel(), {}
        for bits in bit_choices:
            q = Quantizer(bits, signed=True, channels=w.shape[0]).to(w.device).eval()
            q.init_from(original)
            w.data.copy_(q(original))
            if bn_calib:
                recalibrate_bn(model, bn_calib)  # resets, then re-estimates all statistics
            total = 0.0
            for x, ref in zip(batches, reference):
                log_p = F.log_softmax(model(x), dim=1)
                total += F.kl_div(log_p, ref, reduction="sum").item()
            # KL is non-negative; clamp the float round-off of a near-zero value.
            kl[layer.name][int(bits)] = max(0.0, total / sum(len(x) for x in batches))
        w.data.copy_(original)
    return {"sizes": sizes, "kl": kl}


def allocate_bits(sensitivity: dict, target_avg_bits: float, fixed: dict = None) -> dict:
    """Greedy allocation of one bit-width per layer for a target average.

    Args:
        sensitivity: output of `layer_sensitivity`.
        target_avg_bits: desired parameter-weighted mean bit-width over all layers.
        fixed: {layer: bits} for layers whose bit-width is not to be changed
               (they still count towards the average).
    """
    sizes = sensitivity["sizes"]
    kl = {n: {int(b): v for b, v in d.items()} for n, d in sensitivity["kl"].items()}
    fixed = dict(fixed or {})
    choices = sorted(next(iter(kl.values())).keys())
    # More bits can never hurt: remove measurement noise by making the
    # sensitivity non-increasing in the bit-width.
    for name, d in kl.items():
        worst = 0.0
        for b in sorted(d, reverse=True):
            worst = max(worst, d[b])
            d[b] = worst
    bits = {n: fixed.get(n, choices[-1]) for n in sizes}
    total = sum(sizes.values())

    def average():
        return sum(sizes[n] * bits[n] for n in sizes) / total

    while average() > target_avg_bits:
        best, best_cost = None, None
        for n in sizes:
            if n in fixed or bits[n] == choices[0]:
                continue
            lower = choices[choices.index(bits[n]) - 1]
            saved = sizes[n] * (bits[n] - lower)
            cost = (kl[n][lower] - kl[n][bits[n]]) / saved  # output change per bit saved
            if best_cost is None or cost < best_cost:
                best, best_cost = (n, lower), cost
        if best is None:
            break  # every free layer is already at the lowest bit-width
        bits[best[0]] = best[1]
    return bits
