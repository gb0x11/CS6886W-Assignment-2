"""MobileNet-v2 with quantized weights and activations (training-time view).

`QMobileNetV2.from_float(model, cfg)` copies a trained full-precision
MobileNet-v2 and attaches quantizers to it:

  * every conv / linear weight gets a signed per-output-channel quantizer;
  * every activation that is kept in memory between two layers gets a
    quantizer: unsigned after ReLU6, signed for the linear-bottleneck outputs
    and for the residual sums, unsigned for the pooled feature vector;
  * the input image is left as it is: CIFAR-10 pixels are 8-bit integers, so
    the normalised input is exactly representable with 8-bit codes.

BatchNorm layers are kept as separate modules here so that the network can be
fine-tuned with quantization in the loop (stable batch statistics). They are
folded into a per-channel scale and bias when the model is exported to its
packed in-memory form (see `compressor/packed.py`), which leaves the integer
weight codes unchanged.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbv2.model import MobileNetV2

from .config import INPUT_BITS, CompressionConfig
from .quantizers import FULL_PRECISION, Quantizer
from .tracer import NullTracer

_NULL = NullTracer()


def _bits(q: Quantizer) -> int:
    """Bits per element of a tensor produced by quantizer q (32 if it is off)."""
    return q.bits if q.active else FULL_PRECISION


class QConvBN(nn.Module):
    """Quantized-weight conv -> BatchNorm -> (ReLU6) -> quantized activation."""

    def __init__(self, src, name: str, cfg: CompressionConfig):
        super().__init__()
        self.name, self.kind = name, src.kind
        self.conv, self.bn = src.conv, src.bn
        self.relu = src.act is not None
        self.wq = Quantizer(cfg.weight_bits(name, src.kind), signed=True,
                            channels=src.conv.out_channels)
        self.aq = Quantizer(cfg.act_bits(self.relu), signed=not self.relu)
        self.wq.init_from(self.conv.weight.detach())

    def quantized_weight(self):
        return self.wq(self.conv.weight)

    def forward(self, x):
        c = self.conv
        y = F.conv2d(x, self.quantized_weight(), None, c.stride, c.padding, c.dilation, c.groups)
        y = self.bn(y)
        if self.relu:
            y = F.relu6(y, inplace=True)
        return self.aq(y)


class QInvertedResidual(nn.Module):
    def __init__(self, src, name: str, cfg: CompressionConfig):
        super().__init__()
        self.name, self.use_res = name, src.use_res
        self.expand = QConvBN(src.expand, f"{name}.expand", cfg) if src.expand is not None else None
        self.depthwise = QConvBN(src.depthwise, f"{name}.depthwise", cfg)
        self.project = QConvBN(src.project, f"{name}.project", cfg)
        # The residual sum is a new tensor that is stored, so it is re-quantized.
        self.aq_add = Quantizer(cfg.act_bits(relu=False), signed=True) if self.use_res else None

    def forward(self, x, in_name, t=_NULL):
        """`in_name` is the tracer name of the block input (kept alive for the skip)."""
        n, cur = self.name, in_name
        out = x
        if self.expand is not None:
            out = self.expand(out)
            t.alloc(f"{n}.expand", out, _bits(self.expand.aq))
            if not self.use_res:
                t.free(in_name)
            cur = f"{n}.expand"
        out = self.depthwise(out)
        t.alloc(f"{n}.depthwise", out, _bits(self.depthwise.aq))
        if cur != in_name or not self.use_res:
            t.free(cur)
        out = self.project(out)
        t.alloc(f"{n}.project", out, _bits(self.project.aq))
        t.free(f"{n}.depthwise")
        if not self.use_res:
            return out, f"{n}.project"
        out = self.aq_add(x + out)
        t.alloc(f"{n}.add", out, _bits(self.aq_add))
        t.free(f"{n}.project")
        t.free(in_name)
        return out, f"{n}.add"


class QLinear(nn.Module):
    def __init__(self, src: nn.Linear, name: str, cfg: CompressionConfig):
        super().__init__()
        self.name, self.kind, self.linear = name, "classifier", src
        self.wq = Quantizer(cfg.weight_bits(name, "classifier"), signed=True,
                            channels=src.out_features)
        self.wq.init_from(src.weight.detach())

    def quantized_weight(self):
        return self.wq(self.linear.weight)

    def forward(self, x):
        return F.linear(x, self.quantized_weight(), self.linear.bias)


class QMobileNetV2(nn.Module):
    def __init__(self, float_model: MobileNetV2, cfg: CompressionConfig):
        super().__init__()
        m = copy.deepcopy(float_model)  # the full-precision model is left untouched
        self.cfg = cfg
        self.stem = QConvBN(m.stem, "stem", cfg)
        self.blocks = nn.ModuleList(QInvertedResidual(b, f"blocks.{i}", cfg)
                                    for i, b in enumerate(m.blocks))
        self.head = QConvBN(m.head, "head", cfg)
        self.aq_pool = Quantizer(cfg.act_bits(relu=False), signed=False)
        self.dropout = nn.Dropout(m.dropout_p)
        self.classifier = QLinear(m.classifier, "classifier", cfg)
        # The quantizers were created on the CPU: move them next to the weights.
        self.to(next(float_model.parameters()).device)

    @classmethod
    def from_float(cls, float_model: MobileNetV2, cfg: CompressionConfig) -> "QMobileNetV2":
        return cls(float_model, cfg)

    # ----------------------------------------------------------------- forward
    def forward(self, x, tracer=None):
        t = tracer or _NULL
        quantized = any(q.active for q in self.act_quantizers())
        t.alloc("input", x, INPUT_BITS if quantized else FULL_PRECISION)
        x = self.stem(x)
        t.alloc("stem", x, _bits(self.stem.aq))
        t.free("input")
        name = "stem"
        for block in self.blocks:
            x, name = block(x, name, t)
        x = self.head(x)
        t.alloc("head", x, _bits(self.head.aq))
        t.free(name)
        x = self.aq_pool(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))
        t.alloc("pool", x, _bits(self.aq_pool))
        t.free("head")
        x = self.classifier(self.dropout(x))
        t.alloc("logits", x, FULL_PRECISION)  # 10 numbers, left in full precision
        t.free("pool")
        return x

    # ----------------------------------------------------------------- helpers
    def weight_layers(self):
        """All layers that own a weight tensor, in execution order."""
        layers = [self.stem]
        for b in self.blocks:
            layers += [l for l in (b.expand, b.depthwise, b.project) if l is not None]
        return layers + [self.head, self.classifier]

    def act_quantizers(self):
        qs = [self.stem.aq]
        for b in self.blocks:
            qs += [l.aq for l in (b.expand, b.depthwise, b.project) if l is not None]
            if b.aq_add is not None:
                qs.append(b.aq_add)
        return qs + [self.head.aq, self.aq_pool]

    def step_parameters(self):
        """Quantizer step sizes (trained without weight decay)."""
        return [m.step for m in self.modules() if isinstance(m, Quantizer)]

    def set_quant(self, weights: bool = True, activations: bool = True):
        """Switch weight / activation quantization on or off (off = full precision)."""
        for layer in self.weight_layers():
            layer.wq.enabled = weights
        for q in self.act_quantizers():
            q.enabled = activations
        return self

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor):
        """Initialise the activation steps from a calibration batch.

        One evaluation-mode forward pass is enough: each quantizer initialises
        itself (MSE search) on its input the first time it is called, and that
        input has already passed through the quantizers before it, so the
        layers are calibrated in order, each seeing quantized inputs.
        """
        was_training = self.training
        self.eval()
        self(x)
        self.train(was_training)
        return self

    def average_weight_bits(self) -> float:
        """Parameter-weighted mean bit-width of all conv / linear weights."""
        total = bits = 0
        for layer in self.weight_layers():
            n = (layer.conv if isinstance(layer, QConvBN) else layer.linear).weight.numel()
            total += n
            bits += n * _bits(layer.wq)
        return bits / total


@torch.no_grad()
def recalibrate_bn(model: nn.Module, batches) -> None:
    """Re-estimate BatchNorm running statistics under quantization.

    Quantizing weights and activations shifts the mean and variance that each
    BatchNorm sees. Recomputing the running statistics on a few training
    batches (no gradient step) recovers part of the accuracy lost by
    post-training quantization at no cost.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    saved = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
    model.train()
    for i, x in enumerate(batches):
        # momentum 1/(i+1) makes the running statistics the plain average over
        # the batches seen so far (a cumulative moving average).
        for bn in bns:
            bn.momentum = 1.0 / (i + 1)
        model(x)
    for bn, mom in zip(bns, saved):
        bn.momentum = mom
    model.eval()
