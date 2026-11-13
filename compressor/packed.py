"""The compressed model as it is held in memory and run (deployment-time view).

`export_packed(qmodel)` turns a (calibrated or fine-tuned) `QMobileNetV2`
into a `PackedMobileNetV2`, whose entire state is:

    per conv / linear layer
        weight codes   b-bit integers, bit-packed into a uint8 buffer
        scale          one FP16 number per output channel
        bias           one FP16 number per output channel
        activation     its bit-width, sign and step (one FP32 number)
    per residual block one activation step for the sum
    globally           the input mean / std (6 numbers) and the pooled-feature step

BatchNorm disappears: with W = s_c * code (s_c the learned per-channel step)

    BN(conv(x, W)) = conv(x, code) * [s_c * gamma_c / sigma_c]  +  [beta_c - mu_c * gamma_c / sigma_c]
                                      `------ scale_c ------'      `---------- bias_c ----------'

so the four BatchNorm vectors and the quantizer step collapse into the scale
and the bias, and the integer codes are exactly the ones that were trained.

`PackedMobileNetV2` runs inference directly from this state, taking raw uint8
images as input. PyTorch has no convolution kernels for sub-byte integers, so
each layer's codes are expanded to a temporary float tensor while that layer
runs and released afterwards; what persists in RAM is only the packed state,
whose size `nbytes()` is measured from the actual buffers.
"""
import math

import torch
import torch.nn.functional as F

from mbv2.data import CIFAR10_MEAN, CIFAR10_STD

from .config import INPUT_BITS
from .packing import PACKED_META_BYTES, PackedTensor
from .qmodel import QConvBN, QMobileNetV2
from .quantizers import FULL_PRECISION, Quantizer
from .tracer import NullTracer

_NULL = NullTracer()
ACT_PARAM_BYTES = 8  # per activation quantizer: FP32 step + bit-width + sign, padded


class ActQuant:
    """Parameters of one activation quantizer (plain Python numbers)."""

    def __init__(self, bits: int, signed: bool, step: float):
        self.bits, self.signed, self.step = int(bits), bool(signed), float(step)
        self.qn = -(1 << (bits - 1)) if signed else 0
        self.qp = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1

    @classmethod
    def from_quantizer(cls, q: Quantizer):
        if not q.active:
            return None  # activation left in full precision
        return cls(q.bits, q.signed, float(q.step.detach().abs().clamp_min(1e-8)))

    def state(self):
        return {"bits": self.bits, "signed": self.signed, "step": self.step}


class PackedLayer:
    """One conv or linear layer in packed form."""

    def __init__(self, name, kind, weight, scale, bias, stride=1, padding=0, groups=1,
                 relu=False, act=None):
        self.name, self.kind = name, kind
        self.weight = weight      # PackedTensor (codes) or an FP32 tensor if not quantized
        self.scale = scale        # (C_out,) per-channel scale, or None if not quantized
        self.bias = bias          # (C_out,)
        self.stride, self.padding, self.groups = stride, padding, groups
        self.relu, self.act = relu, act

    @property
    def quantized(self) -> bool:
        return isinstance(self.weight, PackedTensor)

    @property
    def shape(self):
        return tuple(self.weight.shape)

    @property
    def weight_bits(self) -> int:
        return self.weight.bits if self.quantized else FULL_PRECISION

    @property
    def num_weights(self) -> int:
        return math.prod(self.shape)

    def code_bytes(self) -> int:
        """Bytes held by the weight values themselves."""
        return self.weight.nbytes if self.quantized else self.weight.numel() * 4

    def overhead_bytes(self) -> int:
        """Per-channel scales and biases plus the layer's bookkeeping."""
        n = self.bias.numel() * self.bias.element_size()
        if self.scale is not None:
            n += self.scale.numel() * self.scale.element_size()
        return n + PACKED_META_BYTES + (ACT_PARAM_BYTES if self.act is not None else 0)

    def float_weight(self, device) -> torch.Tensor:
        """Temporary FP32 weight tensor used while this layer executes."""
        if not self.quantized:
            return self.weight.to(device)
        w = self.weight.codes().to(device=device, dtype=torch.float32)
        return w * self.scale.to(device=device, dtype=torch.float32).view(-1, *([1] * (w.dim() - 1)))

    def state(self):
        w = self.weight
        return {"name": self.name, "kind": self.kind, "stride": self.stride,
                "padding": self.padding, "groups": self.groups, "relu": self.relu,
                "act": self.act.state() if self.act else None, "scale": self.scale,
                "bias": self.bias,
                "weight": ({"data": w.data, "bits": w.bits, "shape": w.shape, "qn": w.qn}
                           if self.quantized else w)}

    @classmethod
    def from_state(cls, s):
        w = s["weight"]
        if isinstance(w, dict):
            w = PackedTensor(w["data"], w["bits"], tuple(w["shape"]), w["qn"])
        act = ActQuant(**s["act"]) if s["act"] else None
        return cls(s["name"], s["kind"], w, s["scale"], s["bias"], s["stride"], s["padding"],
                   s["groups"], s["relu"], act)


def _pack_layer(layer, overhead_dtype) -> PackedLayer:
    """Fold BatchNorm into scale / bias and bit-pack the weight codes of one layer."""
    wq = layer.wq
    if isinstance(layer, QConvBN):
        conv, bn = layer.conv, layer.bn
        inv_std = torch.rsqrt(bn.running_var + bn.eps)
        bn_scale = (bn.weight * inv_std).detach().cpu()
        bias = (bn.bias - bn.running_mean * bn.weight * inv_std).detach().cpu()
        weight, geometry = conv.weight.detach(), dict(stride=conv.stride[0],
                                                      padding=conv.padding[0], groups=conv.groups)
        relu, act = layer.relu, ActQuant.from_quantizer(layer.aq)
    else:  # QLinear
        weight, bias = layer.linear.weight.detach(), layer.linear.bias.detach().cpu()
        bn_scale, geometry, relu, act = torch.ones(weight.shape[0]), {}, False, None
    if wq.active:
        codes = wq.codes(weight).cpu()
        packed = PackedTensor.from_codes(codes, wq.bits, wq.qn)
        scale = (wq.step.detach().abs().clamp_min(1e-8).cpu() * bn_scale).to(overhead_dtype)
    else:  # weights kept in full precision: fold BatchNorm into them directly
        packed = (weight.cpu() * bn_scale.view(-1, *([1] * (weight.dim() - 1)))).float()
        scale = None
    return PackedLayer(layer.name, layer.kind, packed, scale, bias.to(overhead_dtype),
                       relu=relu, act=act, **geometry)


@torch.no_grad()
def export_packed(qmodel: QMobileNetV2, overhead_dtype=torch.float16) -> "PackedMobileNetV2":
    """Convert a quantized training-time model into its packed in-memory form."""
    blocks = []
    for b in qmodel.blocks:
        blocks.append({
            "expand": _pack_layer(b.expand, overhead_dtype) if b.expand is not None else None,
            "depthwise": _pack_layer(b.depthwise, overhead_dtype),
            "project": _pack_layer(b.project, overhead_dtype),
            "use_res": b.use_res,
            "add_act": ActQuant.from_quantizer(b.aq_add) if b.use_res else None,
        })
    return PackedMobileNetV2(
        stem=_pack_layer(qmodel.stem, overhead_dtype), blocks=blocks,
        head=_pack_layer(qmodel.head, overhead_dtype),
        pool_act=ActQuant.from_quantizer(qmodel.aq_pool),
        classifier=_pack_layer(qmodel.classifier, overhead_dtype),
        config=qmodel.cfg.to_dict())


class PackedMobileNetV2:
    """Compressed MobileNet-v2: packed weights, per-channel scale / bias, activation steps."""

    def __init__(self, stem, blocks, head, pool_act, classifier, config=None,
                 mean=CIFAR10_MEAN, std=CIFAR10_STD):
        self.stem, self.blocks, self.head = stem, blocks, head
        self.pool_act, self.classifier = pool_act, classifier
        self.config = config or {}
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.pack_acts = False  # True: also bit-pack every activation (measurement mode)

    # ------------------------------------------------------------ structure
    def layers(self):
        out = [self.stem]
        for b in self.blocks:
            out += [l for l in (b["expand"], b["depthwise"], b["project"]) if l is not None]
        return out + [self.head, self.classifier]

    def eval(self):
        return self  # no training-time behaviour; lets it be used like an nn.Module

    # -------------------------------------------------------------- forward
    def _quant_act(self, act, y, name, t):
        """Quantize an activation tensor, report it to the tracer, return its value."""
        if act is None:
            t.alloc(name, y, FULL_PRECISION)
            return y
        codes = (y / act.step).clamp(act.qn, act.qp).round()
        nbytes = None
        if self.pack_acts:
            # Really store the tensor as packed b-bit codes and read it back.
            packed = PackedTensor.from_codes(codes.to(torch.int64).cpu(), act.bits, act.qn)
            nbytes = packed.nbytes / y.shape[0]
            codes = packed.codes().to(device=y.device, dtype=y.dtype)
        y = codes * act.step
        t.alloc(name, y, act.bits, nbytes)
        return y

    def _conv(self, layer, x, t):
        y = F.conv2d(x, layer.float_weight(x.device), layer.bias.to(x.device, torch.float32),
                     layer.stride, layer.padding, 1, layer.groups)
        if layer.relu:
            y = F.relu6(y)
        return self._quant_act(layer.act, y, layer.name, t)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, tracer=None) -> torch.Tensor:
        """Logits for a batch of raw uint8 images of shape (N, 3, 32, 32)."""
        t = tracer or _NULL
        if images.dtype != torch.uint8:
            raise TypeError("PackedMobileNetV2 takes raw uint8 images; it normalises them itself")
        dev = images.device
        nbytes_in = images[0].numel() if self.pack_acts else None
        t.alloc("input", images, INPUT_BITS if self.head.act is not None else FULL_PRECISION,
                nbytes_in)
        x = images.float().div_(255.0)
        x = (x - self.mean.to(dev).view(1, 3, 1, 1)) / self.std.to(dev).view(1, 3, 1, 1)
        x = self._conv(self.stem, x, t)
        t.free("input")
        name = "stem"
        for i, b in enumerate(self.blocks):
            n, skip, skip_name, cur = f"blocks.{i}", x, name, name
            if b["expand"] is not None:
                x = self._conv(b["expand"], x, t)
                if not b["use_res"]:
                    t.free(skip_name)
                cur = f"{n}.expand"
            x = self._conv(b["depthwise"], x, t)
            if cur != skip_name or not b["use_res"]:
                t.free(cur)
            x = self._conv(b["project"], x, t)
            t.free(f"{n}.depthwise")
            name = f"{n}.project"
            if b["use_res"]:
                x = self._quant_act(b["add_act"], skip + x, f"{n}.add", t)
                t.free(f"{n}.project")
                t.free(skip_name)
                name = f"{n}.add"
        x = self._conv(self.head, x, t)
        t.free(name)
        x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        x = self._quant_act(self.pool_act, x, "pool", t)
        t.free("head")
        c = self.classifier
        x = F.linear(x, c.float_weight(dev), c.bias.to(dev, torch.float32))
        t.alloc("logits", x, FULL_PRECISION)
        t.free("pool")
        return x

    # ----------------------------------------------------------------- size
    def size_report(self) -> dict:
        """In-memory size of the compressed model, split into its parts (bytes)."""
        rows, code_bytes, scale_bytes, bias_bytes = [], 0, 0, 0
        for l in self.layers():
            sb = l.scale.numel() * l.scale.element_size() if l.scale is not None else 0
            bb = l.bias.numel() * l.bias.element_size()
            rows.append({"name": l.name, "kind": l.kind, "shape": list(l.shape),
                         "weights": l.num_weights, "bits": l.weight_bits,
                         "code_bytes": l.code_bytes(), "overhead_bytes": l.overhead_bytes(),
                         "act_bits": l.act.bits if l.act else FULL_PRECISION})
            code_bytes, scale_bytes, bias_bytes = code_bytes + l.code_bytes(), scale_bytes + sb, bias_bytes + bb
        n_act = (sum(1 for l in self.layers() if l.act is not None)
                 + sum(1 for b in self.blocks if b["add_act"] is not None)
                 + (1 if self.pool_act is not None else 0))
        act_bytes = n_act * ACT_PARAM_BYTES
        meta_bytes = len(self.layers()) * PACKED_META_BYTES + 6 * 4  # + input mean / std
        total = code_bytes + scale_bytes + bias_bytes + act_bytes + meta_bytes
        n_weights = sum(r["weights"] for r in rows)
        return {"layers": rows, "num_weights": n_weights,
                "weight_code_bytes": code_bytes, "scale_bytes": scale_bytes,
                "bias_bytes": bias_bytes, "act_param_bytes": act_bytes, "meta_bytes": meta_bytes,
                "overhead_bytes": total - code_bytes, "total_bytes": total,
                "avg_weight_bits": 8.0 * code_bytes / n_weights}

    def nbytes(self) -> int:
        return self.size_report()["total_bytes"]

    # ------------------------------------------------------------ save / load
    def state(self) -> dict:
        return {"format": "packed-mobilenetv2-v1", "config": self.config,
                "mean": self.mean, "std": self.std,
                "stem": self.stem.state(), "head": self.head.state(),
                "classifier": self.classifier.state(),
                "pool_act": self.pool_act.state() if self.pool_act else None,
                "blocks": [{"expand": b["expand"].state() if b["expand"] else None,
                            "depthwise": b["depthwise"].state(), "project": b["project"].state(),
                            "use_res": b["use_res"],
                            "add_act": b["add_act"].state() if b["add_act"] else None}
                           for b in self.blocks]}

    @classmethod
    def from_state(cls, s: dict) -> "PackedMobileNetV2":
        blocks = [{"expand": PackedLayer.from_state(b["expand"]) if b["expand"] else None,
                   "depthwise": PackedLayer.from_state(b["depthwise"]),
                   "project": PackedLayer.from_state(b["project"]), "use_res": b["use_res"],
                   "add_act": ActQuant(**b["add_act"]) if b["add_act"] else None}
                  for b in s["blocks"]]
        return cls(PackedLayer.from_state(s["stem"]), blocks, PackedLayer.from_state(s["head"]),
                   ActQuant(**s["pool_act"]) if s["pool_act"] else None,
                   PackedLayer.from_state(s["classifier"]), s.get("config"),
                   s["mean"].tolist(), s["std"].tolist())
