"""Compression configuration: which bit-width every tensor of MobileNet-v2 gets."""
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

INPUT_BITS = 8  # CIFAR-10 pixels are 8-bit integers, so the input needs no more


@dataclass
class CompressionConfig:
    """Bit-widths of the weights and activations.

    Weights
        w_bits        default for every conv / linear weight
        w_bits_first  stem convolution           (default: w_bits)
        w_bits_dw     depthwise convolutions     (default: w_bits)
        w_bits_last   classifier                 (default: w_bits)
        layer_w_bits  per-layer overrides {layer name: bits}, e.g. produced by
                      the sensitivity-based mixed-precision allocator
    Activations
        a_bits         outputs of ReLU6 layers (the large feature maps)
        a_bits_linear  linear-bottleneck outputs, residual sums and the pooled
                       feature vector (signed / small tensors; default: a_bits)
    A value of 32 means "keep in full precision".
    """
    w_bits: int = 8
    a_bits: int = 8
    w_bits_first: Optional[int] = None
    w_bits_dw: Optional[int] = None
    w_bits_last: Optional[int] = None
    a_bits_linear: Optional[int] = None
    layer_w_bits: Dict[str, int] = field(default_factory=dict)

    def weight_bits(self, name: str, kind: str) -> int:
        """Bit-width of the weights of layer `name` whose role is `kind`."""
        if name in self.layer_w_bits:
            return int(self.layer_w_bits[name])
        by_kind = {"stem": self.w_bits_first, "depthwise": self.w_bits_dw,
                   "classifier": self.w_bits_last}
        return int(by_kind.get(kind) or self.w_bits)

    def act_bits(self, relu: bool) -> int:
        """Bit-width of an activation tensor (after ReLU6, or a linear one)."""
        return int(self.a_bits if relu else (self.a_bits_linear or self.a_bits))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CompressionConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
