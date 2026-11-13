"""MobileNet-v2 for CIFAR-10 (32x32 inputs).

The architecture follows Sandler et al., "MobileNetV2: Inverted Residuals and
Linear Bottlenecks" (CVPR 2018) and the layer layout of the torchvision
reference implementation, written from scratch here so that every
conv + BatchNorm (+ ReLU6) unit is one explicit `ConvBN` module. That makes
the later compression step (BN folding, weight / activation quantization)
a simple one-to-one replacement of `ConvBN` units.

CIFAR adaptation: ImageNet MobileNet-v2 downsamples by 32, which would shrink
a 32x32 image to 1x1. As is standard for CIFAR, the stem and the second
stage use stride 1, so the network downsamples by 8 and the last feature map
is 4x4. Everything else (expansion ratio 6, widths, depths) is unchanged.
"""
import torch
import torch.nn as nn

# (expansion t, output channels c, repeats n, stride s of the first repeat)
# ImageNet uses s = 2 for the second row and for the stem; CIFAR uses 1.
CIFAR_SETTING = [
    (1, 16, 1, 1),
    (6, 24, 2, 1),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


def make_divisible(v: float, divisor: int = 8) -> int:
    """Round a channel count to the nearest multiple of `divisor` (never
    going below 90 % of `v`), as in the original MobileNet code."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBN(nn.Module):
    """Conv2d (no bias) -> BatchNorm2d -> optional ReLU6.

    `kind` records the role of the layer ("stem", "expand", "depthwise",
    "project", "head"); the compression code uses it to pick bit-widths.
    """

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, groups=1, act=True, kind="conv"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=kernel_size // 2,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True) if act else None
        self.kind = kind

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x if self.act is None else self.act(x)


class InvertedResidual(nn.Module):
    """1x1 expand (ReLU6) -> 3x3 depthwise (ReLU6) -> 1x1 linear projection.

    The projection has no non-linearity ("linear bottleneck"). A residual
    connection is used when the block keeps resolution and channel count.
    """

    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = stride == 1 and in_ch == out_ch
        # With expansion 1 (first block) there is no expand convolution.
        self.expand = ConvBN(in_ch, hidden, 1, kind="expand") if expand_ratio != 1 else None
        self.depthwise = ConvBN(hidden, hidden, 3, stride, groups=hidden, kind="depthwise")
        self.project = ConvBN(hidden, out_ch, 1, act=False, kind="project")

    def forward(self, x):
        out = x if self.expand is None else self.expand(x)
        out = self.project(self.depthwise(out))
        return x + out if self.use_res else out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.2, setting=None):
        super().__init__()
        setting = setting or CIFAR_SETTING
        in_ch = make_divisible(32 * width_mult)
        # The last layer keeps 1280 channels for width_mult <= 1 (as in the paper).
        self.last_ch = make_divisible(1280 * max(1.0, width_mult))
        self.width_mult, self.dropout_p = width_mult, dropout

        self.stem = ConvBN(3, in_ch, 3, stride=1, kind="stem")
        blocks = []
        for t, c, n, s in setting:
            out_ch = make_divisible(c * width_mult)
            for i in range(n):
                blocks.append(InvertedResidual(in_ch, out_ch, s if i == 0 else 1, t))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.head = ConvBN(in_ch, self.last_ch, 1, kind="head")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.last_ch, num_classes)
        self._init_weights()

    def _init_weights(self):
        # Same initialization as the reference implementation.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.head(self.blocks(self.stem(x)))
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(self.dropout(x))


def mobilenet_v2(num_classes=10, width_mult=1.0, dropout=0.2) -> MobileNetV2:
    return MobileNetV2(num_classes, width_mult, dropout)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
