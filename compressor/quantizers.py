"""Uniform quantizers with a learned step size, written from scratch.

A b-bit uniform quantizer maps a real tensor x to integer codes and back:

    code  = clamp(round(x / s), qn, qp)          (what is stored, b bits each)
    x_hat = s * code                             (what the next layer sees)

    signed   (weights, linear activations):  qn = -2^(b-1),  qp = 2^(b-1) - 1
    unsigned (activations after ReLU6):      qn = 0,         qp = 2^b - 1

The step s is a parameter. It is initialised by a search that minimises the
mean squared quantization error on the data (the weights, or a calibration
batch of activations), which already gives a good post-training quantizer.
It can then be trained together with the weights (quantization-aware
training) with the straight-through estimator, following "Learned Step Size
Quantization" (Esser et al., ICLR 2020).

No function from torch.ao.quantization / torch.quantization (or any other
compression library) is used anywhere in this package.
"""
import math

import torch
import torch.nn as nn

FULL_PRECISION = 32  # a quantizer with bits >= 32 is the identity


def qrange(bits: int, signed: bool):
    """Integer code range (qn, qp) of a b-bit quantizer."""
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """round(x) in the forward pass, identity gradient in the backward pass."""
    return (x.round() - x).detach() + x


def grad_scale(x: torch.Tensor, g) -> torch.Tensor:
    """Return x unchanged but scale its gradient by g (LSQ step-size gradient scaling)."""
    return (x - x * g).detach() + x * g


class _LearnedStepQuantize(torch.autograd.Function):
    """out = s * clamp(round(x / s), qn, qp) with hand-written gradients.

    Gradients (straight-through estimator for the rounding):
        d out / d x = 1                  if qn <= x/s <= qp, else 0
        d out / d s = round(x/s) - x/s   if qn <= x/s <= qp, else qn or qp
    The step gradient is multiplied by `g` (LSQ gradient scaling). Writing the
    backward pass by hand stores only a mask and one tensor per quantizer,
    instead of the five or six intermediate tensors that the equivalent chain
    of autograd operations (`round_ste`, `grad_scale`, ...) keeps alive, which
    roughly halves the memory of quantization-aware training.
    """

    @staticmethod
    def forward(ctx, x, s, qn, qp, g):
        v = x / s
        codes = v.clamp(qn, qp).round_()
        inside = (v >= qn) & (v <= qp)
        ctx.save_for_backward(inside, torch.where(inside, codes - v, codes))
        ctx.g, ctx.s_shape = g, s.shape
        return codes * s

    @staticmethod
    def backward(ctx, grad_out):
        inside, d_step = ctx.saved_tensors
        grad_x = grad_out * inside
        grad_s = grad_out * d_step
        if len(ctx.s_shape) == 0:
            grad_s = grad_s.sum()
        else:  # per-channel step of shape (C, 1, ..., 1)
            grad_s = grad_s.sum(dim=tuple(range(1, grad_s.dim())), keepdim=True)
        return grad_x, grad_s * ctx.g, None, None, None


@torch.no_grad()
def mse_optimal_step(x: torch.Tensor, qn: int, qp: int, per_channel: bool = False,
                     n_grid: int = 60) -> torch.Tensor:
    """Step size that minimises ||x - s*clamp(round(x/s), qn, qp)||^2.

    The search scans clipping thresholds c = r * max|x| for r on a grid in
    (0, 1] and keeps the best one; s = c / max(|qn|, qp). With
    `per_channel=True`, x has shape (C, n) and one step per row is returned.
    Clipping a few outliers (r < 1) is what makes low bit-widths usable.
    """
    if not per_channel:
        x = x.reshape(1, -1)
    x = x.float()
    max_abs = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    denom = float(max(abs(qn), qp))
    best_err = torch.full_like(max_abs, float("inf"))
    best_step = max_abs / denom
    for r in torch.linspace(1.0, 0.1, n_grid).tolist():
        step = max_abs * r / denom
        err = (x - (x / step).round().clamp(qn, qp) * step).pow(2).sum(dim=1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_step = torch.where(better, step, best_step)
    return best_step.squeeze(1) if per_channel else best_step.reshape(())


class Quantizer(nn.Module):
    """b-bit uniform quantizer with a learned step.

    Args:
        bits:     bit-width; >= 32 means "not quantized" (identity).
        signed:   signed codes (weights, linear activations) or unsigned
                  codes (activations after ReLU6).
        channels: if given, one step per output channel (axis 0 of the
                  tensor), used for weights; otherwise a single step.
    """

    def __init__(self, bits: int, signed: bool, channels: int = None):
        super().__init__()
        self.bits, self.signed, self.channels = int(bits), bool(signed), channels
        self.qn, self.qp = qrange(min(self.bits, 16), signed)
        self.step = nn.Parameter(torch.ones(channels if channels else ()))
        # LSQ gradient scale of the step and the "step has been initialised"
        # flag are buffers so that they survive saving / loading a checkpoint.
        self.register_buffer("gscale", torch.tensor(1.0))
        self.register_buffer("initialized", torch.tensor(False))
        self._ready = False   # Python-side mirror of `initialized` (avoids a device sync per call)
        self.enabled = True   # False: act as identity (full-precision reference)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._ready = bool(self.initialized)

    @property
    def active(self) -> bool:
        return self.enabled and self.bits < FULL_PRECISION

    def extra_repr(self):
        return f"bits={self.bits}, signed={self.signed}, channels={self.channels}"

    def _view(self, x):
        """Step reshaped to broadcast along axis 0 (per-channel) or as a scalar."""
        s = self.step.abs().clamp_min(1e-8)
        return s.view(-1, *([1] * (x.dim() - 1))) if self.channels else s

    @torch.no_grad()
    def init_from(self, x: torch.Tensor, max_samples: int = 2_000_000) -> None:
        """Initialise the step from data with the MSE search."""
        if self.channels:
            step = mse_optimal_step(x.reshape(x.shape[0], -1), self.qn, self.qp, per_channel=True)
            n_per_step = x[0].numel()
        else:
            flat = x.reshape(-1)
            if flat.numel() > max_samples:  # a random subset is enough for one scalar
                flat = flat[torch.randint(0, flat.numel(), (max_samples,), device=flat.device)]
            step = mse_optimal_step(flat, self.qn, self.qp)
            n_per_step = x[0].numel() if x.dim() > 1 else x.numel()
        self.step.copy_(step.to(self.step.device, self.step.dtype))
        # LSQ: scale the step gradient by 1 / sqrt(N * qp) so that its updates
        # are commensurate with the weight updates.
        self.gscale.fill_(1.0 / math.sqrt(n_per_step * max(self.qp, 1)))
        self.initialized.fill_(True)
        self._ready = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return x
        if not self._ready:
            self.init_from(x.detach())
        s = self._view(x)
        if torch.is_grad_enabled() and (x.requires_grad or self.step.requires_grad):
            return _LearnedStepQuantize.apply(x, s, self.qn, self.qp, self.gscale)
        return (x / s).clamp_(self.qn, self.qp).round_() * s

    @torch.no_grad()
    def codes(self, x: torch.Tensor) -> torch.Tensor:
        """Integer codes in [qn, qp] (int32) of x under the current step."""
        return (x / self._view(x)).clamp(self.qn, self.qp).round().to(torch.int32)
