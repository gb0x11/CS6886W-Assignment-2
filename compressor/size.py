"""Size accounting: full-precision reference, compression ratios, activation memory.

All sizes are in-memory sizes in bytes. "MB" in the reports means 2^20 bytes
(the convention of the course template, which divides by 1024 * 1024).
"""
import torch
import torch.nn as nn

from .tracer import ActivationTracer

MB = 1024 * 1024


def fp32_model_bytes(model: nn.Module) -> dict:
    """In-memory size of the uncompressed model, measured from its tensors.

    Counts every parameter and every buffer (BatchNorm running statistics
    included, because they are needed at inference time).
    """
    weights = sum(m.weight.numel() * m.weight.element_size() for m in model.modules()
                  if isinstance(m, (nn.Conv2d, nn.Linear)))
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return {"weight_bytes": weights, "param_bytes": params, "buffer_bytes": buffers,
            "total_bytes": params + buffers,
            "num_weights": sum(m.weight.numel() for m in model.modules()
                               if isinstance(m, (nn.Conv2d, nn.Linear)))}


@torch.no_grad()
def trace_activations(model, x: torch.Tensor) -> ActivationTracer:
    """Run one forward pass and record every stored activation tensor.

    `model` is a QMobileNetV2 (x: normalised float batch) or a
    PackedMobileNetV2 (x: raw uint8 batch).
    """
    tracer = ActivationTracer()
    model.eval()
    model(x, tracer=tracer)
    return tracer


@torch.no_grad()
def fp32_activation_elements(float_model: nn.Module, x: torch.Tensor) -> dict:
    """Independent count of the activation tensors of the uncompressed model.

    Forward hooks are placed on the float model itself (one per conv-BN-act
    unit, per residual block that adds, on the pooling layer and on the
    classifier), and the input is added. This is the FP32 reference the
    compressed activations are compared with, obtained without any of the
    quantization code.
    """
    from mbv2.model import ConvBN, InvertedResidual  # local import: avoids a cycle

    sizes = {"input": x[0].numel()}
    hooks = []

    def record(name):
        return lambda _m, _i, out: sizes.__setitem__(name, out[0].numel())

    for name, m in float_model.named_modules():
        if isinstance(m, ConvBN) or isinstance(m, (nn.AdaptiveAvgPool2d, nn.Linear)):
            hooks.append(m.register_forward_hook(record(name)))
        elif isinstance(m, InvertedResidual) and m.use_res:
            hooks.append(m.register_forward_hook(record(name + ".add")))
    float_model.eval()
    float_model(x)
    for h in hooks:
        h.remove()
    elements = sum(sizes.values())
    return {"tensors": len(sizes), "elements": elements, "bytes": elements * 4, "sizes": sizes}


def compression_summary(packed, float_model, tracer: ActivationTracer) -> dict:
    """All size numbers of one compressed model, relative to the FP32 model."""
    ref = fp32_model_bytes(float_model)
    rep = packed.size_report()
    act = tracer.summary()
    out = {
        # model = everything that must be in memory to run inference
        "fp32_model_bytes": ref["total_bytes"],
        "model_bytes": rep["total_bytes"],
        "model_size_mb": rep["total_bytes"] / MB,
        "fp32_model_size_mb": ref["total_bytes"] / MB,
        "compression_ratio": ref["total_bytes"] / rep["total_bytes"],
        # weights = conv / linear weight tensors only
        "fp32_weight_bytes": ref["weight_bytes"],
        "weight_bytes": rep["weight_code_bytes"],
        "weight_compression_ratio": ref["weight_bytes"] / rep["weight_code_bytes"],
        "avg_weight_bits": rep["avg_weight_bits"],
        # overheads kept next to the codes
        "overhead_bytes": rep["overhead_bytes"],
        "scale_bytes": rep["scale_bytes"], "bias_bytes": rep["bias_bytes"],
        "act_param_bytes": rep["act_param_bytes"], "meta_bytes": rep["meta_bytes"],
        "overhead_fraction": rep["overhead_bytes"] / rep["total_bytes"],
    }
    out.update({
        "act_compression_ratio": act["act_ratio"],
        "act_peak_compression_ratio": act["act_peak_ratio"],
        "act_total_mb": act["act_total_bytes"] / MB,
        "act_total_mb_fp32": act["act_total_bytes_fp32"] / MB,
        "act_peak_mb": act["act_peak_bytes"] / MB,
        "act_peak_mb_fp32": act["act_peak_bytes_fp32"] / MB,
        "act_avg_bits": act["act_avg_bits"],
    })
    return out
