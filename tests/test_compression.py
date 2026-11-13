"""Unit tests of the compression code (run on CPU, no dataset needed).

    python -m pytest tests -q
"""
import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compressor.config import CompressionConfig  # noqa: E402
from compressor.packed import PackedMobileNetV2, export_packed  # noqa: E402
from compressor.packing import PackedTensor, pack_bits, unpack_bits  # noqa: E402
from compressor.qmodel import QMobileNetV2  # noqa: E402
from compressor.quantizers import (Quantizer, grad_scale, mse_optimal_step, qrange,  # noqa: E402
                                    round_ste)
from compressor.size import (compression_summary, fp32_activation_elements,  # noqa: E402
                              fp32_model_bytes, trace_activations)
from mbv2.data import CIFAR10_MEAN, CIFAR10_STD  # noqa: E402
from mbv2.model import mobilenet_v2  # noqa: E402


def random_model(seed=0):
    """A MobileNet-v2 with random weights and non-trivial BatchNorm statistics."""
    torch.manual_seed(seed)
    m = mobilenet_v2()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.3, 0.3)
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 1.5)
    m.classifier.bias.data.uniform_(-0.1, 0.1)
    return m.eval()


def images(n=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 3, 32, 32), generator=g, dtype=torch.uint8)


def normalise(u8):
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (u8.float().div(255.0) - mean) / std


# ------------------------------------------------------------------ bit packing
@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8, 11, 16])
def test_pack_roundtrip_and_size(bits):
    g = torch.Generator().manual_seed(bits)
    for n in (1, 7, 8, 9, 1000, 4097):
        v = torch.randint(0, 1 << bits, (n,), generator=g)
        packed = pack_bits(v, bits)
        assert packed.dtype == torch.uint8
        assert packed.numel() == math.ceil(n * bits / 8)       # exactly b bits per value
        assert torch.equal(unpack_bits(packed, bits, n), v)    # lossless


def test_pack_rejects_out_of_range():
    with pytest.raises(ValueError):
        pack_bits(torch.tensor([4]), 2)
    with pytest.raises(ValueError):
        pack_bits(torch.tensor([-1]), 2)


@pytest.mark.parametrize("bits,signed", [(2, True), (3, True), (4, False), (8, True)])
def test_packed_tensor_signed_codes(bits, signed):
    qn, qp = qrange(bits, signed)
    codes = torch.randint(qn, qp + 1, (5, 3, 3, 3))
    pt = PackedTensor.from_codes(codes, bits, qn)
    assert pt.nbytes == math.ceil(codes.numel() * bits / 8)
    assert torch.equal(pt.codes(), codes)


# ------------------------------------------------------------------- quantizers
def test_qrange():
    assert qrange(2, True) == (-2, 1) and qrange(8, True) == (-128, 127)
    assert qrange(2, False) == (0, 3) and qrange(8, False) == (0, 255)


@pytest.mark.parametrize("bits,signed", [(2, True), (4, True), (4, False), (8, False)])
def test_quantizer_outputs_lie_on_the_grid(bits, signed):
    torch.manual_seed(0)
    x = torch.randn(4, 16, 8, 8)
    x = x.abs() if not signed else x
    q = Quantizer(bits, signed).eval()
    y = q(x)
    codes = y / q.step.abs()
    assert torch.allclose(codes, codes.round(), atol=1e-4)
    assert codes.min() >= q.qn - 1e-4 and codes.max() <= q.qp + 1e-4
    assert y.unique().numel() <= (1 << bits)
    assert torch.equal(q.codes(x).float(), codes.round())


def test_quantizer_error_shrinks_with_bits():
    torch.manual_seed(0)
    x = torch.randn(10000)
    errs = [(Quantizer(b, True).eval()(x) - x).pow(2).mean().item() for b in (2, 3, 4, 6, 8)]
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 1e-3


def test_mse_step_beats_minmax_at_low_bits():
    torch.manual_seed(0)
    x = torch.randn(20000)
    qn, qp = qrange(3, True)
    s_mse = mse_optimal_step(x, qn, qp)
    s_max = x.abs().max() / max(abs(qn), qp)

    def err(s):
        return (x - (x / s).round().clamp(qn, qp) * s).pow(2).mean()
    assert err(s_mse) < err(s_max)


def test_per_channel_steps_follow_channel_ranges():
    torch.manual_seed(0)
    w = torch.randn(4, 8, 3, 3) * torch.tensor([0.01, 0.1, 1.0, 10.0]).view(4, 1, 1, 1)
    q = Quantizer(4, True, channels=4)
    q.init_from(w)
    s = q.step.detach()
    assert (s[1:] > s[:-1] * 3).all()   # steps scale with the channel magnitude
    rel = ((q.eval()(w) - w).flatten(1).norm(dim=1) / w.flatten(1).norm(dim=1))
    assert rel.max() < 0.2              # every channel is represented well


def test_straight_through_gradients():
    torch.manual_seed(0)
    x = torch.randn(64, requires_grad=True)
    q = Quantizer(3, True).train()
    q.init_from(x.detach())
    q(x).sum().backward()
    inside = (x.detach() / q.step.detach()).abs() < q.qp
    assert torch.allclose(x.grad[inside], torch.ones_like(x.grad[inside]))   # STE inside the range
    assert q.step.grad is not None and q.step.grad.abs() > 0                 # the step is learned


@pytest.mark.parametrize("bits,signed,channels", [(2, True, None), (4, False, None), (3, True, 6)])
def test_handwritten_gradients_match_autograd_ste(bits, signed, channels):
    """The custom backward pass equals the textbook STE chain built from autograd ops."""
    torch.manual_seed(0)
    x = torch.randn(6, 5, 4, 4) * 2
    x = (x.abs() if not signed else x).requires_grad_(True)
    q = Quantizer(bits, signed, channels=channels).train()
    q.init_from(x.detach())
    q.step.data *= 0.5                      # make sure part of x is clipped
    weight = torch.randn_like(x)            # arbitrary downstream gradient

    out = q(x)
    (out * weight).sum().backward()
    gx, gs = x.grad.clone(), q.step.grad.clone()

    x.grad = None
    q.step.grad = None
    s = grad_scale(q._view(x), q.gscale)    # reference: LSQ written with autograd ops
    ref = round_ste((x / s).clamp(q.qn, q.qp)) * s
    (ref * weight).sum().backward()
    assert torch.allclose(out, ref)
    assert torch.allclose(gx, x.grad, atol=1e-6)
    assert torch.allclose(gs, q.step.grad, rtol=1e-4, atol=1e-7)


def test_full_precision_quantizer_is_identity():
    x = torch.randn(3, 5)
    assert torch.equal(Quantizer(32, True)(x), x)
    q = Quantizer(4, True)
    q.enabled = False
    assert torch.equal(q(x), x)


# --------------------------------------------------------- quantized MobileNet-v2
def test_qmodel_with_quantization_off_equals_float_model():
    m = random_model()
    x = normalise(images())
    q = QMobileNetV2.from_float(m, CompressionConfig(w_bits=4, a_bits=4)).eval().set_quant(False, False)
    assert torch.allclose(q(x), m(x), atol=1e-5)


def test_bit_widths_follow_the_config():
    cfg = CompressionConfig(w_bits=3, a_bits=4, w_bits_first=8, w_bits_dw=5, w_bits_last=7,
                            a_bits_linear=6, layer_w_bits={"head": 2})
    q = QMobileNetV2.from_float(random_model(), cfg)
    assert q.stem.wq.bits == 8 and q.classifier.wq.bits == 7 and q.head.wq.bits == 2
    assert q.blocks[1].depthwise.wq.bits == 5 and q.blocks[1].expand.wq.bits == 3
    assert q.blocks[1].expand.aq.bits == 4 and not q.blocks[1].expand.aq.signed
    assert q.blocks[1].project.aq.bits == 6 and q.blocks[1].project.aq.signed
    assert q.blocks[2].aq_add.bits == 6 and q.aq_pool.bits == 6


@pytest.mark.parametrize("w_bits,a_bits", [(8, 8), (4, 4), (2, 3)])
def test_packed_model_reproduces_the_quantized_model(w_bits, a_bits):
    """BatchNorm folding + bit-packing + FP16 overheads change (almost) nothing."""
    m = random_model()
    u8 = images(16)
    q = QMobileNetV2.from_float(m, CompressionConfig(w_bits=w_bits, a_bits=a_bits)).eval()
    q.calibrate(normalise(u8))
    ref = q(normalise(u8))
    # With FP32 overheads only float round-off remains: a few activations that
    # sit on a rounding boundary flip by one code (bit-exact at low bit-widths,
    # below 2 % of the logit norm at 8 bits) and no prediction changes.
    p32 = export_packed(q, torch.float32)
    out32 = p32(u8)
    assert (out32 - ref).norm() / ref.norm() < 2e-2
    assert torch.equal(out32.argmax(1), ref.argmax(1))
    # FP16 overheads: every scale / bias is the FP32 value rounded to half
    # precision (relative error below 2^-11), the weight codes are identical,
    # and the logits stay close even for this untrained, chaotic network.
    p16 = export_packed(q)
    for l16, l32 in zip(p16.layers(), p32.layers()):
        assert l16.scale.dtype == l16.bias.dtype == torch.float16
        assert torch.equal(l16.weight.data, l32.weight.data)
        assert torch.allclose(l16.scale.float(), l32.scale, rtol=6e-4, atol=1e-7)
        assert torch.allclose(l16.bias.float(), l32.bias, rtol=6e-4, atol=1e-7)
    assert (p16(u8) - ref).norm() / ref.norm() < 0.25


def test_packing_activations_does_not_change_the_output():
    m = random_model()
    u8 = images(2)
    q = QMobileNetV2.from_float(m, CompressionConfig(w_bits=4, a_bits=3, a_bits_linear=6)).eval()
    q.calibrate(normalise(u8))
    packed = export_packed(q)
    ref = packed(u8)
    packed.pack_acts = True
    assert torch.equal(packed(u8), ref)


def test_save_load_roundtrip(tmp_path):
    m = random_model()
    u8 = images(4)
    q = QMobileNetV2.from_float(m, CompressionConfig(w_bits=3, a_bits=4, w_bits_first=8)).eval()
    q.calibrate(normalise(u8))
    packed = export_packed(q)
    path = tmp_path / "packed.pt"
    torch.save(packed.state(), path)
    loaded = PackedMobileNetV2.from_state(torch.load(path, weights_only=False))
    assert torch.equal(loaded(u8), packed(u8))
    assert loaded.nbytes() == packed.nbytes()


# -------------------------------------------------------------- size accounting
def test_fp32_reference_size():
    m = random_model()
    ref = fp32_model_bytes(m)
    assert ref["num_weights"] == 2_202_560
    assert ref["param_bytes"] == 2_236_682 * 4
    # running mean / var (float32) + num_batches_tracked (int64) of 52 BatchNorm layers
    assert ref["buffer_bytes"] == 2 * 17_056 * 4 + 52 * 8


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_packed_size_matches_hand_count(bits):
    m = random_model()
    u8 = images(2)
    q = QMobileNetV2.from_float(m, CompressionConfig(w_bits=bits, a_bits=8)).eval()
    q.calibrate(normalise(u8))
    rep = export_packed(q).size_report()
    # Each layer is packed separately, so each one is rounded up to a whole byte.
    expected_codes = sum(math.ceil(l["weights"] * bits / 8) for l in rep["layers"])
    assert rep["weight_code_bytes"] == expected_codes
    assert abs(rep["avg_weight_bits"] - bits) < 0.01
    channels = 17_056 + 10                       # conv output channels + classifier rows
    assert rep["scale_bytes"] == channels * 2 and rep["bias_bytes"] == channels * 2   # FP16
    assert rep["total_bytes"] == (rep["weight_code_bytes"] + rep["scale_bytes"] + rep["bias_bytes"]
                                  + rep["act_param_bytes"] + rep["meta_bytes"])
    assert len(rep["layers"]) == 53              # 52 convolutions + 1 linear layer


def test_activation_accounting():
    m = random_model()
    u8 = images(1)
    x = normalise(u8)
    cfg = CompressionConfig(w_bits=4, a_bits=4, a_bits_linear=8)
    q = QMobileNetV2.from_float(m, cfg).eval()
    q.calibrate(x)
    packed = export_packed(q)

    hooks = fp32_activation_elements(m, x)        # independent count on the float model
    t_q = trace_activations(q, x).summary()
    t_p = trace_activations(packed, u8).summary()
    assert t_q["act_elements"] == t_p["act_elements"] == hooks["elements"]
    assert t_q["act_total_bytes_fp32"] == hooks["bytes"]
    assert t_q == t_p                             # training-time and packed views agree

    # 4-bit ReLU6 maps, 8-bit linear maps / input / pooled vector, FP32 logits
    relu = sum(n for k, n in hooks["sizes"].items()
               if k == "stem" or k == "head" or k.endswith(("expand", "depthwise")))
    logits = 10
    small = hooks["elements"] - relu - logits
    assert t_q["act_total_bytes"] == (relu * 4 + small * 8 + logits * 32) / 8
    assert t_q["act_peak_bytes"] < t_q["act_total_bytes"]
    assert t_q["act_peak_bytes_fp32"] == (2 * 147_456 + 24_576) * 4   # block 2: in + out + skip

    # Measured: really bit-pack every activation of one image and add up the bytes.
    packed.pack_acts = True
    tr = trace_activations(packed, u8)
    measured = sum(r["packed_bytes"] for r in tr.records if r["packed_bytes"] is not None)
    assert abs(measured + logits * 4 - t_q["act_total_bytes"]) <= len(tr.records)  # byte padding


def test_compression_summary_is_consistent():
    m = random_model()
    u8 = images(1)
    q = QMobileNetV2.from_float(m, CompressionConfig(w_bits=4, a_bits=4)).eval()
    q.calibrate(normalise(u8))
    packed = export_packed(q)
    s = compression_summary(packed, m, trace_activations(packed, u8))
    assert 7.0 < s["compression_ratio"] < 8.0 < s["weight_compression_ratio"] + 1e-6
    assert abs(s["weight_compression_ratio"] - 8.0) < 0.01
    assert s["model_bytes"] == packed.nbytes()
    assert 0 < s["overhead_fraction"] < 0.1


# ------------------------------------------------------- mixed-precision allocator
def test_allocator_gives_fewer_bits_to_large_insensitive_layers():
    from compressor.sensitivity import allocate_bits
    choices = (2, 4, 8)
    sens = {"sizes": {"small_sensitive": 100, "big_robust": 10_000, "big_sensitive": 10_000},
            "kl": {"small_sensitive": {2: 5.0, 4: 1.0, 8: 0.0},
                   "big_robust": {2: 0.02, 4: 0.01, 8: 0.0},
                   "big_sensitive": {2: 9.0, 4: 0.5, 8: 0.0}}}
    bits = allocate_bits(sens, target_avg_bits=4.0)
    assert bits["big_robust"] == 2 and bits["small_sensitive"] == 8
    avg = sum(sens["sizes"][n] * b for n, b in bits.items()) / sum(sens["sizes"].values())
    assert avg <= 4.0 and set(bits.values()) <= set(choices)
    # a fixed layer keeps its bit-width
    assert allocate_bits(sens, 3.0, fixed={"big_robust": 8})["big_robust"] == 8
    # an unreachable target stops at the lowest bit-width instead of looping
    assert set(allocate_bits(sens, 0.5).values()) == {2}


def test_layer_sensitivity_and_allocation_on_mobilenet():
    from compressor.sensitivity import allocate_bits, layer_sensitivity
    m = random_model()
    table = layer_sensitivity(m, [normalise(images(4))], bit_choices=(2, 4, 8))
    assert len(table["kl"]) == 53 and sum(table["sizes"].values()) == 2_202_560
    for name, d in table["kl"].items():
        assert all(v >= 0 for v in d.values())
        assert d[8] <= d[2] + 1e-6           # 8 bits disturb the output less than 2 bits
    bits = allocate_bits(table, 3.0, fixed={"stem": 8, "classifier": 8})
    avg = sum(table["sizes"][n] * b for n, b in bits.items()) / 2_202_560
    assert avg <= 3.0 and bits["stem"] == 8 and bits["classifier"] == 8
    # the weights of the float model are restored after the measurement
    assert torch.equal(random_model().stem.conv.weight, m.stem.conv.weight)
    cfg = CompressionConfig(w_bits=8, a_bits=8, layer_w_bits=bits)
    q = QMobileNetV2.from_float(m, cfg)
    assert abs(q.average_weight_bits() - avg) < 1e-6
