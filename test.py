"""Evaluate the baseline and a compressed MobileNet-v2 (same interface as the course template).

    # template command: post-training quantization at the given bit-widths
    python test.py --weight_quant_bits 8 --activation_quant_bits 8

    # evaluate a compressed model produced by compress.py (for example the final one)
    python test.py --checkpoint checkpoints/compressed/best.pt

It prints the test accuracy before and after compression and the in-memory
compression summary (model, weights, activations, overheads).
"""
import argparse

import torch

from compressor.config import CompressionConfig
from compressor.packed import export_packed
from compressor.pipeline import (bn_batches, calibration_batches, evaluate_packed, load_baseline,
                                 load_packed, measure, post_training_quantize)
from mbv2.data import GpuAugment, load_cifar10
from mbv2.engine import evaluate
from mbv2.utils import get_device, set_seed, str2bool


def print_summary(s: dict, report: dict) -> None:
    mb = 1024 * 1024
    print("=== Compression summary (in-memory sizes) ===")
    print(f"FP32 model size:        {s['fp32_model_size_mb']:.3f} MB "
          f"({s['fp32_model_bytes']:,} bytes: parameters + BatchNorm buffers)")
    print(f"Compressed model size:  {s['model_size_mb']:.3f} MB ({s['model_bytes']:,} bytes)")
    print(f"  weight codes          {s['weight_bytes']:>10,} bytes "
          f"(average {s['avg_weight_bits']:.2f} bits per weight)")
    print(f"  per-channel scales    {s['scale_bytes']:>10,} bytes")
    print(f"  biases                {s['bias_bytes']:>10,} bytes")
    print(f"  activation steps      {s['act_param_bytes']:>10,} bytes")
    print(f"  layer metadata        {s['meta_bytes']:>10,} bytes")
    print(f"  overhead share        {100 * s['overhead_fraction']:.1f} % of the compressed model")
    print(f"Compression ratio, model:       {s['compression_ratio']:.2f}x")
    print(f"Compression ratio, weights:     {s['weight_compression_ratio']:.2f}x")
    print(f"Compression ratio, activations: {s['act_compression_ratio']:.2f}x "
          f"(all feature maps of one image: {s['act_total_mb_fp32']:.3f} MB -> "
          f"{s['act_total_mb']:.3f} MB; measured from packed buffers "
          f"{s['act_measured_bytes'] / mb:.3f} MB)")
    print(f"  peak activation memory        {s['act_peak_compression_ratio']:.2f}x "
          f"({s['act_peak_mb_fp32']:.3f} MB -> {s['act_peak_mb']:.3f} MB)")
    bits = sorted({r["bits"] for r in report["layers"]})
    print(f"Weight bit-widths used: {bits}")
    print("=============================================")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weight_quant_bits", type=int, default=8, help="bits of the weights")
    p.add_argument("--activation_quant_bits", type=int, default=8, help="bits of the activations")
    p.add_argument("--checkpoint", default="",
                   help="packed model from compress.py; if given, the bit-width flags are ignored")
    p.add_argument("--baseline", default="checkpoints/baseline.pt")
    p.add_argument("--bn_recalibration", type=str2bool, default=True)
    p.add_argument("--calib_images", type=int, default=2048)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    train_set, _, test_set = load_cifar10(args.data_dir, device)
    eval_tf = GpuAugment(device, augment=False)

    model = load_baseline(args.baseline, device)
    _, test_acc = evaluate(model, test_set, eval_tf)
    print(f"Test Acc={test_acc:.2f}%")

    if args.checkpoint:
        packed = load_packed(args.checkpoint)
        print(f"Loaded compressed model {args.checkpoint}")
    else:
        cfg = CompressionConfig(w_bits=args.weight_quant_bits, a_bits=args.activation_quant_bits)
        calib = calibration_batches(train_set, eval_tf, args.calib_images, seed=args.seed)
        bn_calib = (bn_batches(train_set, device, args.calib_images, seed=args.seed)
                    if args.bn_recalibration else None)
        qmodel = post_training_quantize(model, cfg, calib, bn_calib)
        packed = export_packed(qmodel)
    _, q_acc = evaluate_packed(packed, test_set)
    print(f"Quantized Test Acc={q_acc:.2f}%")
    summary = measure(packed, model, test_set)
    print_summary(summary, packed.size_report())


if __name__ == "__main__":
    main()
