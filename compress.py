"""Compress the trained MobileNet-v2 at one configuration and log the result (Questions 2 to 4).

    # post-training quantization, 8-bit weights and activations
    python compress.py --method ptq --weight_quant_bits 8 --activation_quant_bits 8

    # quantization-aware fine-tuning, 4-bit weights and activations
    python compress.py --method qat --weight_quant_bits 4 --activation_quant_bits 4

    # mixed precision: sensitivity-guided allocation to an average of 2.5 bits per weight
    python compress.py --method qat --mixed_avg_bits 2.5 --activation_quant_bits 4 \
        --a_bits_linear 8 --w_bits_first 8 --w_bits_last 8

Pipeline
    1. load the full-precision baseline (the teacher) and choose the bit-widths
       (uniform, per layer type, or by the sensitivity-guided allocator);
    2. attach quantizers and calibrate their steps on training images;
    3. PTQ: re-estimate BatchNorm statistics.  QAT: fine-tune weights and steps
       with the straight-through estimator and knowledge distillation;
    4. fold BatchNorm, bit-pack the weight codes, export the packed model;
    5. evaluate the *packed* model on the test set and measure its in-memory
       size, its overheads and its activation memory.

Outputs
    checkpoints/compressed/<name>.pt   the packed model (what is held in memory)
    results/runs/<name>.json           configuration, metrics, per-layer table
    a W&B run with the same numbers (for the parallel-coordinates chart)
"""
import argparse
import os
import time

import torch
import yaml

from compressor.config import CompressionConfig
from compressor.packed import export_packed
from compressor.pipeline import (calibration_batches, evaluate_packed, load_baseline, measure,
                                 post_training_quantize, save_packed)
from compressor.qmodel import QMobileNetV2
from compressor.sensitivity import allocate_bits, layer_sensitivity
from mbv2.data import GpuAugment, load_cifar10
from mbv2.engine import build_sgd, evaluate, train_one_epoch, warmup_cosine
from mbv2.utils import get_device, save_json, set_seed, str2bool

DEFAULTS = dict(
    name="",                       # default: derived from the configuration
    baseline="checkpoints/baseline.pt",
    method="qat",                  # "ptq" (no training) or "qat" (fine-tuning)
    # ---- bit-widths (32 = keep full precision)
    weight_quant_bits=8,           # every conv / linear weight unless overridden below
    activation_quant_bits=8,       # activations after ReLU6
    w_bits_first=0, w_bits_dw=0, w_bits_last=0,   # 0 = same as weight_quant_bits
    a_bits_linear=0,               # linear / residual / pooled activations; 0 = same as activations
    mixed_avg_bits=0.0,            # > 0: sensitivity-guided mixed precision with this average
    bit_choices="2,3,4,5,6,8",     # candidate bit-widths of the allocator
    overhead_dtype="fp16",         # precision of per-channel scales and biases (fp16 / fp32)
    # ---- calibration
    calib_images=2048, bn_recalibration=True, sensitivity_images=2000,
    # ---- quantization-aware fine-tuning
    epochs=15, batch_size=128, lr=0.01, momentum=0.9, weight_decay=1e-4, warmup_epochs=1,
    kd_alpha=0.7, kd_temperature=4.0, label_smoothing=0.0, cutout=16, dropout=0.0,
    # ---- misc
    data_dir="./data", seed=42, device="auto", out_dir=".",
    wandb_project="cs6886w-a2-mobilenetv2-compression", wandb_mode="", group="",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML file with any of the options below")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=str2bool if isinstance(v, bool) else type(v))
    args = p.parse_args(argv)
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})
    if not cfg["name"]:
        cfg["name"] = default_name(cfg)
    return cfg


def default_name(cfg) -> str:
    w = (f"mix{cfg['mixed_avg_bits']:g}" if cfg["mixed_avg_bits"] > 0
         else f"w{cfg['weight_quant_bits']}")
    name = f"{cfg['method']}_{w}_a{cfg['activation_quant_bits']}"
    for key, tag in (("w_bits_first", "f"), ("w_bits_dw", "dw"), ("w_bits_last", "l"),
                     ("a_bits_linear", "al")):
        if cfg[key]:
            name += f"_{tag}{cfg[key]}"
    return name


def sensitivity_table(float_model, train_set, eval_tf, cfg):
    """Per-layer sensitivities, cached on disk (they depend only on the baseline)."""
    import json
    choices = [int(b) for b in str(cfg["bit_choices"]).split(",")]
    path = os.path.join(cfg["out_dir"], "results", "sensitivity.json")
    if os.path.exists(path):
        with open(path) as f:
            table = json.load(f)
        if sorted(int(b) for b in next(iter(table["kl"].values()))) == sorted(choices):
            return table
    print("[sensitivity] measuring per-layer sensitivity ...", flush=True)
    batches = calibration_batches(train_set, eval_tf, cfg["sensitivity_images"], 250, seed=1)
    table = layer_sensitivity(float_model, batches, choices)
    save_json(table, path)
    return table


def build_config(cfg, float_model, train_set, eval_tf) -> CompressionConfig:
    cc = CompressionConfig(
        w_bits=cfg["weight_quant_bits"], a_bits=cfg["activation_quant_bits"],
        w_bits_first=cfg["w_bits_first"] or None, w_bits_dw=cfg["w_bits_dw"] or None,
        w_bits_last=cfg["w_bits_last"] or None, a_bits_linear=cfg["a_bits_linear"] or None)
    if cfg["mixed_avg_bits"] > 0:
        table = sensitivity_table(float_model, train_set, eval_tf, cfg)
        fixed = {}
        if cfg["w_bits_first"]:
            fixed["stem"] = cfg["w_bits_first"]
        if cfg["w_bits_last"]:
            fixed["classifier"] = cfg["w_bits_last"]
        cc.layer_w_bits = allocate_bits(table, cfg["mixed_avg_bits"], fixed)
    return cc


def init_wandb(cfg):
    mode = cfg["wandb_mode"] or os.environ.get("WANDB_MODE", "online")
    if mode == "disabled":
        return None
    import wandb
    run = wandb.init(project=cfg["wandb_project"], name=cfg["name"], job_type=cfg["method"],
                     group=cfg["group"] or None, config=cfg, mode=mode, dir=cfg["out_dir"])
    wandb.define_metric("epoch")
    for k in ("train_loss", "train_acc", "test_loss", "test_acc", "lr"):
        wandb.define_metric(k, step_metric="epoch")
    return run


def compress(cfg):
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    name = cfg["name"]
    train_set, _, test_set = load_cifar10(cfg["data_dir"], device)
    eval_tf = GpuAugment(device, augment=False)

    teacher = load_baseline(cfg["baseline"], device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    _, baseline_acc = evaluate(teacher, test_set, eval_tf)

    cc = build_config(cfg, teacher, train_set, eval_tf)
    run = init_wandb(cfg)
    print(f"[{name}] device={device} baseline test acc {baseline_acc:.2f}%  method={cfg['method']}")

    # ---- quantizers + calibration (this is already post-training quantization)
    calib = calibration_batches(train_set, eval_tf, cfg["calib_images"], seed=cfg["seed"])
    qmodel = post_training_quantize(teacher, cc, calib, cfg["bn_recalibration"])
    _, ptq_acc = evaluate(qmodel, test_set, eval_tf, batch_size=500)
    print(f"[{name}] average weight bits {qmodel.average_weight_bits():.3f} | "
          f"after calibration (PTQ) test acc {ptq_acc:.2f}%", flush=True)

    # ---- quantization-aware fine-tuning
    history, t0 = [], time.time()
    if cfg["method"] == "qat" and cfg["epochs"] > 0:
        qmodel.dropout.p = cfg["dropout"]
        optimizer = build_sgd(qmodel, cfg["lr"], cfg["momentum"], cfg["weight_decay"])
        steps = len(train_set) // cfg["batch_size"]
        scheduler = warmup_cosine(optimizer, cfg["epochs"] * steps, cfg["warmup_epochs"] * steps)
        train_tf = GpuAugment(device, augment=True, cutout=cfg["cutout"], seed=cfg["seed"])
        gen = torch.Generator().manual_seed(cfg["seed"])
        for epoch in range(1, cfg["epochs"] + 1):
            lr, te = optimizer.param_groups[0]["lr"], time.time()
            tr_loss, tr_acc = train_one_epoch(
                qmodel, train_set, cfg["batch_size"], optimizer, scheduler, train_tf, gen,
                cfg["label_smoothing"], teacher, cfg["kd_alpha"], cfg["kd_temperature"])
            te_loss, te_acc = evaluate(qmodel, test_set, eval_tf, batch_size=500)
            row = dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, test_loss=te_loss,
                       test_acc=te_acc, lr=lr, epoch_time=time.time() - te)
            history.append(row)
            if run is not None:
                run.log(row)
            print(f"[{name}] ep {epoch:2d}/{cfg['epochs']} lr {lr:.2e} | train {tr_loss:.4f} "
                  f"{tr_acc:6.2f}% | test {te_loss:.4f} {te_acc:6.2f}% | {row['epoch_time']:.0f}s",
                  flush=True)
            if tr_loss != tr_loss:
                raise RuntimeError("fine-tuning diverged")
    qmodel.eval()
    _, fakequant_acc = evaluate(qmodel, test_set, eval_tf, batch_size=500)

    # ---- export: fold BatchNorm, bit-pack, measure; the packed model is what is reported
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[cfg["overhead_dtype"]]
    packed = export_packed(qmodel, dtype)
    test_loss, test_acc = evaluate_packed(packed, test_set)
    sizes = measure(packed, teacher, test_set)
    records = sizes.pop("act_records")
    report = packed.size_report()
    file_bytes = save_packed(packed, os.path.join(cfg["out_dir"], "checkpoints", "compressed",
                                                  f"{name}.pt"),
                             {"name": name, "test_acc": test_acc, "baseline_acc": baseline_acc})
    final = {
        "method": cfg["method"],
        "weight_quant_bits": cfg["weight_quant_bits"] if cfg["mixed_avg_bits"] <= 0
        else sizes["avg_weight_bits"],
        "activation_quant_bits": cfg["activation_quant_bits"],
        "quantized_acc": test_acc, "quantized_loss": test_loss,
        "baseline_acc": baseline_acc, "acc_drop": baseline_acc - test_acc,
        "ptq_acc": ptq_acc, "fakequant_acc": fakequant_acc,
        "checkpoint_file_mb": file_bytes / 1024 / 1024,
        "train_time_s": time.time() - t0, **sizes,
    }
    print(f"[{name}] packed model: test acc {test_acc:.2f}% (drop {final['acc_drop']:+.2f}) | "
          f"size {sizes['model_size_mb']:.3f} MB | model x{sizes['compression_ratio']:.2f} "
          f"weights x{sizes['weight_compression_ratio']:.2f} "
          f"activations x{sizes['act_compression_ratio']:.2f}", flush=True)
    save_json({"name": name, "config": cfg, "compression": cc.to_dict(), "final": final,
               "history": history, "layers": report["layers"], "activations": records},
              os.path.join(cfg["out_dir"], "results", "runs", f"{name}.json"))
    if run is not None:
        run.summary.update(final)
        run.finish()
    return final


if __name__ == "__main__":
    compress(parse_args())
