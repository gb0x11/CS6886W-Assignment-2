"""Run the compression sweep of Question 3 (one `compress.py` process per configuration).

    python run_sweep.py --dry_run          # list the configurations
    python run_sweep.py --group ptq        # post-training quantization points (minutes)
    python run_sweep.py --group qat        # quantization-aware fine-tuning points
    python run_sweep.py --group mixed      # sensitivity-guided mixed precision
    python run_sweep.py                    # everything

A configuration is skipped when results/runs/<name>.json already exists, so
the sweep can be interrupted and resumed. Every run logs to W&B; the
parallel-coordinates chart is built from those runs.
"""
import argparse
import os
import subprocess
import sys

# Bit-widths that "protect" the few tensors that are tiny in memory but that
# the network is most sensitive to: the stem and the classifier weights and
# the narrow linear-bottleneck activations stay at 8 bits.
PROTECT = dict(w_bits_first=8, w_bits_last=8, a_bits_linear=8)


def w(wb, ab, **kw):
    return dict(weight_quant_bits=wb, activation_quant_bits=ab, **kw)


SWEEP = {
    # Post-training quantization: no training, calibration + BatchNorm re-estimation only.
    "ptq": [dict(method="ptq", **w(8, 8)), dict(method="ptq", **w(6, 6)),
            dict(method="ptq", **w(4, 8)), dict(method="ptq", **w(8, 4)),
            dict(method="ptq", **w(4, 4)), dict(method="ptq", **w(3, 3))],
    # Fine-tuning with the sensitive tensors protected.
    "protect": [dict(method="qat", **w(4, 4), **PROTECT), dict(method="qat", **w(3, 4), **PROTECT),
                dict(method="qat", **w(2, 4), **PROTECT), dict(method="qat", **w(3, 3), **PROTECT)],
    # Sensitivity-guided mixed precision at several average bit-widths.
    "mixed": [dict(method="qat", mixed_avg_bits=3.0, activation_quant_bits=4, **PROTECT),
              dict(method="qat", mixed_avg_bits=2.5, activation_quant_bits=4, **PROTECT),
              dict(method="qat", mixed_avg_bits=2.2, activation_quant_bits=4, **PROTECT),
              dict(method="qat", mixed_avg_bits=2.5, activation_quant_bits=3, **PROTECT)],
    # Quantization-aware fine-tuning, the same bit-width everywhere (no protection).
    "qat": [dict(method="qat", **w(4, 4)), dict(method="qat", **w(3, 3)),
            dict(method="qat", **w(2, 2))],
}


def run_name(cfg) -> str:
    """Must match compress.default_name."""
    wpart = (f"mix{cfg['mixed_avg_bits']:g}" if cfg.get("mixed_avg_bits", 0) > 0
             else f"w{cfg['weight_quant_bits']}")
    name = f"{cfg['method']}_{wpart}_a{cfg['activation_quant_bits']}"
    for key, tag in (("w_bits_first", "f"), ("w_bits_dw", "dw"), ("w_bits_last", "l"),
                     ("a_bits_linear", "al")):
        if cfg.get(key):
            name += f"_{tag}{cfg[key]}"
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", nargs="+", default=["all"], choices=list(SWEEP) + ["all"])
    p.add_argument("--epochs", type=int, default=None, help="override the QAT epochs")
    p.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    groups = list(SWEEP) if "all" in args.group else args.group
    for g in groups:
        for cfg in SWEEP[g]:
            name = run_name(cfg)
            done = os.path.exists(os.path.join("results", "runs", f"{name}.json"))
            print(f"[{g}] {name}{'  (done)' if done else ''}", flush=True)
            if done or args.dry_run:
                continue
            cmd = [sys.executable, "compress.py", f"--group={g}", f"--wandb_mode={args.wandb_mode}"]
            cmd += [f"--{k}={v}" for k, v in cfg.items()]
            if args.epochs is not None and cfg["method"] == "qat":
                cmd.append(f"--epochs={args.epochs}")
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
