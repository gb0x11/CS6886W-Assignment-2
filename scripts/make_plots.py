"""Build the figures and LaTeX tables of the report from the result files.

    python scripts/make_plots.py [--best <run name>]

Reads   results/baseline.json, results/runs/*.json, results/sensitivity.json
Writes  report/figures/*.png, report/tables/*.tex, report/numbers.tex, results/summary.csv

`--best` selects the run reported as the final compressed model; by default
the selection rule of the report is applied: the highest model compression
ratio among the runs whose test accuracy is within `--max_drop` points of the
full-precision baseline.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

FIG, TAB = "report/figures", "report/tables"
MB = 1024 * 1024
# Categorical palette, fixed order (never cycled).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                              "#e87ba4", "#008300")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef4fb", "#2a78d6", "#0b2f5c"])

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "lines.linewidth": 1.6,
})

FAMILY = {"ptq": ("Post-training quantization", ORANGE, "s"),
          "qat": ("QAT, uniform bit-width", BLUE, "o"),
          "protect": ("QAT, sensitive tensors at 8 bits", AQUA, "D"),
          "mixed": ("QAT, mixed precision (allocator)", MAGENTA, "^"),
          "acts": ("QAT, mixed precision, linear tensors at 6 bits", YELLOW, "v"),
          "final": ("Final model (longer fine-tuning)", GREEN, "*")}


def tex(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def load_runs():
    runs = []
    for path in sorted(glob.glob("results/runs/*.json")):
        with open(path) as f:
            r = json.load(f)
        c, fin = r["config"], r["final"]
        family = c.get("group") or ("ptq" if c["method"] == "ptq" else "qat")
        r["row"] = dict(
            name=r["name"], family=family, method=c["method"],
            w_bits=fin["avg_weight_bits"], w_nominal=c["weight_quant_bits"],
            mixed=c.get("mixed_avg_bits", 0) > 0,
            a_bits=c["activation_quant_bits"],
            a_bits_linear=c.get("a_bits_linear") or c["activation_quant_bits"],
            acc=fin["quantized_acc"], drop=fin["acc_drop"], ptq_acc=fin["ptq_acc"],
            size_mb=fin["model_size_mb"], ratio=fin["compression_ratio"],
            w_ratio=fin["weight_compression_ratio"], a_ratio=fin["act_compression_ratio"],
            a_peak_ratio=fin["act_peak_compression_ratio"],
            overhead=fin["overhead_fraction"], epochs=len(r["history"]))
        runs.append(r)
    return runs


# ---------------------------------------------------------------- baseline (Q1)
def baseline_curves(base, path):
    h = pd.DataFrame(base["history"])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, (col, title) in zip(axes, [("loss", "Loss"), ("acc", "Top-1 accuracy (%)")]):
        ax.plot(h["epoch"], h[f"train_{col}"], color=BLUE, label="train (augmented batches)")
        ax.plot(h["epoch"], h[f"test_{col}"], color=ORANGE, label="test")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch")
        ax.legend(loc="best")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_ylim(max(0, h["test_acc"].iloc[5:].min() - 10), 100)
    axes[2].plot(h["epoch"], h["lr"], color=INK2)
    axes[2].set_title("Learning rate", loc="left")
    axes[2].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- trade-off (Q3)
def tradeoff(df, baseline_acc, best, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, zoom in zip(axes, (False, True)):
        for fam, (label, color, marker) in FAMILY.items():
            d = df[df["family"] == fam]
            if len(d):
                ax.scatter(d["ratio"], d["acc"], s=110 if marker == "*" else 42, color=color,
                           marker=marker, label=label,
                           edgecolor="white", linewidth=0.8, zorder=3)
        ax.axhline(baseline_acc, color=INK2, lw=1, ls="--", zorder=1)
        ax.set_xlabel("Model compression ratio (in memory, all overheads included)")
        if zoom:
            ax.set_ylim(baseline_acc - 6, baseline_acc + 0.8)
            ax.set_title("Zoom: within 6 points of the baseline", loc="left")
            for _, r in df[df["acc"] > baseline_acc - 6].iterrows():
                ax.annotate(short_label(r), (r["ratio"], r["acc"]), fontsize=6.5, color=INK,
                            xytext=(4, -9), textcoords="offset points")
        else:
            ax.set_ylabel("Test top-1 accuracy (%)")
            ax.set_title("All compression runs", loc="left")
            ax.text(df["ratio"].min(), baseline_acc + 0.6, f"FP32 baseline {baseline_acc:.2f} %",
                    fontsize=7.5, color=INK2, va="bottom")
        if best is not None:
            b = df[df["name"] == best].iloc[0]
            ax.scatter([b["ratio"]], [b["acc"]], s=190, facecolor="none", edgecolor=INK,
                       linewidth=1.3, zorder=4)
    # Legend under the panels, so that it can never cover a point.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def short_label(r) -> str:
    wtxt = f"W{r['w_bits']:.2f}" if r["mixed"] else f"W{int(r['w_nominal'])}"
    return f"{wtxt}/A{int(r['a_bits'])}"


# ------------------------------------------------- per-layer bit allocation (Q2b)
KIND_COLOR = {"expand": BLUE, "depthwise": ORANGE, "project": AQUA}


def bit_allocation(run, path):
    layers = run["layers"]
    x = np.arange(len(layers))
    colors = [KIND_COLOR.get(l["kind"], YELLOW) for l in layers]
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(10.5, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1]})
    a0.bar(x, [l["bits"] for l in layers], color=colors, width=0.72)
    a0.set_ylabel("Weight bits")
    a0.set_yticks(range(0, 9, 2))
    a0.set_title("Bit-width of every weight tensor of the final compressed model", loc="left")
    a1.bar(x, [l["weights"] for l in layers], color=colors, width=0.72)
    a1.set_yscale("log")
    a1.set_ylabel("Number of weights")
    a1.set_xlabel("Layer (execution order): stem, 17 inverted-residual blocks, head, classifier")
    a1.set_xticks([0, len(layers) - 2, len(layers) - 1], ["stem", "head", "fc"], fontsize=7)
    a1.set_xlim(-0.8, len(layers) - 0.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA, YELLOW)]
    a0.legend(handles, ["1x1 expand", "3x3 depthwise", "1x1 project", "stem / head / classifier"],
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.32), fontsize=7.5)
    for a in (a0, a1):
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def sensitivity_heatmap(table, path):
    names = list(table["kl"])
    bits = sorted(int(b) for b in table["kl"][names[0]])
    M = np.array([[max(table["kl"][n][str(b)], 1e-7) for n in names] for b in bits])
    fig, ax = plt.subplots(figsize=(10.5, 2.3))
    im = ax.imshow(M, aspect="auto", cmap=SEQ, norm=LogNorm(vmin=max(M.min(), 1e-6), vmax=M.max()))
    ax.set_yticks(range(len(bits)), [f"{b} bits" for b in bits])
    ax.set_xticks([0, len(names) - 2, len(names) - 1], ["stem", "head", "fc"], fontsize=7)
    ax.set_xlabel("Layer whose weights are quantized (execution order)")
    ax.grid(False)
    ax.set_title("Sensitivity: KL divergence of the output when one layer is quantized", loc="left")
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03, label="KL (log scale)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------- activation memory (Q4b)
def activation_memory(run, path):
    acts = run["activations"]
    x = np.arange(len(acts))
    fp32 = np.array([a["fp32_bits"] for a in acts]) / 8 / 1024
    comp = np.array([a["size_bits"] for a in acts]) / 8 / 1024
    fig, ax = plt.subplots(figsize=(10.5, 3.1))
    ax.bar(x, fp32, color="#c9d9ee", width=0.8, label="FP32 (uncompressed)")
    ax.bar(x, comp, color=BLUE, width=0.8, label="compressed")
    ax.set_ylabel("Size of the tensor for one image (KB)")
    ax.set_xlabel("Activation tensor (execution order): input, stem, blocks, head, pooled, logits")
    ax.set_xticks([])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("Every activation tensor of one forward pass", loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- tables
def write(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def sweep_table(df, best, path):
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"Run & Method & \shortstack{W bits\\(avg)} & \shortstack{A bits\\(ReLU6 / lin.)} & "
             r"\shortstack{Test\\acc. (\%)} & \shortstack{Drop\\(pts)} & \shortstack{Size\\(MB)} & "
             r"\shortstack{Model\\ratio} & \shortstack{Weight\\ratio} & \shortstack{Act.\\ratio} \\",
             r"\midrule"]
    order = {f: i for i, f in enumerate(FAMILY)}
    d = df.assign(o=df["family"].map(order).fillna(9)).sort_values(["o", "ratio"])
    last = None
    for _, r in d.iterrows():
        if last is not None and r["family"] != last:
            lines.append(r"\midrule")
        last = r["family"]
        name = tex(r["name"])
        if r["name"] == best:
            name = rf"\textbf{{{name}}}"
        method = "PTQ" if r["method"] == "ptq" else f"QAT, {int(r['epochs'])} ep."
        lines.append(f"{name} & {method} & {r['w_bits']:.2f} & {int(r['a_bits'])} / "
                     f"{int(r['a_bits_linear'])} & {r['acc']:.2f} & {r['drop']:+.2f} & "
                     f"{r['size_mb']:.3f} & {r['ratio']:.2f} & {r['w_ratio']:.2f} & "
                     f"{r['a_ratio']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


def layer_group_table(run, path):
    """Which layers are compressed, and how (grouped by layer type)."""
    groups = [("stem", "Stem 3x3 conv"), ("expand", "1x1 expand convs"),
              ("depthwise", "3x3 depthwise convs"), ("project", "1x1 project convs (linear)"),
              ("head", "Head 1x1 conv"), ("classifier", "Classifier (linear)")]
    lines = [r"\begin{tabular}{lrrlrrl}", r"\toprule",
             r"Layer type & Layers & Weights & Weight bits & Code bytes & Overhead bytes & "
             r"Output activation \\", r"\midrule"]
    for kind, label in groups:
        ls = [l for l in run["layers"] if l["kind"] == kind]
        bits = sorted({l["bits"] for l in ls})
        btxt = str(bits[0]) if len(bits) == 1 else f"{bits[0]} to {bits[-1]}"
        abits = sorted({l["act_bits"] for l in ls})
        if kind == "classifier":
            atxt = "logits, FP32"
        else:
            sign = "signed" if kind == "project" else "unsigned"
            atxt = f"{abits[0]}-bit {sign}"
        lines.append(f"{label} & {len(ls)} & {sum(l['weights'] for l in ls):,} & {btxt} & "
                     f"{sum(l['code_bytes'] for l in ls):,} & "
                     f"{sum(l['overhead_bytes'] for l in ls):,} & {atxt} \\\\")
    tot = run["layers"]
    lines += [r"\midrule",
              f"Total & {len(tot)} & {sum(l['weights'] for l in tot):,} & "
              f"{run['final']['avg_weight_bits']:.2f} (avg) & "
              f"{sum(l['code_bytes'] for l in tot):,} & {sum(l['overhead_bytes'] for l in tot):,} & \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


def overhead_table(run, base_bytes, path):
    f = run["final"]
    n_ch = f["scale_bytes"] // 2 if run["config"]["overhead_dtype"] == "fp16" else f["scale_bytes"] // 4
    obits = 16 if run["config"]["overhead_dtype"] == "fp16" else 32
    rows = [("Weight codes (bit-packed)", f"{sum(l['weights'] for l in run['layers']):,} weights",
             f"{f['avg_weight_bits']:.2f} bits (avg)", f["weight_bytes"]),
            ("Per-channel scales", f"{n_ch:,} channels", f"{obits} bits (FP{obits})", f["scale_bytes"]),
            ("Per-channel biases", f"{n_ch:,} channels", f"{obits} bits (FP{obits})", f["bias_bytes"]),
            ("Activation quantizer steps", f"{f['act_param_bytes'] // 8} quantizers",
             "8 bytes each", f["act_param_bytes"]),
            ("Layer metadata, input mean / std", f"{len(run['layers'])} layers",
             "16 bytes each + 24", f["meta_bytes"])]
    lines = [r"\begin{tabular}{lllrr}", r"\toprule",
             r"Component & Count & Precision & Bytes & Share (\%) \\", r"\midrule"]
    for label, count, prec, b in rows:
        lines.append(f"{label} & {count} & {prec} & {b:,} & {100 * b / f['model_bytes']:.2f} \\\\")
    lines += [r"\midrule",
              f"Compressed model & & & {f['model_bytes']:,} & 100.00 \\\\",
              f"of which overhead & & & {f['overhead_bytes']:,} & "
              f"{100 * f['overhead_fraction']:.2f} \\\\",
              f"FP32 model (parameters + BatchNorm buffers) & & 32 bits & {base_bytes:,} & \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


def activation_table(run, path):
    acts = run["activations"]

    def group(a):
        n = a["name"]
        if n in ("input", "pool", "logits"):
            return {"input": "Input image (uint8 pixels)", "pool": "Pooled feature vector",
                    "logits": "Logits"}[n]
        if n.endswith((".project", ".add")):
            return "Linear bottleneck outputs and residual sums"
        return "ReLU6 feature maps (stem, expand, depthwise, head)"
    order = ["Input image (uint8 pixels)", "ReLU6 feature maps (stem, expand, depthwise, head)",
             "Linear bottleneck outputs and residual sums", "Pooled feature vector", "Logits"]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Activation tensors & Tensors & Elements & Bits & FP32 bytes & Compressed bytes \\",
             r"\midrule"]
    for g in order:
        a = [x for x in acts if group(x) == g]
        bits = sorted({x["bits"] for x in a})
        lines.append(f"{g} & {len(a)} & {sum(x['elements'] for x in a):,} & "
                     f"{', '.join(map(str, bits))} & {sum(x['fp32_bits'] for x in a) // 8:,} & "
                     f"{sum(x['size_bits'] for x in a) / 8:,.0f} \\\\")
    lines += [r"\midrule",
              f"Total (one image) & {len(acts)} & {sum(x['elements'] for x in acts):,} & "
              f"{run['final']['act_avg_bits']:.2f} (avg) & "
              f"{sum(x['fp32_bits'] for x in acts) // 8:,} & "
              f"{sum(x['size_bits'] for x in acts) / 8:,.0f} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


def numbers(base, run, df, path):
    f, b = run["final"], base["final"]
    macros = {
        "BaselineAcc": f"{b['test_acc']:.2f}", "BaselineTrainAcc": f"{b['train_acc_clean']:.2f}",
        "BaselineEpochs": str(b["epochs"]), "BaselineBestEpochAcc": f"{b['best_epoch_test_acc']:.2f}",
        "BaselineHours": f"{b['train_time_s'] / 3600:.1f}",
        "FPModelMB": f"{f['fp32_model_size_mb']:.3f}", "FPModelBytes": f"{f['fp32_model_bytes']:,}",
        "BestName": tex(run["name"]), "BestAcc": f"{f['quantized_acc']:.2f}",
        "BestDrop": f"{f['acc_drop']:.2f}", "BestSizeMB": f"{f['model_size_mb']:.3f}",
        "BestSizeBytes": f"{f['model_bytes']:,}", "BestRatio": f"{f['compression_ratio']:.2f}",
        "BestWeightRatio": f"{f['weight_compression_ratio']:.2f}",
        "BestActRatio": f"{f['act_compression_ratio']:.2f}",
        "BestActPeakRatio": f"{f['act_peak_compression_ratio']:.2f}",
        "BestAvgWBits": f"{f['avg_weight_bits']:.2f}", "BestAvgABits": f"{f['act_avg_bits']:.2f}",
        "BestOverheadPct": f"{100 * f['overhead_fraction']:.1f}",
        "BestOverheadBytes": f"{f['overhead_bytes']:,}",
        "BestActMB": f"{f['act_total_mb']:.3f}", "FPActMB": f"{f['act_total_mb_fp32']:.3f}",
        "BestActPeakMB": f"{f['act_peak_mb']:.3f}", "FPActPeakMB": f"{f['act_peak_mb_fp32']:.3f}",
        "BestActMeasuredBytes": f"{f['act_measured_bytes']:,.0f}",
        "BestFakeQuantAcc": f"{f['fakequant_acc']:.2f}", "BestPtqAcc": f"{f['ptq_acc']:.2f}",
        "BestEpochs": str(len(run["history"])), "NumRuns": str(len(df)),
        "MaxRatio": f"{df['ratio'].max():.2f}", "MaxWeightRatio": f"{df['w_ratio'].max():.2f}",
        "MaxActRatio": f"{df['a_ratio'].max():.2f}",
    }
    acts, layers, c = run["activations"], run["layers"], run["config"]
    linear = [a for a in acts if a["name"].endswith((".project", ".add")) or a["name"] == "pool"]
    small = [l for l in layers if l["kind"] in ("stem", "classifier")]
    bits = [l["bits"] for l in layers]
    macros.update({
        "BestABits": str(c["activation_quant_bits"]),
        "BestALinBits": str(c.get("a_bits_linear") or c["activation_quant_bits"]),
        "LinActPct": f"{100 * sum(a['fp32_bits'] for a in linear) / sum(a['fp32_bits'] for a in acts):.0f}",
        "SmallWeightPct": f"{100 * sum(l['weights'] for l in small) / sum(l['weights'] for l in layers):.1f}",
        "BestWeightBytes": f"{f['weight_bytes']:,}", "FPWeightBytes": f"{f['fp32_weight_bytes']:,}",
        "BestMinWBits": str(min(bits)), "BestMaxWBits": str(max(bits)),
        "BestTwoBitLayers": str(sum(b == 2 for b in bits)),
        "BestActFPHookBytes": f"{f['act_fp32_hook_bytes']:,}",
        "BestTrainMin": f"{f['train_time_s'] / 60:.0f}",
        "BestCkptMB": f"{f['checkpoint_file_mb']:.3f}",
        "BestLr": f"{c['lr']:g}",
    })
    write(path, [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--best", default="", help="name of the run reported as the final model")
    p.add_argument("--max_drop", type=float, default=1.0,
                   help="accuracy budget (points below the baseline) of the selection rule")
    args = p.parse_args()
    os.makedirs(FIG, exist_ok=True)

    with open("results/baseline.json") as f:
        base = json.load(f)
    baseline_curves(base, f"{FIG}/baseline_curves.png")

    runs = load_runs()
    if not runs:
        print("no compression runs yet")
        return
    df = pd.DataFrame([r["row"] for r in runs])
    df.to_csv("results/summary.csv", index=False, float_format="%.5g")
    baseline_acc = runs[0]["final"]["baseline_acc"]

    best = args.best
    if not best:
        ok = df[df["drop"] <= args.max_drop]
        best = (ok if len(ok) else df).sort_values("ratio").iloc[-1]["name"]
    run = next(r for r in runs if r["name"] == best)
    print(df.sort_values("ratio")[["name", "w_bits", "a_bits", "acc", "drop", "size_mb", "ratio",
                                   "w_ratio", "a_ratio"]].to_string(index=False))
    print("final model:", best)

    tradeoff(df, baseline_acc, best, f"{FIG}/tradeoff.png")
    bit_allocation(run, f"{FIG}/bit_allocation.png")
    activation_memory(run, f"{FIG}/activation_memory.png")
    if os.path.exists("results/sensitivity.json"):
        with open("results/sensitivity.json") as f:
            sensitivity_heatmap(json.load(f), f"{FIG}/sensitivity.png")
    sweep_table(df, best, f"{TAB}/sweep.tex")
    layer_group_table(run, f"{TAB}/layers.tex")
    overhead_table(run, run["final"]["fp32_model_bytes"], f"{TAB}/overheads.tex")
    activation_table(run, f"{TAB}/activations.tex")
    numbers(base, run, df, "report/numbers.tex")


if __name__ == "__main__":
    sys.exit(main())
