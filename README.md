# CS6886W Assignment 2: Compressing MobileNet-v2 on CIFAR-10

**Gowtham B (CS25M508)**, CS6886W Systems Engineering for Deep Learning, IIT Madras

MobileNet-v2 is trained on CIFAR-10 and then compressed with a quantization
method written from scratch: learned-step uniform quantizers for weights and
activations, mixed precision chosen by a sensitivity-guided allocator,
quantization-aware fine-tuning with distillation, BatchNorm folding and
bit-packing of the weight codes. No compression library or API is used
(nothing from `torch.quantization`, `torch.ao`, `torch.nn.utils.prune` or any
other package). All sizes are in-memory sizes and include every overhead.

<!-- RESULTS -->

## Repository layout

Training, evaluation and compression are separate.

| Path | Responsibility |
| --- | --- |
| `mbv2/model.py` | MobileNet-v2 (CIFAR variant: stride 1 in the stem and in stage 2) |
| `mbv2/data.py` | CIFAR-10 loading, normalization and augmentation (on the GPU) |
| `mbv2/engine.py` | training and evaluation loops, SGD and schedule builders, distillation loss |
| `train.py` | baseline training (Question 1), resumable |
| `compress.py` | one compression run: calibration, PTQ or QAT, export, measurement (Questions 2 to 4) |
| `test.py` | evaluation of the baseline and of a compressed model (the interface of the course template) |
| `run_sweep.py` | the sweep of Question 3 (one `compress.py` process per configuration) |
| `compressor/quantizers.py` | uniform quantizer with a learned step, hand-written gradients |
| `compressor/config.py` | bit-width configuration (global, per layer type, per layer) |
| `compressor/qmodel.py` | MobileNet-v2 with quantizers (training-time view), BatchNorm re-estimation |
| `compressor/sensitivity.py` | per-layer sensitivity and mixed-precision bit allocation |
| `compressor/packing.py` | bit-packing of integer codes (1 to 16 bits per value) |
| `compressor/packed.py` | the packed in-memory model: export, inference, size report |
| `compressor/size.py`, `compressor/tracer.py` | model-size and activation-memory accounting |
| `compressor/pipeline.py` | steps shared by `compress.py` and `test.py` |
| `tests/test_compression.py` | 44 unit tests |
| `scripts/` | figures and tables of the report, failure analysis, W&B report |
| `results/` | metrics of every run (JSON), `results/summary.csv` |
| `checkpoints/baseline.pt` | trained full-precision model |
| `checkpoints/compressed/best.pt` | final compressed model (packed) |
| `report/` | LaTeX source and figures of the report |

## Environment

| | |
| --- | --- |
| Machine | Apple MacBook Pro, M1 Pro, 16 GB (GPU through the PyTorch MPS backend) |
| OS | macOS 27.2 |
| Python | 3.14.3 |
| torch / torchvision | 2.14.1 / 0.29.1 |
| numpy | 2.5.4 |
| wandb | 0.31.0 |
| matplotlib / pandas | 3.11.2 / 3.1.0 |
| PyYAML / pytest | 6.0.3 / 9.2.0 |

All versions are pinned in `requirements.txt`. The code selects the device
automatically (`cuda`, then `mps`, then `cpu`); `--device` overrides it.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CIFAR-10 is downloaded to `./data` on first use (`torchvision.datasets.CIFAR10`
layout, `data/cifar-10-batches-py`).

## Seeds

Every script takes `--seed` (default 42). The seed fixes Python, NumPy and
PyTorch, the weight initialization, the shuffling of the training set, the
augmentation (crop offsets, flips, Cutout positions) and the choice of the
calibration images. The reported numbers were all produced with seed 42.
Results on another machine or backend can differ in the last digits because
GPU floating-point arithmetic is not bit-identical across devices.

## Exact commands

### 1. Train the baseline (Question 1)

```bash
python train.py                 # 200 epochs, about 5 hours on the M1 Pro
python train.py --resume        # continue an interrupted run
python scripts/analyze_baseline.py   # per-class accuracy, confusion matrix, failure examples
```

Settings (defaults of `train.py`): width multiplier 1.0, dropout 0.2, SGD with
Nesterov momentum 0.9, peak learning rate 0.1 with 5 warm-up epochs and cosine
decay, batch size 128, weight decay 5e-4 (not on BatchNorm or biases), label
smoothing 0.1, random crop (padding 4), horizontal flip, Cutout 16.
Output: `checkpoints/baseline.pt`, `results/baseline.json`.

### 2. Evaluate with the template interface

```bash
# post-training quantization at the given bit-widths (no training)
python test.py --weight_quant_bits 8 --activation_quant_bits 8

# the final compressed model of the report
python test.py --checkpoint checkpoints/compressed/best.pt
```

`test.py` prints the test accuracy before and after compression and the
in-memory compression summary (model, weights, activations, overheads).

### 3. One compression run (Questions 2 to 4)

```bash
# post-training quantization
python compress.py --method ptq --weight_quant_bits 4 --activation_quant_bits 8

# quantization-aware fine-tuning, the same bit-width everywhere
python compress.py --method qat --weight_quant_bits 4 --activation_quant_bits 4

# mixed precision: the allocator picks one bit-width per layer for an average of 2.5 bits,
# stem and classifier weights and the linear activations stay at 8 bits
python compress.py --method qat --mixed_avg_bits 2.5 --activation_quant_bits 4 \
    --w_bits_first 8 --w_bits_last 8 --a_bits_linear 8
```

Each run writes `checkpoints/compressed/<name>.pt` (the packed model),
`results/runs/<name>.json` and a W&B run. `python compress.py --help` lists all
options (bit-widths per layer type, epochs, learning rate, distillation
weights, FP16 or FP32 overheads).

### 4. The whole sweep and the report material (Question 3)

```bash
python run_sweep.py --dry_run    # list the configurations
python run_sweep.py              # run them all, one at a time (skips finished runs)
python scripts/make_plots.py     # figures, LaTeX tables, results/summary.csv
python scripts/wandb_report.py   # W&B report with the parallel-coordinates chart
```

### 5. Tests

```bash
python -m pytest tests -q
```

The tests cover the bit-packing round trip for all bit-widths, the quantizer
grid and its hand-written gradients, the equivalence of the packed model and
the fine-tuned model, saving and loading, the size accounting against hand
counts, the activation-memory accounting and the bit allocator.

## How the sizes are measured

* **Model size**: the bytes of the buffers that the packed model holds in
  memory: bit-packed weight codes, one FP16 scale and one FP16 bias per output
  channel (the folded BatchNorm and quantizer step), 8 bytes per activation
  quantizer, 16 bytes of metadata per layer and 24 bytes for the input mean and
  standard deviation. The reference is the FP32 model with its BatchNorm
  buffers (9,083,592 bytes).
* **Weight compression ratio**: FP32 bytes of the 2,202,560 conv and linear
  weights divided by the bytes of their packed codes.
* **Activation compression ratio**: one test image is passed through the
  packed model with every activation tensor stored as bit-packed codes; the
  bytes of all those tensors are added up and compared with the same tensors
  in FP32 (counted independently with forward hooks on the uncompressed
  model). The ratio at the point of peak live memory is reported as well.
* "MB" means 2^20 bytes, as in the course template.

Weights & Biases is used for logging. Use `--wandb_mode offline` or
`--wandb_mode disabled` to run without an account.
