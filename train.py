"""Train the full-precision MobileNet-v2 baseline on CIFAR-10 (Question 1).

    python train.py                      # the configuration used in the report
    python train.py --epochs 5 --name smoke --wandb_mode disabled

Outputs
    checkpoints/<name>.pt          final weights (state_dict + config + metrics)
    checkpoints/<name>_last.pt     resumable training state, rewritten every epoch
    results/<name>.json            per-epoch history and final metrics

The run can be interrupted at any time; `python train.py --resume` continues
from the last finished epoch (same W&B run, same random streams).
"""
import argparse
import os
import time

import torch
import yaml

from mbv2.data import GpuAugment, load_cifar10
from mbv2.engine import build_sgd, evaluate, train_one_epoch, warmup_cosine
from mbv2.model import count_params, mobilenet_v2
from mbv2.utils import atomic_torch_save, get_device, save_json, set_seed, str2bool

DEFAULTS = dict(
    name="baseline",
    # model
    width_mult=1.0, dropout=0.2,
    # optimisation
    epochs=200, batch_size=128, lr=0.1, momentum=0.9, nesterov=True,
    weight_decay=5e-4, warmup_epochs=5, label_smoothing=0.1,
    # data
    cutout=16, val_size=0, data_dir="./data",
    # misc
    seed=42, device="auto", out_dir=".",
    wandb_project="cs6886w-a2-mobilenetv2-compression", wandb_mode=None, resume=False,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML file with any of the options below")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=str2bool, nargs="?", const=True)
        elif v is None:
            p.add_argument(f"--{k}")
        else:
            p.add_argument(f"--{k}", type=type(v))
    args = p.parse_args(argv)
    # Precedence: defaults < YAML file < explicit command-line flags.
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})
    return cfg


def init_wandb(cfg, run_id=None):
    mode = cfg["wandb_mode"] or os.environ.get("WANDB_MODE", "online")
    if mode == "disabled":
        return None
    import wandb
    run = wandb.init(project=cfg["wandb_project"], name=cfg["name"], job_type="baseline",
                     config=cfg, mode=mode, dir=cfg["out_dir"], id=run_id, resume="allow")
    wandb.define_metric("epoch")
    for k in ("train_loss", "train_acc", "test_loss", "test_acc", "lr"):
        wandb.define_metric(k, step_metric="epoch")
    return run


def train(cfg):
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    name = cfg["name"]
    ckpt_dir = os.path.join(cfg["out_dir"], "checkpoints")
    last_path = os.path.join(ckpt_dir, f"{name}_last.pt")

    train_set, val_set, test_set = load_cifar10(cfg["data_dir"], device, cfg["val_size"])
    eval_set = val_set if val_set is not None else test_set  # monitored every epoch
    eval_name = "val" if val_set is not None else "test"
    train_tf = GpuAugment(device, augment=True, cutout=cfg["cutout"], seed=cfg["seed"])
    eval_tf = GpuAugment(device, augment=False)
    shuffle_gen = torch.Generator().manual_seed(cfg["seed"])

    model = mobilenet_v2(10, cfg["width_mult"], cfg["dropout"]).to(device)
    optimizer = build_sgd(model, cfg["lr"], cfg["momentum"], cfg["weight_decay"], cfg["nesterov"])
    steps_per_epoch = len(train_set) // cfg["batch_size"]
    scheduler = warmup_cosine(optimizer, cfg["epochs"] * steps_per_epoch,
                              cfg["warmup_epochs"] * steps_per_epoch)

    history, start_epoch, run_id, elapsed = [], 1, None, 0.0
    if cfg["resume"] and os.path.exists(last_path):
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        train_tf.gen.set_state(state["aug_gen"])
        shuffle_gen.set_state(state["shuffle_gen"])
        history, run_id, elapsed = state["history"], state["wandb_id"], state["elapsed"]
        start_epoch = len(history) + 1
        print(f"[{name}] resuming at epoch {start_epoch}")

    run = init_wandb(cfg, run_id)
    print(f"[{name}] device={device} params={count_params(model):,} "
          f"train={len(train_set)} {eval_name}={len(eval_set)} steps/epoch={steps_per_epoch}")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_set, cfg["batch_size"], optimizer, scheduler,
                                          train_tf, shuffle_gen, cfg["label_smoothing"])
        ev_loss, ev_acc = evaluate(model, eval_set, eval_tf)
        dt = time.time() - t0
        elapsed += dt
        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
               f"{eval_name}_loss": ev_loss, f"{eval_name}_acc": ev_acc, "lr": lr, "epoch_time": dt}
        history.append(row)
        if run is not None:
            run.log(row)
        print(f"[{name}] ep {epoch:3d}/{cfg['epochs']} lr {lr:.2e} | train {tr_loss:.4f} "
              f"{tr_acc:6.2f}% | {eval_name} {ev_loss:.4f} {ev_acc:6.2f}% | {dt:.0f}s", flush=True)
        if tr_loss != tr_loss:  # NaN
            raise RuntimeError(f"training diverged at epoch {epoch}")
        atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "history": history,
                           "aug_gen": train_tf.gen.get_state(),
                           "shuffle_gen": shuffle_gen.get_state(),
                           "wandb_id": run.id if run is not None else None,
                           "elapsed": elapsed, "config": cfg}, last_path)

    # The reported model is the one after the last epoch: nothing is selected
    # on the test set.
    test_loss, test_acc = evaluate(model, test_set, eval_tf)
    train_loss, train_acc = evaluate(model, train_set, eval_tf)  # clean (un-augmented) train set
    final = {"test_acc": test_acc, "test_loss": test_loss, "train_acc_clean": train_acc,
             "train_loss_clean": train_loss, "epochs": len(history), "params": count_params(model),
             "train_time_s": elapsed,
             "best_epoch_test_acc": max(h.get("test_acc", 0.0) for h in history)}
    print(f"[{name}] final test top-1 {test_acc:.2f}% (clean train {train_acc:.2f}%)")
    atomic_torch_save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                       "config": cfg, "metrics": final}, os.path.join(ckpt_dir, f"{name}.pt"))
    save_json({"name": name, "config": cfg, "history": history, "final": final},
              os.path.join(cfg["out_dir"], "results", f"{name}.json"))
    if run is not None:
        run.summary.update(final)
        run.finish()
    return final


if __name__ == "__main__":
    train(parse_args())
