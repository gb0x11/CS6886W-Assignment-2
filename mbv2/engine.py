"""Training and evaluation loops shared by baseline training and QAT fine-tuning."""
import torch
import torch.nn.functional as F

from .data import iterate


def distillation_loss(student_logits, teacher_logits, temperature: float):
    """KL(teacher || student) on temperature-softened distributions (Hinton et al.)."""
    t = temperature
    log_p = F.log_softmax(student_logits / t, dim=1)
    q = F.softmax(teacher_logits / t, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (t * t)


def train_one_epoch(model, split, batch_size, optimizer, scheduler, transform, gen,
                    label_smoothing=0.0, teacher=None, kd_alpha=0.0, kd_temperature=4.0,
                    grad_clip=0.0):
    """One pass over the training split.

    If `teacher` is given, the loss is
        (1 - kd_alpha) * CE(student, labels) + kd_alpha * KD(student, teacher),
    which is used during quantization-aware fine-tuning with the full-precision
    network as the teacher.

    Returns mean loss and accuracy (%) measured on the augmented batches as
    they are trained on (the usual "running" training metrics).
    """
    model.train()
    device = split.labels.device
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    n = 0
    for x, y in iterate(split, batch_size, transform, shuffle=True, gen=gen, drop_last=True):
        out = model(x)
        loss = F.cross_entropy(out, y, label_smoothing=label_smoothing)
        if teacher is not None and kd_alpha > 0:
            with torch.no_grad():
                t_out = teacher(x)
            loss = (1 - kd_alpha) * loss + kd_alpha * distillation_loss(out, t_out, kd_temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Accumulate on device; one host sync at the end of the epoch.
        loss_sum += loss.detach() * len(y)
        correct += (out.argmax(1) == y).sum()
        n += len(y)
    return (loss_sum / n).item(), (100.0 * correct / n).item()


@torch.no_grad()
def evaluate(model, split, transform, batch_size=1000):
    """Mean cross-entropy loss and top-1 accuracy (%) on a split."""
    model.eval()
    device = split.labels.device
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    for x, y in iterate(split, batch_size, transform):
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum")
        correct += (out.argmax(1) == y).sum()
    n = len(split)
    return (loss_sum / n).item(), (100.0 * correct / n).item()


@torch.no_grad()
def predict(model, split, transform, batch_size=1000):
    """Predicted class index for every image of a split."""
    model.eval()
    return torch.cat([model(x).argmax(1) for x, _ in iterate(split, batch_size, transform)])


def build_sgd(model, lr, momentum=0.9, weight_decay=5e-4, nesterov=True, extra_no_decay=()):
    """SGD where weight decay is applied to conv / linear weights only.

    BatchNorm scales and shifts, biases and (for QAT) quantizer step sizes are
    one-dimensional parameters that should not be pulled towards zero.
    """
    decay, no_decay = [], []
    skip = {id(p) for p in extra_no_decay}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or id(p) in skip else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=nesterov)


def warmup_cosine(optimizer, total_steps, warmup_steps=0):
    """Per-step schedule: linear warm-up from 0.01x to 1x, then cosine decay to 0."""
    import math

    def factor(step):
        if step < warmup_steps:
            return 0.01 + 0.99 * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
