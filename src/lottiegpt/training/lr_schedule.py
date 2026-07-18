import math


def cosine_with_warmup_lr(step: int, base_lr: float, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * coeff)
