from __future__ import annotations

import math

import torch

from lottiegpt.inference.generate import json_validity_rate, sample_animations
from lottiegpt.model.transformer import LottieGPT
from lottiegpt.tokenizer.lottie_tokenizer import Vocab
from lottiegpt.training.dataset import PackedShard


@torch.no_grad()
def eval_loss(model: LottieGPT, shard: PackedShard, block_size: int, batch_size: int, n_batches: int, device: str) -> dict:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = shard.get_batch(block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {"val_loss": mean_loss, "val_ppl": math.exp(min(mean_loss, 20))}


@torch.no_grad()
def eval_json_validity(model: LottieGPT, vocab: Vocab, n_samples: int, max_new_tokens: int, device: str) -> dict:
    model.eval()
    docs = sample_animations(model, vocab, n_samples=n_samples, max_new_tokens=max_new_tokens, device=device)
    model.train()
    return {"json_validity_rate": json_validity_rate(docs)}
