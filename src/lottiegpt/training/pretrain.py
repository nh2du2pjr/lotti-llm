"""Causal-LM pretraining loop over packed Lottie token shards.

Usage (from repo root, after `pip install -e .`):
    python -m lottiegpt.training.pretrain --config configs/pretrain.yaml \
        --shard-dir data/shards/m2_smoke --device cuda

Checkpointing is time-based (checkpoint_every_minutes) rather than purely
step-based, since Colab sessions can end at any point in wall-clock time,
not at a convenient step boundary.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from lottiegpt.model.config import ModelConfig
from lottiegpt.model.transformer import LottieGPT
from lottiegpt.tokenizer.lottie_tokenizer import Vocab
from lottiegpt.training.dataset import PackedShard
from lottiegpt.training.eval import eval_json_validity, eval_loss
from lottiegpt.training.lr_schedule import cosine_with_warmup_lr


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(path: Path, model, optimizer, step: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "model_config": model.cfg.__dict__,
        },
        tmp_path,
    )
    tmp_path.replace(path)  # atomic-ish on the same filesystem
    print(f"[checkpoint] saved step {step} -> {path}")


def prune_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    ckpts = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    for stale in ckpts[:-keep_last_n]:
        stale.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--shard-dir", type=str, default="data/shards/m2_smoke")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = ModelConfig.from_yaml(cfg["model_config"])
    max_steps = args.max_steps or cfg["max_steps"]
    checkpoint_dir = Path(args.checkpoint_dir or cfg["checkpoint_dir"])
    device = args.device

    train_shard = PackedShard(args.shard_dir, "train")
    val_shard = PackedShard(args.shard_dir, "val")
    vocab = Vocab()
    assert vocab.vocab_size == model_cfg.vocab_size, (
        f"vocab mismatch: tokenizer={vocab.vocab_size} model_config={model_cfg.vocab_size}"
    )

    model = LottieGPT(model_cfg).to(device)
    print(f"model params: {model.num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )

    start_step = 0
    if args.resume_from_checkpoint:
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"resumed from {args.resume_from_checkpoint} at step {start_step}")

    block_size = model_cfg.max_seq_len
    micro_bs = cfg["batch"]["micro_batch_size"]
    grad_accum = cfg["batch"]["grad_accum_steps"]
    use_amp = cfg.get("precision", "bf16") == "bf16" and device == "cuda"
    amp_dtype = torch.bfloat16

    last_checkpoint_time = time.time()
    checkpoint_every_s = cfg["checkpoint_every_minutes"] * 60

    model.train()
    t0 = time.time()
    for step in range(start_step, max_steps):
        lr = cosine_with_warmup_lr(step, cfg["optim"]["lr"], cfg["optim"]["warmup_steps"], max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum):
            x, y = train_shard.get_batch(block_size, micro_bs, device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            (loss / grad_accum).backward()
            accum_loss += loss.item() / grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
        optimizer.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | {dt:.1f}s")
            t0 = time.time()

        if step % cfg["eval_every"] == 0 and step > 0:
            metrics = eval_loss(model, val_shard, block_size, micro_bs, n_batches=10, device=device)
            print(f"[eval step {step}] {metrics}")
            validity = eval_json_validity(model, vocab, n_samples=8, max_new_tokens=1024, device=device)
            print(f"[eval step {step}] {validity}")

        if time.time() - last_checkpoint_time > checkpoint_every_s or step == max_steps - 1:
            save_checkpoint(checkpoint_dir / f"step_{step}.pt", model, optimizer, step, cfg)
            prune_checkpoints(checkpoint_dir, cfg["keep_last_n_checkpoints"])
            last_checkpoint_time = time.time()

    print("training complete")


if __name__ == "__main__":
    main()
