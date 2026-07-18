"""Random fixed-length window sampling over a packed token shard (see
data/prepare_shards.py) — the standard nanoGPT-style get_batch, not a
per-document torch Dataset. Simpler, and avoids padding waste given how
widely document lengths vary in this corpus."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class PackedShard:
    def __init__(self, shard_dir: str | Path, split: str, mmap: bool = True):
        shard_dir = Path(shard_dir)
        mode = "r" if mmap else None
        self.tokens = np.load(shard_dir / f"{split}_tokens.npy", mmap_mode=mode)
        self.offsets = np.load(shard_dir / f"{split}_offsets.npy")

    def __len__(self) -> int:
        return len(self.tokens)

    def get_batch(self, block_size: int, batch_size: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.tokens)
        if n <= block_size + 1:
            raise ValueError(
                f"shard has only {n} tokens, need > block_size+1 ({block_size + 1}) to sample a window"
            )
        starts = np.random.randint(0, n - block_size - 1, size=batch_size)
        x = np.stack([self.tokens[s:s + block_size].astype(np.int64) for s in starts])
        y = np.stack([self.tokens[s + 1:s + block_size + 1].astype(np.int64) for s in starts])
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        if device != "cpu":
            x_t = x_t.pin_memory().to(device, non_blocking=True)
            y_t = y_t.pin_memory().to(device, non_blocking=True)
        return x_t, y_t
