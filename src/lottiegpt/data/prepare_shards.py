"""Tokenize a batch of v1-supported Lottie animations into a single packed
token stream (nanoGPT-style): one big uint16 array, documents concatenated
back-to-back, no padding. Training samples random fixed-length windows from
this stream (see training/dataset.py) rather than one-doc-per-example
batching — simpler and avoids wasting compute on padding when document
lengths vary this much (see configs/tokenizer.yaml's length notes).

Animations that don't fit in `max_token_len` after encoding are dropped
entirely, not truncated — a truncated document would teach the model that
cut-off JSON is a normal thing to emit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lottiegpt.data.download import fetch_filtered_sample, is_v1_supported, load_jsonl, save_jsonl
from lottiegpt.tokenizer.lottie_tokenizer import LottieTokenizer, TokenizeError


def tokenize_corpus(
    records: list[dict],
    tokenizer: LottieTokenizer,
    max_token_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (packed_tokens uint16, doc_offsets int64) — doc_offsets[i] is
    the start index of document i in packed_tokens (for doc-aligned eval
    sampling); packed_tokens has no separator beyond each doc's own BOS/EOS."""
    chunks: list[np.ndarray] = []
    offsets = [0]
    n_dropped_long = 0
    n_dropped_error = 0
    for rec in records:
        try:
            ids = tokenizer.encode(rec)
        except (TokenizeError, KeyError):
            # KeyError catches any Bodymovin field the schema hasn't been
            # taught yet (e.g. "ml2" surfaced only after scanning ~10k real
            # animations) — one record's unfamiliar field shouldn't crash a
            # multi-minute tokenization pass over the rest of the corpus.
            n_dropped_error += 1
            continue
        if len(ids) > max_token_len:
            n_dropped_long += 1
            continue
        arr = np.array(ids, dtype=np.uint16)
        chunks.append(arr)
        offsets.append(offsets[-1] + len(arr))

    packed = np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)
    print(
        f"tokenized {len(chunks)}/{len(records)} animations "
        f"({n_dropped_long} too long, {n_dropped_error} tokenize errors), "
        f"{packed.size} tokens total"
    )
    return packed, np.array(offsets, dtype=np.int64)


def build_shard(
    out_dir: str | Path,
    num_samples: int,
    max_token_len: int,
    val_fraction: float = 0.03,
    seed: int = 0,
    source_jsonl: str | Path | None = None,
    max_scanned: int = 20000,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source_jsonl is not None and Path(source_jsonl).exists():
        records = load_jsonl(source_jsonl)
        records = [r for r in records if is_v1_supported(r)]
    else:
        # over-fetch: some fraction of v1-supported animations will still be
        # dropped below for exceeding max_token_len (see tokenize_corpus)
        records = fetch_filtered_sample(num_samples, max_scanned=max_scanned)
        if source_jsonl is not None:
            save_jsonl(records, source_jsonl)

    rng = np.random.default_rng(seed)
    ids = np.arange(len(records))
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids, train_ids = ids[:n_val], ids[n_val:]

    tok = LottieTokenizer()
    for split, split_ids in (("train", train_ids), ("val", val_ids)):
        split_records = [records[i] for i in split_ids]
        packed, offsets = tokenize_corpus(split_records, tok, max_token_len)
        np.save(out_dir / f"{split}_tokens.npy", packed)
        np.save(out_dir / f"{split}_offsets.npy", offsets)
        print(f"{split}: {len(split_records)} animations -> {packed.size} tokens")

    meta = {
        "vocab_size": tok.vocab.vocab_size,
        "max_token_len": max_token_len,
        "num_records": len(records),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote shard meta: {meta}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/shards/m2_smoke")
    parser.add_argument("--num-samples", type=int, default=30000)
    parser.add_argument("--max-token-len", type=int, default=8192)
    parser.add_argument("--source-jsonl", type=str, default=None)
    parser.add_argument("--max-scanned", type=int, default=20000)
    args = parser.parse_args()

    build_shard(
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        max_token_len=args.max_token_len,
        source_jsonl=args.source_jsonl,
        max_scanned=args.max_scanned,
    )
