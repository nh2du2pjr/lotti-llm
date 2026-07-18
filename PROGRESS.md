# Progress log

Living status doc for the lotti-llm project — a from-scratch ~33M param
transformer that learns the Lottie (Bodymovin) animation format, meant to
eventually be fine-tuned for text-to-animation generation. Read this first
when picking the project back up; it exists so a session interruption or a
fresh conversation doesn't lose the "why" behind decisions already made.

See `README.md` for the CC-BY-NC-SA-4.0 data-license constraint (this
project is research/personal-use scope, not commercial, unless that's
revisited).

## Status: M2 complete, M3 (full pretraining run) in progress

**Right now (check this first):** a background data-fetch process is
running on a Colab **CPU** runtime (deliberately switched off the GPU —
see below), started via:

```
nohup python -u -m lottiegpt.data.prepare_shards \
  --out-dir /content/drive/MyDrive/lotti-llm-artifacts/shards/m3 \
  --num-samples 150000 --max-scanned 300000 --max-token-len 8192 \
  --source-jsonl /content/drive/MyDrive/lotti-llm-artifacts/raw_cache/m3_raw.jsonl \
  > /content/drive/MyDrive/lotti-llm-artifacts/m3_fetch_log.txt 2>&1 &
```

- **Running on CPU, not GPU.** Colab disconnects a session after ~1h of
  *GPU* idle time regardless of other activity, and this fetch is pure
  network+CPU work — keeping the GPU (RTX PRO 6000 Blackwell, 96GB)
  allocated and idle for the ~4h fetch would both waste quota and risk
  exactly that disconnect. Switch back to a GPU runtime (G4) only once the
  fetch+tokenize finishes and the actual training run is ready to start.
- Target: 150k v1-supported animations (up from M2's ~10k smoke shard),
  scanning up to 300k raw rows to get there (~55% v1-support rate observed).
- **Expect ~4-4.5 hours** for the fetch alone, at the ~18-20 rows/sec the
  `datasets-server` rows API sustains (see gotcha #2 below) — this is the
  dominant bottleneck, not compute.
- Check progress: `!tail -30 /content/drive/MyDrive/lotti-llm-artifacts/m3_fetch_log.txt`
  (must use `python -u`, unbuffered — a plain `python -m ...` redirected to
  a file buffers stdout and the log looks empty for a long time even though
  it's working; hit this once already, fixed by restarting with `-u`).
- **It's resumable, but not perfectly** (see gotcha #6 and especially #7
  below) — re-running the exact same command picks up from the last
  *Drive-synced* checkpoint, which lags behind the last *processed* record
  because Colab's Drive mount buffers writes. A graceful stop (let it reach
  a natural pause, or kill and wait a few seconds) before switching
  runtimes loses far less than an abrupt VM termination does — already
  lost ~10 minutes of scanning once switching GPU->CPU by not doing this.
- Once the fetch finishes, `prepare_shards` tokenizes everything and writes
  `train_tokens.npy` / `val_tokens.npy` / `meta.json` to
  `/content/drive/MyDrive/lotti-llm-artifacts/shards/m3/` — then the actual
  training run (`lottiegpt.training.pretrain --shard-dir
  .../shards/m3 --checkpoint-dir .../checkpoints/pretrain`) is the next step,
  not yet started as of this writing.

- **M0** — repo scaffold, packaging, GitHub remote (`nh2du2pjr/lotti-llm`,
  public — see "Repo is public" below).
- **M1** — tokenizer (`src/lottiegpt/tokenizer/`): Lottie JSON <-> token ids,
  100% structural round-trip verified on a 332-animation sample. Vocab size
  settled at **3916**.
- **M2** — model (`src/lottiegpt/model/transformer.py`, ~33.47M params,
  RoPE, tied embeddings) + training loop (`src/lottiegpt/training/`) +
  data-shard pipeline (`src/lottiegpt/data/`). Verified end-to-end **on real
  GPU** (Colab, NVIDIA RTX PRO 6000 Blackwell, 96GB): loss went 8.37 -> 0.59
  over 210 steps on a ~10k-animation smoke shard, checkpoint save *and*
  resume both confirmed working.
- **M3** (in progress) — full pretraining run on a much larger corpus slice,
  now that the pipeline is proven. This is the current focus.
- **M4** (not started) — text-to-animation finetuning.

## Key decisions and why

- **v1 schema scope** (`src/lottiegpt/tokenizer/schema.py` docstring has the
  full rationale): only 2D shape/null layers, no effects/masks/track-mattes/
  time-remap/AE-expressions/solid-color layers. Chosen after scanning real
  data — these excluded features are rare in "icon/UI animation" style
  Lotties (the target domain) and each would have needed a materially
  different encoding scheme (e.g. solid layers store color as a hex
  *string*, not the rgba-float-array scheme used everywhere else).
- **Tokenizer is code, not data-mined**: vocab layout and numeric-family
  classification live in `schema.py` / `lottie_tokenizer.py` as constants +
  conditional logic, not a YAML config loaded at runtime. Tried to keep it
  YAML-driven early on; abandoned that because family selection has real
  context-dependent rules (e.g. a bare `"s"` key means scale under a
  transform, size under an ellipse, trim-start% under a trim path) that
  don't fit a flat schema. `configs/tokenizer.yaml` is now a **descriptive
  snapshot only** — not loaded by any script.
- **max_seq_len = 8192**, not the ~1536 assumed in early planning. Real
  token lengths are much longer than expected (a single `{a,k}` property
  wrapper costs ~4 structural tokens on top of its value) — p50=3813,
  p90=19588 on the calibration sample. 8192 covers ~72% of animations
  losslessly; longer ones are **filtered out of training, not truncated**
  (truncation would teach the model that cut-off JSON is normal output).
- **`micro_batch_size` measured, not guessed**: probed 16/24/32 directly
  against the real card at `max_seq_len=8192` — 39.6GB / 59.1GB / 78.7GB
  peak. Settled on **24** (`configs/pretrain.yaml` and `finetune.yaml`),
  leaving ~36GB headroom for eval-time generation + allocator
  fragmentation rather than running training itself at ~83% of the card.
  48 OOM'd outright. `pretrain.py` now logs peak GPU memory alongside
  loss/lr (`--device cuda` runs only) and exposes `--micro-batch-size`/
  `--grad-accum-steps` CLI overrides specifically so this kind of probing
  doesn't require editing the config file each time.

## Gotchas hit in practice (fixed, but worth knowing about)

1. **`.gitignore` anchoring bug**: an unanchored `data/` pattern was
   silently excluding `src/lottiegpt/data/` (the real source module, not the
   dataset cache dir) from git. Fixed by anchoring to `/data/`. If a
   `src/lottiegpt/**` file ever mysteriously isn't showing up in `git
   status`, check for this class of bug first.
2. **HF `datasets-server` rows API is not built for bulk scanning.** It's
   the dataset-viewer preview endpoint, not a bulk export API — pagination
   through tens of thousands of rows hits both transient 500s and 429
   (rate-limit) errors under sustained load. `src/lottiegpt/data/download.py`
   now retries with backoff (`_fetch_batch_with_retry`, honors `Retry-After`
   on 429) and logs progress every 20 batches, but throughput is still only
   ~15-20 rows/sec. **For M3's larger fetch this is the main bottleneck** —
   if it's too slow, the fix is switching to the proper `datasets` library
   streaming path (parquet-backed, not the preview API), which needs a way
   to skip decoding the `image`/`video` columns (they require
   Pillow/torchcodec and aren't needed) — not yet implemented, flagged here
   as the first thing to try if fetch time becomes a real blocker.
3. **New Bodymovin fields keep surfacing at scale.** `"ml2"` (an
   animatable-property companion to the static `"ml"` miter-limit fallback)
   only showed up once we tokenized ~10k real animations — it wasn't in the
   332-sample M1 calibration set. `prepare_shards.tokenize_corpus` now
   catches `KeyError` per-record (in addition to the tokenizer's own
   `TokenizeError`) so one more unanticipated field drops that single
   record instead of crashing a multi-minute batch job. **Expect more of
   these** as the corpus scales further; the fix pattern is: add the key to
   `schema.STRUCTURAL_KEYS`, route it through `_classify_numeric` /
   `_property_family_for_key` in `lottie_tokenizer.py`, done.
4. **`configs/model_small.yaml`'s `vocab_size` must match the tokenizer's
   actual `Vocab().vocab_size`** (`pretrain.py` asserts this at startup).
   It drifted once already (3915 -> 3916) when `ml2` was added — the vocab
   is code-derived, so this config value doesn't update itself. If you add
   a new structural key or enum, bump this.
5. **Colab disconnects on idle**, even in the same session that stays
   allocated for up to 24h while a cell is *running*. Watching output
   without running anything can trigger a "disconnected due to inactivity"
   dialog (just click reconnect — the underlying VM and its disk survive;
   in-progress data on `/content` is safe as long as you don't restart the
   runtime, only reconnect the frontend).
6. **A background fetch needs `python -u` and resumable checkpointing, or
   a late failure loses everything.** Two separate incidents drove this:
   (a) redirecting `python -m ...` output to a log file with plain
   buffering meant `tail`/`cat` on the log showed nothing for a long time
   even though the process was working — fixed by running `python -u`
   (unbuffered) whenever output needs to be watched live; (b) a fetch that
   only buffers results in memory and saves once at the end loses *all*
   progress if it dies late (hit this once on a genuine HTTP failure after
   ~16 minutes of scanning). `fetch_filtered_sample` now takes a
   `checkpoint_path`: it appends each accepted record to that file
   immediately and persists scan offset to a `.meta.json` sidecar after
   every batch, so re-running the identical command resumes instead of
   restarting from offset 0 (verified against the real API — second call
   with an already-met target returns instantly with no extra requests).
   `prepare_shards.py`'s `--source-jsonl` flag wires this in automatically.
7. **Long-running background work should write to Drive, not `/content`** —
   but Drive-mounted writes are *not* immediately durable either.
   `/content` is wiped on a full runtime *reset* (as opposed to a mere
   frontend reconnect, which is harmless — see gotcha #5), so the M3 fetch
   checkpoint, log, and shard output all live under
   `/content/drive/MyDrive/lotti-llm-artifacts/`. That helped, but did not
   fully solve durability: Colab's Drive mount is a FUSE filesystem that
   buffers writes locally before syncing to actual Drive storage, and an
   *abrupt* VM termination (e.g. switching runtime type while a process is
   still writing, rather than stopping it first) can lose whatever hadn't
   synced yet. Hit this in practice deliberately switching GPU->CPU
   mid-fetch (see "Status" — went from ~10k scanned back to a small
   fraction) — not catastrophic this time (a few minutes of scanning, not
   hours, since it happened early), but the lesson is: **before any
   intentional runtime-type switch, gracefully stop background processes
   that write to Drive and give them a few seconds before switching**,
   rather than relying on the checkpoint alone to survive an abrupt
   termination.

## Repo is public

Made public deliberately (not accidentally) so Colab could `git clone` it
without a GitHub token. The repo contains only code/config, never the
CC-BY-NC-SA-4.0 dataset itself (`data/` is gitignored and nothing under it
has ever been committed) — see the license note in README.md for the
implication this still has for anything trained *on* the data.

## M3 kickoff checklist

- [x] Empirically raise `micro_batch_size` — measured 16/24/32, settled on
  24 (see "Key decisions" above).
- [x] Mount Google Drive; all M3 artifacts (fetch checkpoint, log, shard
  output, and `checkpoint_dir` in `configs/pretrain.yaml`) point at
  `/content/drive/MyDrive/lotti-llm-artifacts/`.
- [x] Kick off the large fetch+tokenize as a resumable background process
  (see "Status" section at the top — running as of this writing).
- [ ] **Not started yet**: the actual M3 training run
  (`lottiegpt.training.pretrain --shard-dir .../shards/m3 --checkpoint-dir
  .../checkpoints/pretrain`), once the fetch above finishes. Use
  `configs/pretrain.yaml`'s `max_steps`/`eval_every` as a starting point,
  adjust `max_steps` upward if the corpus ends up smaller/larger than the
  150k target changes the compute-optimal token budget (see the plan's
  original ~20 tokens/param Chinchilla-style estimate).
- [ ] Once training is running for real, watch the JSON-validity-rate eval
  metric (`training/eval.py`), not just loss — loss dropping fast on a
  small/repetitive corpus doesn't by itself confirm the model is learning
  generalizable Lottie structure vs. overfitting the sample.

## Where things live

- Tokenizer/model/training code: `src/lottiegpt/`
- Config snapshots: `configs/` (`model_small.yaml` is the real architecture
  config; `tokenizer.yaml` is descriptive-only, see above)
- Local fixture corpus (gitignored, regenerate via
  `python -m lottiegpt.data.download`): `data/fixtures/sample_lottie.jsonl`
- Tests: `tests/` (`pytest tests/` — 15 tests, all passing as of M2)
- Colab notebook used for M2: a fresh `Untitled0.ipynb` created directly in
  the Colab UI (not committed to the repo — the repo's own
  `notebooks/00_setup_check.ipynb` is the thin clone+GPU-check cell, real
  work happens by running `python -m lottiegpt....` modules from cells, not
  notebook-native code)
