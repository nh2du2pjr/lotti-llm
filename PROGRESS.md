# Progress log

Living status doc for the lotti-llm project — a from-scratch ~33M param
transformer that learns the Lottie (Bodymovin) animation format, meant to
eventually be fine-tuned for text-to-animation generation. Read this first
when picking the project back up; it exists so a session interruption or a
fresh conversation doesn't lose the "why" behind decisions already made.

See `README.md` for the CC-BY-NC-SA-4.0 data-license constraint (this
project is research/personal-use scope, not commercial, unless that's
revisited).

## Status: M2 complete, M3 (full pretraining run) starting

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
- **GPU is barely utilized**: the M2 smoke test used only ~10.5GB of the
  96GB card at `micro_batch_size=4`. There is a lot of headroom to push
  batch size up for M3 — see the "M3 kickoff checklist" below, this is an
  open TODO being addressed now.

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

## Repo is public

Made public deliberately (not accidentally) so Colab could `git clone` it
without a GitHub token. The repo contains only code/config, never the
CC-BY-NC-SA-4.0 dataset itself (`data/` is gitignored and nothing under it
has ever been committed) — see the license note in README.md for the
implication this still has for anything trained *on* the data.

## M3 kickoff checklist (what's being worked on right now)

- [ ] Empirically raise `micro_batch_size` in `configs/pretrain.yaml` — test
  a few values against real GPU memory (started at 4, ~10.5GB used out of
  96GB, so there's room for a large increase) rather than guessing.
- [ ] Fetch + tokenize a much larger training shard than the M2 smoke
  test's ~10k animations. Bottlenecked by the `datasets-server` throughput
  issue above — plan for this to take a while, and prefer running it as a
  background process on the Colab VM (`nohup ... &`, log to a file, poll
  the log) over a single blocking cell, so a long fetch doesn't require
  continuously watching the notebook.
- [ ] Mount Google Drive and point `checkpoint_dir` at it before any long
  training run — the M2 smoke test used a local `/content` checkpoint dir,
  which does *not* survive a runtime reset (only reconnects survive it).
  This is the difference between "disconnect is an inconvenience" and
  "disconnect loses the whole run."
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
