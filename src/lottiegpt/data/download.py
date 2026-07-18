"""Pull a filtered sample of native Lottie JSON animations from OmniLottie/MMLottie-2M
via the HF datasets-server rows API (avoids the `datasets` library's image/video
column decoding, which pulls in heavy optional deps we don't need for text-only work).

v1 scope is restricted to animations that are fully expressible with the simplified
schema in `lottiegpt.tokenizer.schema`: 2D only, no external assets (no precomp/image/
audio/text/camera layers) — see schema.py for the rationale.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from lottiegpt.tokenizer import schema

DATASET = "OmniLottie/MMLottie-2M"
CONFIG = "Lottie"
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset={DATASET.replace('/', '%2F')}&config={CONFIG}&split=train"
)


def _contains_key(obj, keys: set[str]) -> bool:
    """True if `keys` appears anywhere in the (possibly nested) structure."""
    if isinstance(obj, dict):
        if keys & obj.keys():
            return True
        return any(_contains_key(v, keys) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_key(item, keys) for item in obj)
    return False


def _all_shape_types_supported(obj) -> bool:
    if isinstance(obj, dict):
        ty = obj.get("ty")
        if isinstance(ty, str) and ty not in schema.SUPPORTED_SHAPE_TYPES:
            return False
        return all(_all_shape_types_supported(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_all_shape_types_supported(item) for item in obj)
    return True


def _has_expression(obj) -> bool:
    """True if any property carries an AE expression ("x": "<js code>").
    "x" is also the (numeric) easing-handle x-component key, so this must
    check the value type, not just key presence."""
    if isinstance(obj, dict):
        if isinstance(obj.get("x"), str):
            return True
        return any(_has_expression(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_expression(item) for item in obj)
    return False


def is_v1_supported(lottie_json: dict) -> bool:
    """Reject anything the v1 schema (schema.py) can't losslessly round-trip:
    3D, external assets, effects/masks/track-mattes/time-remap, non-standard
    canvas size, layer flags outside their assumed-constant defaults, or any
    shape type outside the supported set."""
    layers = lottie_json.get("layers", [])
    if not layers:
        return False
    if lottie_json.get("ddd", 0) != 0:
        return False
    if lottie_json.get("assets"):
        return False
    if lottie_json.get("w") != schema.CANVAS_SIZE or lottie_json.get("h") != schema.CANVAS_SIZE:
        return False
    if _contains_key(lottie_json, schema.REJECT_IF_PRESENT):
        return False
    if _has_expression(lottie_json):
        return False
    for layer in layers:
        if layer.get("ty") not in schema.SUPPORTED_LAYER_TYPES:
            return False
        if layer.get("ddd", 0) != 0:
            return False
        if layer.get("ao", 0) != 0:
            return False
        if layer.get("sr", 1) != 1:
            return False
        if schema.REJECT_IF_LAYER_HAS_KEY & layer.keys():
            return False
    if not _all_shape_types_supported(lottie_json):
        return False
    return True


def _fetch_batch_with_retry(url: str, max_retries: int = 8, timeout: int = 30) -> dict:
    """The datasets-server API occasionally returns a transient 5xx under
    sustained pagination load, and rate-limits (429) unauthenticated clients
    making frequent requests — without a retry, one bad batch after minutes
    of scanning throws away all progress (this happened in practice twice:
    once on a 500, once on sustained 429s)."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            last_error = e
            retry_after = e.headers.get("Retry-After") if e.headers else None
            if e.code == 429:
                backoff = float(retry_after) if retry_after else min(10 * (attempt + 1), 90)
            else:
                backoff = min(2 ** attempt, 30)
            print(f"  [retry {attempt + 1}/{max_retries}] HTTP {e.code} — retrying in {backoff:.0f}s")
            time.sleep(backoff)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_error = e
            backoff = min(2 ** attempt, 30)
            print(f"  [retry {attempt + 1}/{max_retries}] {e} — retrying in {backoff}s")
            time.sleep(backoff)
    raise RuntimeError(f"giving up after {max_retries} retries") from last_error


def fetch_filtered_sample(
    target_count: int,
    max_scanned: int = 20000,
    batch_size: int = 100,
    sleep_s: float = 1.0,
    progress_every: int = 20,
) -> list[dict]:
    """Scan the dataset in order (offset 0, 100, 200, ...) and keep animations
    matching `is_v1_supported`, until `target_count` are collected or
    `max_scanned` rows have been examined."""
    kept: list[dict] = []
    offset = 0
    scanned = 0
    batches = 0
    while len(kept) < target_count and scanned < max_scanned:
        url = f"{ROWS_URL}&offset={offset}&length={batch_size}"
        data = _fetch_batch_with_retry(url)
        rows = data.get("rows", [])
        if not rows:
            break
        for row in rows:
            scanned += 1
            raw = row["row"]["lottie_json"]
            lj = json.loads(raw) if isinstance(raw, str) else raw
            if is_v1_supported(lj):
                kept.append(lj)
        offset += batch_size
        batches += 1
        if batches % progress_every == 0:
            print(f"  scanned {scanned}, kept {len(kept)} ({100 * len(kept) / scanned:.0f}%)")
        time.sleep(sleep_s)
    return kept


def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--out", type=str, default="data/fixtures/sample_lottie.jsonl")
    args = parser.parse_args()

    sample = fetch_filtered_sample(args.count)
    save_jsonl(sample, args.out)
    print(f"saved {len(sample)} animations to {args.out}")
