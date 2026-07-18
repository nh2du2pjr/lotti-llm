"""M1 exit-criterion check: encode -> decode every animation in a fixture
file and report structural exactness + numeric reconstruction error + the
per-family clipping rate, so tokenizer.yaml's bin allocation can be tuned
against real data.

Usage:
    python scripts/round_trip_check.py [path/to/sample.jsonl]
"""
from __future__ import annotations

import sys

from lottiegpt.data.download import load_jsonl
from lottiegpt.tokenizer import schema
from lottiegpt.tokenizer.lottie_tokenizer import LottieTokenizer, TokenizeError


def _normalize(obj, is_layer: bool = False):
    """Mirror the tokenizer's own stripping/forcing so the comparison is
    against what it *intends* to preserve, not incidental cosmetic fields."""
    if isinstance(obj, dict):
        drop = schema.DROP_KEYS | ({"ddd", "ao", "sr"} if is_layer else set())
        out = {k: _normalize(v) for k, v in obj.items() if k not in drop}
        if is_layer:
            out["ddd"] = 0
            out["ao"] = 0
            out.setdefault("sr", 1)
        return out
    if isinstance(obj, list):
        return [_normalize(item, is_layer=is_layer) for item in obj]
    return obj


def numeric_diff(a, b, path="") -> list[tuple[str, float]]:
    """Collect (path, abs_error) for every numeric leaf where a/b differ, and
    every structural mismatch as an infinite-error entry."""
    diffs: list[tuple[str, float]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in keys:
            if k not in a or k not in b:
                diffs.append((f"{path}.{k}", float("inf")))
                continue
            diffs.extend(numeric_diff(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((f"{path}[len]", float("inf")))
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(numeric_diff(x, y, f"{path}[{i}]"))
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            diffs.append((path, float("inf")))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a != b:
            diffs.append((path, abs(a - b)))
    else:
        if a != b:
            diffs.append((path, float("inf")))
    return diffs


def main(fixture_path: str) -> None:
    records = load_jsonl(fixture_path)
    tok = LottieTokenizer()

    n_ok = 0
    n_tokenize_error = 0
    n_json_invalid = 0
    token_lengths = []
    max_abs_error = 0.0
    structural_mismatches = 0
    total_diffs = 0

    for i, rec in enumerate(records):
        try:
            ids = tok.encode(rec)
        except TokenizeError as e:
            n_tokenize_error += 1
            print(f"[{i}] ENCODE ERROR: {e}")
            continue

        token_lengths.append(len(ids))
        try:
            decoded = tok.decode(ids)
        except Exception as e:
            n_json_invalid += 1
            print(f"[{i}] DECODE ERROR: {e}")
            continue

        expected = {
            "fr": rec["fr"], "ip": rec["ip"], "op": rec["op"],
            "layers": [_normalize(layer, is_layer=True) for layer in rec["layers"]],
        }
        actual = {"fr": decoded["fr"], "ip": decoded["ip"], "op": decoded["op"], "layers": decoded["layers"]}

        diffs = numeric_diff(expected, actual)
        finite = [d for _, d in diffs if d != float("inf")]
        inf_count = sum(1 for _, d in diffs if d == float("inf"))
        structural_mismatches += inf_count
        total_diffs += len(diffs)
        if finite:
            max_abs_error = max(max_abs_error, max(finite))
        if inf_count == 0:
            n_ok += 1
        elif i < 5:
            for p, d in diffs[:10]:
                print(f"[{i}] MISMATCH {p}: {d}")

    n = len(records)
    print(f"\n=== round-trip summary over {n} animations ===")
    print(f"encode errors:   {n_tokenize_error}")
    print(f"decode errors:   {n_json_invalid}")
    print(f"structurally OK: {n_ok}/{n} ({100*n_ok/n:.1f}%)")
    print(f"total structural mismatches (missing/extra/type/bool/enum diffs): {structural_mismatches}")
    print(f"max numeric abs error (post-quantization): {max_abs_error:.4f}")
    if token_lengths:
        token_lengths.sort()
        print(f"token length: min={token_lengths[0]} p50={token_lengths[len(token_lengths)//2]} "
              f"p90={token_lengths[int(len(token_lengths)*0.9)]} max={token_lengths[-1]}")

    print("\n=== per-family clip rate ===")
    for name, rate in tok.vocab.quantizer.clip_report().items():
        print(f"{name:16s} {rate*100:.2f}%")

    print(f"\nvocab_size = {tok.vocab.vocab_size}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/fixtures/sample_lottie.jsonl"
    main(path)
