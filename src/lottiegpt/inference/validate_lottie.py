"""Minimal structural validity check for a decoded Lottie animation — used as
the "JSON-validity rate" training metric (see plan: track this because token
generation can produce truncated/malformed output that the tokenizer's
decode() will have already patched up structurally, but the *content* might
still be nonsensical, e.g. an empty layer list)."""
from __future__ import annotations

REQUIRED_TOP_KEYS = {"v", "fr", "ip", "op", "w", "h", "layers"}


def is_valid_lottie(doc: dict) -> tuple[bool, str]:
    if not isinstance(doc, dict):
        return False, "not a dict"
    missing = REQUIRED_TOP_KEYS - doc.keys()
    if missing:
        return False, f"missing top-level keys: {missing}"
    if not isinstance(doc["layers"], list) or len(doc["layers"]) == 0:
        return False, "layers is empty or not a list"
    for i, layer in enumerate(doc["layers"]):
        if not isinstance(layer, dict):
            return False, f"layer {i} is not a dict"
        if "ty" not in layer or "ks" not in layer:
            return False, f"layer {i} missing ty/ks"
    return True, "ok"
