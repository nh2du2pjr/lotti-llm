"""Lottie JSON <-> token id sequence, for the v1 schema in `schema.py`.

Design summary (see schema.py docstring for the full rationale):

- Structure (dicts/lists/keys) is encoded as a pre-order tree walk with
  BEGIN_DICT/END_DICT/BEGIN_LIST/END_LIST markers and one KEY_<name> token
  per retained dict key. This is fully generic — it doesn't matter whether a
  dict is "a layer" or "a shape item", the same four structural tokens plus
  KEY tokens losslessly capture arbitrary Bodymovin nesting.
- Closed-set string/int values (shape type, blend mode, ...) get a dedicated
  ENUM_<name>_<value> token.
- Continuous values go through `quantize.Quantizer`, keyed by a "numeric
  family" (position, rotation, opacity, ...) chosen from local context
  (current key, enclosing shape type, and — for keyframe fields — the family
  of the animated property being encoded, since a keyframe's "s"/"e" fields
  don't carry that information on their own).
- Because each family owns a *disjoint* range of the vocab, decoding a
  numeric token back to a float needs no context at all: the token id alone
  says which family (and thus which (lo, hi, bins)) to use. Context tracking
  is therefore only needed on the encode side.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from lottiegpt.tokenizer import schema
from lottiegpt.tokenizer.quantize import Quantizer

SPECIAL_TOKENS = [
    "PAD", "BOS", "EOS", "SEP", "UNK",
    "BEGIN_DICT", "END_DICT", "BEGIN_LIST", "END_LIST",
    "NULL",
]

# per-layer / top-level fields we force to a constant instead of tokenizing
LAYER_FORCED_KEYS = {"ddd", "ao", "sr"}
TOP_FORCED_KEYS = {"v", "ddd", "assets", "w", "h", "nm", "markers", "meta", "props"}


def _build_enum_table() -> tuple[dict[tuple[str, Any], int], dict[int, tuple[str, Any]]]:
    """Flat (enum_name, value) -> local id table, shared across all enum families."""
    fwd: dict[tuple[str, Any], int] = {}
    rev: dict[int, tuple[str, Any]] = {}
    i = 0
    for enum_name, values in schema.ENUMS.items():
        for v in values:
            fwd[(enum_name, v)] = i
            rev[i] = (enum_name, v)
            i += 1
    # the "a" flag on animated-property wrappers is its own tiny 0/1 enum
    for v in (0, 1):
        fwd[("a_flag", v)] = i
        rev[i] = ("a_flag", v)
        i += 1
    return fwd, rev


class Vocab:
    """Deterministic, code-defined token layout (not mined from data — see
    schema.py). Layout, in order: special tokens, KEY_<name> tokens, ENUM
    tokens, one contiguous block per numeric family, then a reserved range
    for the text sub-tokenizer trained in M4 (untouched/unused until then)."""

    def __init__(self, text_subvocab_size: int = 1450):
        self.quantizer = Quantizer()
        offset = 0

        self.special_ids = {name: i for i, name in enumerate(SPECIAL_TOKENS)}
        offset += len(SPECIAL_TOKENS)

        self.key_ids: dict[str, int] = {}
        for key in schema.STRUCTURAL_KEYS:
            self.key_ids[key] = offset
            offset += 1
        self.key_ids_rev = {v: k for k, v in self.key_ids.items()}

        enum_fwd, enum_rev = _build_enum_table()
        self.enum_base = offset
        self.enum_fwd = enum_fwd  # (enum_name, value) -> local id
        self.enum_rev = enum_rev  # local id -> (enum_name, value)
        offset += len(enum_fwd)

        self.family_base: dict[str, int] = {}
        for family in schema.NUMERIC_FAMILIES:
            self.family_base[family] = offset
            offset += self.quantizer.bins(family)
        self.family_ranges = [
            (base, base + self.quantizer.bins(name), name)
            for name, base in self.family_base.items()
        ]

        self.text_base = offset
        self.text_subvocab_size = text_subvocab_size
        offset += text_subvocab_size

        self.vocab_size = offset

    # -- special / structural -------------------------------------------------
    def special(self, name: str) -> int:
        return self.special_ids[name]

    def key(self, name: str) -> int:
        if name not in self.key_ids:
            raise KeyError(f"key {name!r} not in STRUCTURAL_KEYS vocab")
        return self.key_ids[name]

    def key_name(self, token_id: int) -> str | None:
        return self.key_ids_rev.get(token_id)

    def enum(self, enum_name: str, value: Any) -> int:
        return self.enum_base + self.enum_fwd[(enum_name, value)]

    def is_enum(self, token_id: int) -> bool:
        return self.enum_base <= token_id < self.enum_base + len(self.enum_fwd)

    def decode_enum(self, token_id: int) -> Any:
        _, value = self.enum_rev[token_id - self.enum_base]
        return value

    # -- numeric ---------------------------------------------------------------
    def numeric(self, family: str, value: float) -> int:
        return self.family_base[family] + self.quantizer.encode(family, value)

    def is_numeric(self, token_id: int) -> str | None:
        for lo, hi, name in self.family_ranges:
            if lo <= token_id < hi:
                return name
        return None

    def decode_numeric(self, token_id: int) -> float:
        for lo, hi, name in self.family_ranges:
            if lo <= token_id < hi:
                return self.quantizer.decode(name, token_id - lo)
        raise ValueError(f"token {token_id} is not in any numeric family range")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@dataclass
class _Ctx:
    shape_ty: str | None = None
    property_family: str | None = None
    in_keyframe: bool = False


def _classify_numeric(key: str, ctx: _Ctx) -> tuple[str, str] | tuple[str, None]:
    """Returns ("numeric", family) or ("enum", enum_name) for a scalar value
    found under `key` in the given context.

    Note: keyframe "s"/"e" fields are relabeled to key="k" by the caller
    (_encode_value) before recursing, with in_keyframe cleared — so a plain
    property *value* (whether it came from a static "k" or a keyframe's "s"/
    "e") is always classified by the single `key == "k"` rule below, using
    the family threaded through ctx.property_family. This function therefore
    only ever sees in_keyframe=True for t/i/o/to/ti/h, the fields that are
    unique to keyframe dicts.
    """
    if ctx.in_keyframe:
        if key == "t":
            return ("numeric", "time_frame")
        if key in ("i", "o", "x", "y"):
            # "i"/"o" are usually {"x":..,"y":..} dicts (handled generically
            # by the dict path) but some exports use a bare scalar directly;
            # "x"/"y" are their scalar/per-component leaves either way.
            return ("numeric", "easing")
        if key in ("to", "ti"):
            return ("numeric", "position")
        return ("numeric", "misc_scalar")

    if key == "k" and ctx.property_family:
        # the value of a property (static "k", or a relabeled keyframe "s"/
        # "e") always takes the family of its enclosing property.
        return ("numeric", ctx.property_family)
    if key in ("v", "i", "o"):
        # bezier path vertices ("v") and their tangent handles ("i"/"o") —
        # only reachable here (in_keyframe=False) for path shape data, where
        # they're always position-like coordinates.
        return ("numeric", "position")
    if key == "a":
        # the animated-property flag ("a": 0/1) as a *leaf*. The anchor-point
        # *property* is also stored under an outer key "a", but that's a
        # dict (the {a,k} wrapper) resolved by _property_family_for_key at
        # the wrapper-detection site, never through this leaf classifier.
        return ("enum", "a_flag")
    if key == "fr":
        return ("numeric", "frame_rate")
    if key in ("ip", "op", "st"):
        return ("numeric", "time_frame")
    if key == "p" and ctx.shape_ty in ("gf", "gs"):
        return ("numeric", "count_int")  # gradient stop count
    if key in ("ind", "parent", "pt"):
        return ("numeric", "count_int")
    if key == "c" and ctx.shape_ty == "rp":
        return ("numeric", "count_int")  # repeater copies
    if key == "c":
        return ("numeric", "color_channel")
    if key in ("or", "os", "ir", "is", "ml"):
        return ("numeric", "misc_scalar")
    if key == "w" and ctx.shape_ty in ("st", "gs"):
        return ("numeric", "stroke_width")
    if key == "p":
        return ("numeric", "position")
    if key in ("r", "sk", "sa"):
        return ("numeric", "rotation_deg")
    if key == "g":
        return ("numeric", "misc_scalar")
    if key == "m":
        if ctx.shape_ty == "tm":
            return ("enum", "trim_multiple")
        if ctx.shape_ty == "rp":
            return ("enum", "repeater_composite")
        return ("numeric", "misc_scalar")
    if key == "mm":
        return ("enum", "mm_mode")
    if key == "lc":
        return ("enum", "lc")
    if key == "lj":
        return ("enum", "lj")
    if key == "bm":
        return ("enum", "bm")
    if key == "sy":
        return ("enum", "sy")
    if key == "t":
        return ("enum", "gradient_type")
    return ("numeric", "misc_scalar")


def _property_family_for_key(outer_key: str, shape_ty: str | None) -> str:
    """Family for an animated-property wrapper ({"a":.., "k":..}), keyed by
    the name it's stored under in *its own parent* dict. Deliberately
    separate from `_classify_numeric`, which classifies leaf scalars — the
    two must not share a dispatch table because a handful of key names (most
    notably "a") mean different things as a wrapper key vs. as a leaf."""
    if outer_key == "a":
        return "position"  # anchor point
    if outer_key == "p":
        return "position"
    if outer_key == "ks":
        return "position"  # path shape's own bezier data wrapper
    if outer_key == "s":
        if shape_ty in ("el", "rc", "sr"):
            return "position"
        if shape_ty == "tm":
            return "percent"
        return "scale_pct"
    if outer_key == "e":
        return "percent" if shape_ty == "tm" else "misc_scalar"
    if outer_key in ("r", "sk", "sa"):
        return "rotation_deg"
    if outer_key == "o":
        return "percent"
    if outer_key == "c":
        return "count_int" if shape_ty == "rp" else "color_channel"
    if outer_key == "w":
        return "stroke_width" if shape_ty in ("st", "gs") else "misc_scalar"
    if outer_key in ("or", "os", "ir", "is", "ml", "g"):
        return "misc_scalar"
    if outer_key == "pt":
        return "count_int"
    if outer_key in ("so", "eo"):
        return "percent"
    return "misc_scalar"


class TokenizeError(ValueError):
    pass


class LottieTokenizer:
    def __init__(self, vocab: Vocab | None = None):
        self.vocab = vocab or Vocab()

    # -- public API -------------------------------------------------------
    def encode(self, lottie_json: dict) -> list[int]:
        v = self.vocab
        body = {
            "fr": lottie_json["fr"],
            "ip": lottie_json["ip"],
            "op": lottie_json["op"],
            "layers": lottie_json["layers"],
        }
        tokens = [v.special("BOS")]
        tokens += self._encode_value(body, key="__root__", ctx=_Ctx())
        tokens.append(v.special("EOS"))
        return tokens

    def decode(self, token_ids: list[int]) -> dict:
        v = self.vocab
        ids = list(token_ids)
        if ids and ids[0] == v.special("BOS"):
            ids = ids[1:]
        if ids and ids[-1] == v.special("EOS"):
            ids = ids[:-1]
        body = _StackDecoder(v).run(ids)
        layers = [self._apply_layer_defaults(layer) for layer in body.get("layers", [])]
        return {
            "v": schema.FORCED_CONSTANTS["v"],
            "fr": body["fr"],
            "ip": body["ip"],
            "op": body["op"],
            "w": schema.FORCED_CONSTANTS["w"],
            "h": schema.FORCED_CONSTANTS["h"],
            "ddd": 0,
            "assets": [],
            "layers": layers,
            "markers": [],
        }

    # -- encode helpers -----------------------------------------------------
    def _apply_layer_defaults(self, layer: dict) -> dict:
        layer = dict(layer)
        layer["ddd"] = 0
        layer["ao"] = 0
        layer.setdefault("sr", 1)
        return layer

    def _strip_layer(self, layer: dict) -> dict:
        return {k: v for k, v in layer.items() if k not in LAYER_FORCED_KEYS and k not in schema.DROP_KEYS}

    def _strip_generic(self, d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in schema.DROP_KEYS}

    def _encode_value(self, value: Any, key: str, ctx: _Ctx) -> list[int]:
        v = self.vocab
        toks: list[int] = []

        if isinstance(value, dict):
            is_layer = key == "layers"  # value here is one layer dict (called per-item from list handling)
            d = self._strip_layer(value) if is_layer else self._strip_generic(value)

            local_ctx = ctx
            if d.get("ty") in schema.SUPPORTED_SHAPE_TYPES:
                local_ctx = replace(local_ctx, shape_ty=d["ty"])
            if "k" in d and "a" in d:
                # this dict is an animated-property wrapper; its family is
                # determined by the key it's stored under in the parent, and
                # threaded down so nested keyframes' s/e can use it.
                prop_family = _property_family_for_key(key, local_ctx.shape_ty)
                local_ctx = replace(local_ctx, property_family=prop_family)

            toks.append(v.special("BEGIN_DICT"))
            for k, val in d.items():
                toks.append(v.key(k))  # real key, always — decode depends on it
                if local_ctx.in_keyframe and k in ("s", "e"):
                    # a keyframe's start/end holds "the property value at
                    # this time", exactly like a static property's "k" field
                    # — relabel so it goes through the same classification
                    # (and, if it's a list, the same property_family-based
                    # element classification) rather than keyframe-field
                    # rules that don't apply to it.
                    toks += self._encode_value(val, key="k", ctx=replace(local_ctx, in_keyframe=False))
                else:
                    toks += self._encode_value(val, key=k, ctx=local_ctx)
            toks.append(v.special("END_DICT"))
            return toks

        if isinstance(value, list):
            is_keyframe_list = key == "k" and len(value) > 0 and isinstance(value[0], dict)
            child_ctx = replace(ctx, in_keyframe=is_keyframe_list) if is_keyframe_list else ctx
            toks.append(v.special("BEGIN_LIST"))
            for item in value:
                toks += self._encode_value(item, key=key, ctx=child_ctx)
            toks.append(v.special("END_LIST"))
            return toks

        if value is None:
            return [v.special("NULL")]

        if isinstance(value, bool):
            return [v.enum("bool", value)]

        if isinstance(value, str):
            if key == "ty":
                return [v.enum("ty_shape", value)]
            if key == "n":
                return [v.enum("dash_type", value)]
            raise TokenizeError(f"unexpected free string under key {key!r}: {value!r}")

        if isinstance(value, (int, float)):
            if key == "ty":
                return [v.enum("ty_layer", int(value))]
            kind, fam = _classify_numeric(key, ctx)
            if kind == "enum":
                return [v.enum(fam, int(value))]
            return [v.numeric(fam, float(value))]

        raise TokenizeError(f"unsupported value type {type(value)} under key {key!r}")


class _StackDecoder:
    """Mirrors _encode_value's structural walk in reverse. Does not need the
    encode-side context (shape_ty / property_family / in_keyframe) because a
    numeric token's family — and thus how to decode it — is determined by
    which disjoint id-range it falls in, not by surrounding structure."""

    def __init__(self, vocab: Vocab):
        self.v = vocab

    def run(self, ids: list[int]) -> dict:
        v = self.v
        # stack of (container, pending_key_or_None); container is dict or
        # list. Empty stack means "no open frame" — the *first* BEGIN_DICT
        # becomes the eventual result, not a child placed into a parent.
        stack: list[tuple[Any, str | None]] = []
        result: dict = {}

        def place(value: Any) -> None:
            nonlocal result
            if not stack:
                return  # value before any BEGIN_DICT — malformed, drop
            container, pending_key = stack[-1]
            if isinstance(container, list):
                container.append(value)
            else:
                if pending_key is None:
                    return  # malformed: value with no pending key, drop
                container[pending_key] = value
                stack[-1] = (container, None)

        def close_frame() -> None:
            nonlocal result
            finished, _ = stack.pop()
            if stack:
                place(finished)
            elif isinstance(finished, dict):
                result = finished

        i = 0
        n = len(ids)
        while i < n:
            tok = ids[i]
            i += 1

            if tok == v.special("BEGIN_DICT"):
                stack.append(({}, None))
                continue
            if tok == v.special("END_DICT"):
                if stack:
                    close_frame()
                continue
            if tok == v.special("BEGIN_LIST"):
                stack.append(([], None))
                continue
            if tok == v.special("END_LIST"):
                if stack:
                    close_frame()
                continue
            if tok == v.special("NULL"):
                place(None)
                continue
            if tok == v.special("PAD"):
                continue

            key_name = v.key_name(tok)
            if key_name is not None:
                if stack:
                    container, _ = stack[-1]
                    if isinstance(container, dict):
                        stack[-1] = (container, key_name)
                continue

            if v.is_enum(tok):
                place(v.decode_enum(tok))
                continue

            family = v.is_numeric(tok)
            if family is not None:
                val = v.decode_numeric(tok)
                if family in ("count_int", "time_frame", "frame_rate"):
                    val = int(val)
                place(val)
                continue
            # unknown token id (e.g. stray text-subvocab token from a
            # generation that didn't cleanly separate stages) — skip it.

        # auto-close any still-open frames (truncated generation)
        while stack:
            close_frame()

        return result
