"""Curated Bodymovin/Lottie schema for the v1 tokenizer scope.

v1 deliberately supports a subset of the format, chosen by scanning ~250 real
animations from OmniLottie/MMLottie-2M (see scripts/analyze_schema.py-style
exploration in the M1 milestone): plain 2D vector-shape animations built from
solid/null/shape layers, with no external assets, layer effects, masks/track
mattes, or time-remapping. That subset already covers the large majority of
"icon/UI animation" style Lotties, which is the target domain — text layers,
precomps, images, effects and masks are out of scope for now (tracked as an M5
stretch item) rather than half-supported.

Two numeric encoding strategies are used (see quantize.py):
- "exact_int": integer fields where exactness matters structurally (layer
  index/parent references, counts) or where a modest fixed range covers the
  data losslessly (frame numbers).
- "quant_float": continuous fields where a bounded, binned approximation is
  acceptable (positions, percentages, angles, colors, ...).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Filtering: which animations are in scope at all
# ---------------------------------------------------------------------------

# Solid layers (ty=1) are excluded: their color is a "#RRGGBB" hex *string*
# (sw/sh/sc fields) rather than the rgba-float-array scheme used everywhere
# else in the schema, and they account for well under 1% of animations in
# the dataset — not worth a whole separate string-encoding path for v1.
SUPPORTED_LAYER_TYPES = {3, 4}  # null, shape
SUPPORTED_SHAPE_TYPES = {
    "gr",  # group
    "tr",  # transform
    "sh",  # path
    "fl",  # fill
    "st",  # stroke
    "mm",  # merge paths
    "el",  # ellipse
    "tm",  # trim path
    "gf",  # gradient fill
    "gs",  # gradient stroke
    "rc",  # rect
    "rp",  # repeater
    "sr",  # star / polygon
}
CANVAS_SIZE = 512  # every sample observed in the dataset uses 512x512

# top-level / layer-level keys whose presence means "reject this animation"
# (effects, masks, track mattes, time remap — semantically load-bearing
# features we don't attempt to model/reconstruct in v1)
REJECT_IF_PRESENT = {"ef", "hasMask", "masksProperties", "td", "tt", "tp"}
REJECT_IF_LAYER_HAS_KEY = {"tm"}  # layer-level time-remap (not shape-level "tm"!)

# ---------------------------------------------------------------------------
# Keys stripped before tokenization: cosmetic / expression-engine metadata
# that isn't needed to reconstruct a valid, visually-equivalent animation.
# ---------------------------------------------------------------------------
DROP_KEYS = {
    "nm",    # human-readable name
    "mn",    # match name (AE internal id)
    "ix",    # property index (expressions only)
    "cix",   # same, for shape groups
    "np",    # number-of-properties hint, recomputed structurally on decode
    "l",     # dimension hint on multi-dim properties, recovered from array length
    "ct",    # collapse-transformation flag, meaningful only for precomps (unsupported)
    "en",    # "enabled" flag, only meaningful on effects (rejected already)
    "inv", "mode",  # mask-only fields (masks are rejected already)
    "dr", "cm",     # rare AE export artifacts, no known rendering effect
}

# fields that are forced to a fixed constant on decode rather than tokenized
FORCED_CONSTANTS = {
    "v": "5.9.6",
    "ddd": 0,
    "ao": 0,
    "sr_layer": 1,  # layer-level stretch; shape-level "sr" (star) is unrelated
    "assets": [],
    "w": CANVAS_SIZE,
    "h": CANVAS_SIZE,
}

# ---------------------------------------------------------------------------
# Keys that survive into the token stream (need a KEY_<name> token). Grouped
# by where they occur just for documentation; the tokenizer itself doesn't
# care about the grouping, only about the flat set of names.
# ---------------------------------------------------------------------------
STRUCTURAL_KEYS = [
    # composition
    "fr", "ip", "op", "layers",
    # layer
    "ind", "ty", "parent", "ks", "shapes", "st", "bm",
    # transform (both layer "ks" and shape "tr")
    "o", "r", "p", "a", "s", "sk", "sa",
    # animated property wrapper + keyframe fields (bezier easing "i"/"o" are
    # themselves {"x":..,"y":..} dicts, x/y each scalar or per-component list)
    "k", "t", "i", "e", "to", "ti", "h", "x", "y",
    # shape: path
    "c", "v",
    # shape: fill / stroke ("ml2" is the animatable-property companion to
    # the static "ml" miter-limit fallback some exporters emit for old players)
    "w", "lc", "lj", "ml", "ml2", "d", "n",
    # shape: merge paths
    "mm",
    # shape: trim path
    "m",
    # shape: gradient stops container
    "g", "pt",
    # shape: star
    "or", "os", "ir", "is", "sy",
    # shape: repeater fade + its own nested transform sub-object
    "so", "eo", "tr",
    # visibility + generic container
    "hd", "it",
]
# de-dup while preserving order
STRUCTURAL_KEYS = list(dict.fromkeys(STRUCTURAL_KEYS))

# ---------------------------------------------------------------------------
# Closed-set enum values: (key_name) -> set of literal values that get their
# own ENUM token instead of going through numeric quantization.
# ---------------------------------------------------------------------------
ENUMS: dict[str, list] = {
    "ty_layer": [1, 3, 4],
    "ty_shape": sorted(SUPPORTED_SHAPE_TYPES),
    "bm": list(range(0, 17)),          # blend modes 0-16
    "lc": [1, 2, 3],                    # stroke line cap
    "lj": [1, 2, 3],                    # stroke line join
    "mm_mode": [1, 2, 3, 4, 5],         # merge-paths mode
    "sy": [1, 2],                       # star vs polygon
    "dash_type": ["d", "g", "o"],       # stroke dash-array entry: dash/gap/offset
    "gradient_type": [1, 2],            # linear vs radial
    "trim_multiple": [1, 2, 3],         # trim-path "multiple shapes" mode
    "repeater_composite": [1, 2],       # repeater composite order
    "bool": [True, False],
}

# ---------------------------------------------------------------------------
# Numeric family definitions: name -> (kind, range, bins)
# Ranges were sized from empirical percentiles on a 243-animation fixture
# sample (see data/fixtures/sample_lottie.jsonl, regenerated via
# `python -m lottiegpt.data.download`) and standard Bodymovin conventions.
# ---------------------------------------------------------------------------
NUMERIC_FAMILIES: dict[str, dict] = {
    "position":      {"kind": "quant_float", "lo": -3000.0, "hi": 3000.0, "bins": 384},
    "scale_pct":     {"kind": "quant_float", "lo": -200.0,  "hi": 500.0,  "bins": 192},
    "rotation_deg":  {"kind": "quant_float", "lo": -1080.0, "hi": 1080.0, "bins": 256},
    "percent":       {"kind": "quant_float", "lo": -50.0,   "hi": 150.0,  "bins": 128},
    "color_channel": {"kind": "quant_float", "lo": 0.0,     "hi": 1.0,    "bins": 64},
    "stroke_width":  {"kind": "quant_float", "lo": 0.0,     "hi": 200.0,  "bins": 96},
    "easing":        {"kind": "quant_float", "lo": -0.5,    "hi": 1.5,    "bins": 64},
    "misc_scalar":   {"kind": "quant_float", "lo": -50.0,   "hi": 1000.0, "bins": 192},
    "count_int":     {"kind": "exact_int",   "lo": -5,      "hi": 150},
    # p50=35, p90/p99 well under 500 in the calibration sample; a few
    # animations run longer (observed max ~3900) and simply clip — exact_int
    # cost scales linearly with range so this is a size/precision tradeoff,
    # not a correctness one (clipped-but-valid still decodes to a real frame).
    "time_frame":    {"kind": "exact_int",   "lo": -100,    "hi": 600},
    "frame_rate":    {"kind": "exact_int",   "lo": 8,       "hi": 120},
}
