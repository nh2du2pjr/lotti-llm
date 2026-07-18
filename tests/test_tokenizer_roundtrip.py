import json
from pathlib import Path

import pytest

from lottiegpt.data.download import load_jsonl
from lottiegpt.tokenizer.lottie_tokenizer import LottieTokenizer

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "sample_lottie.jsonl"


def _normalize(obj, is_layer=False):
    """Mirror the tokenizer's own key-stripping/forcing (see schema.DROP_KEYS
    and LottieTokenizer._apply_layer_defaults) so comparisons check what the
    tokenizer *intends* to preserve, not incidental cosmetic fields."""
    from lottiegpt.tokenizer import schema

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


def assert_roundtrips(lottie_json: dict, tol: float = 10.0) -> None:
    """`tol` bounds acceptable post-quantization drift, not exactness — the
    coarsest quant_float families (e.g. rotation_deg, position) have bin
    widths of several units, so an exact 0 can legitimately decode a few
    units away from 0. This is a plausibility check, not a precision
    benchmark; see scripts/round_trip_check.py for per-family error stats."""
    tok = LottieTokenizer()
    ids = tok.encode(lottie_json)
    decoded = tok.decode(ids)

    expected_layers = [_normalize(layer, is_layer=True) for layer in lottie_json["layers"]]
    assert decoded["fr"] == lottie_json["fr"]
    assert decoded["ip"] == lottie_json["ip"]
    assert decoded["op"] == lottie_json["op"]
    assert len(decoded["layers"]) == len(expected_layers)
    _assert_close(expected_layers, decoded["layers"], tol)


def _assert_close(expected, actual, tol):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(expected.keys()) == set(actual.keys()), (expected.keys(), actual.keys())
        for k in expected:
            _assert_close(expected[k], actual[k], tol)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(expected) == len(actual)
        for e, a in zip(expected, actual):
            _assert_close(e, a, tol)
    elif isinstance(expected, bool):
        assert expected == actual
    elif isinstance(expected, (int, float)):
        assert abs(expected - actual) <= tol, (expected, actual)
    else:
        assert expected == actual


BASE = {"fr": 30, "ip": 0, "op": 60, "w": 512, "h": 512, "ddd": 0, "assets": []}


def _anim(a, k):
    return {"a": a, "k": k, "ix": 1}


def test_static_rect_fill_transform_roundtrips():
    doc = {
        **BASE,
        "layers": [{
            "ind": 1, "ty": 4, "sr": 1,
            "ip": 0, "op": 60, "st": 0, "bm": 0, "ao": 0, "ddd": 0,
            "ks": {
                "o": _anim(0, 100), "r": _anim(0, 0),
                "p": _anim(0, [256.0, 256.0, 0.0]),
                "a": _anim(0, [0.0, 0.0, 0.0]),
                "s": _anim(0, [100.0, 100.0, 100.0]),
                "sk": _anim(0, 0), "sa": _anim(0, 0),
            },
            "shapes": [{
                "ty": "gr",
                "it": [
                    {"ty": "rc", "p": _anim(0, [0.0, 0.0]), "s": _anim(0, [80.0, 40.0]), "r": _anim(0, 4.0)},
                    {"ty": "fl", "c": _anim(0, [1.0, 0.0, 0.0, 1.0]), "o": _anim(0, 100)},
                    {"ty": "tr", "p": _anim(0, [0.0, 0.0]), "a": _anim(0, [0.0, 0.0]),
                     "s": _anim(0, [100.0, 100.0]), "r": _anim(0, 0), "o": _anim(0, 100),
                     "sk": _anim(0, 0), "sa": _anim(0, 0)},
                ],
            }],
        }],
    }
    assert_roundtrips(doc)


def test_animated_position_and_opacity_keyframes_roundtrip():
    doc = {
        **BASE,
        "layers": [{
            "ind": 1, "ty": 3, "sr": 1, "ip": 0, "op": 60, "st": 0, "bm": 0, "ao": 0, "ddd": 0,
            "ks": {
                "o": {"a": 1, "k": [
                    {"i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}, "t": 0, "s": [0]},
                    {"t": 30, "s": [100]},
                ]},
                "r": _anim(0, 0),
                "p": {"a": 1, "k": [
                    {"i": {"x": 0.667, "y": 1}, "o": {"x": 0.333, "y": 0}, "t": 0,
                     "s": [0.0, 0.0, 0.0], "to": [10.0, 0.0, 0.0], "ti": [-10.0, 0.0, 0.0]},
                    {"t": 30, "s": [100.0, 50.0, 0.0]},
                ]},
                "a": _anim(0, [0.0, 0.0, 0.0]),
                "s": _anim(0, [100.0, 100.0, 100.0]),
                "sk": _anim(0, 0), "sa": _anim(0, 0),
            },
            "shapes": [],
        }],
    }
    assert_roundtrips(doc)


def test_path_shape_with_bezier_and_stroke_dash_roundtrips():
    doc = {
        **BASE,
        "layers": [{
            "ind": 1, "ty": 4, "sr": 1, "ip": 0, "op": 60, "st": 0, "bm": 0, "ao": 0, "ddd": 0,
            "ks": {
                "o": _anim(0, 100), "r": _anim(0, 0), "p": _anim(0, [0.0, 0.0, 0.0]),
                "a": _anim(0, [0.0, 0.0, 0.0]), "s": _anim(0, [100.0, 100.0, 100.0]),
                "sk": _anim(0, 0), "sa": _anim(0, 0),
            },
            "shapes": [{
                "ty": "gr",
                "it": [
                    {"ty": "sh", "ks": _anim(0, {
                        "i": [[0.0, -10.0], [10.0, 0.0]],
                        "o": [[0.0, 10.0], [-10.0, 0.0]],
                        "v": [[50.0, 0.0], [0.0, 50.0]],
                        "c": True,
                    })},
                    {
                        "ty": "st", "c": _anim(0, [0.0, 0.0, 0.0, 1.0]), "o": _anim(0, 100),
                        "w": _anim(0, 12.5), "lc": 2, "lj": 2, "ml": 4, "bm": 0,
                        "d": [
                            {"n": "d", "v": _anim(0, 8.0)},
                            {"n": "g", "v": _anim(0, 4.0)},
                        ],
                    },
                    {"ty": "tr", "p": _anim(0, [0.0, 0.0]), "a": _anim(0, [0.0, 0.0]),
                     "s": _anim(0, [100.0, 100.0]), "r": _anim(0, 0), "o": _anim(0, 100),
                     "sk": _anim(0, 0), "sa": _anim(0, 0)},
                ],
            }],
        }],
    }
    assert_roundtrips(doc)


@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="fixture corpus not fetched (run lottiegpt.data.download)")
def test_real_corpus_sample_round_trips_structurally():
    """Bulk check against real downloaded animations — exit criterion for M1.
    Only checks structural exactness (see scripts/round_trip_check.py for the
    full report including numeric error and per-family clip rates)."""
    records = load_jsonl(FIXTURE_PATH)
    tok = LottieTokenizer()
    failures = []
    for i, rec in enumerate(records):
        ids = tok.encode(rec)
        decoded = tok.decode(ids)
        expected_layers = [_normalize(layer, is_layer=True) for layer in rec["layers"]]
        try:
            _assert_close(expected_layers, decoded["layers"], tol=float("inf"))
        except AssertionError as e:
            failures.append((i, str(e)))
    assert not failures, f"{len(failures)}/{len(records)} animations failed structural round-trip: {failures[:5]}"
