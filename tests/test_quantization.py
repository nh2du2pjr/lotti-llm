from lottiegpt.tokenizer.quantize import Quantizer


def test_quant_float_roundtrip_within_half_bin_width():
    q = Quantizer()
    lo, hi, bins = -1080.0, 1080.0, 256
    bin_width = (hi - lo) / bins
    for value in (-1000.0, -0.3, 0.0, 12.34, 999.9):
        tok = q.encode("rotation_deg", value)
        back = q.decode("rotation_deg", tok)
        assert abs(back - value) <= bin_width / 2 + 1e-6


def test_quant_float_clips_out_of_range_instead_of_raising():
    q = Quantizer()
    tok_low = q.encode("percent", -10_000.0)
    tok_high = q.encode("percent", 10_000.0)
    assert 0 <= tok_low < q.bins("percent")
    assert 0 <= tok_high < q.bins("percent")
    assert q.stats["percent"].clipped == 2


def test_exact_int_is_lossless_inside_range():
    q = Quantizer()
    for value in (-5, 0, 1, 42, 300):
        tok = q.encode("time_frame", value)
        assert q.decode("time_frame", tok) == float(value)


def test_exact_int_clips_outside_range():
    q = Quantizer()
    spec = q.families["time_frame"]
    tok = q.encode("time_frame", spec["hi"] + 1000)
    assert q.decode("time_frame", tok) == spec["hi"]
    assert q.stats["time_frame"].clipped == 1


def test_bins_reports_correct_family_size():
    q = Quantizer()
    assert q.bins("color_channel") == q.families["color_channel"]["bins"]
    spec = q.families["count_int"]
    assert q.bins("count_int") == int(spec["hi"]) - int(spec["lo"]) + 1
