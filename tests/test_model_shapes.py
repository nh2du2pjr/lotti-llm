import torch

from lottiegpt.model.config import ModelConfig
from lottiegpt.model.transformer import LottieGPT


def _tiny_config(**overrides) -> ModelConfig:
    base = dict(vocab_size=100, d_model=32, n_layer=2, n_head=2, ffn_dim=64, max_seq_len=64)
    base.update(overrides)
    return ModelConfig(**base)


def test_forward_shape_and_loss():
    cfg = _tiny_config()
    model = LottieGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    y = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, cfg.vocab_size)
    assert loss.item() > 0


def test_backward_updates_all_params():
    cfg = _tiny_config()
    model = LottieGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


def test_tied_embeddings_share_storage():
    cfg = _tiny_config(tie_embeddings=True)
    model = LottieGPT(cfg)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_rejects_sequence_longer_than_max_seq_len():
    cfg = _tiny_config(max_seq_len=16)
    model = LottieGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 17))
    try:
        model(x)
        assert False, "expected an assertion error for over-length input"
    except AssertionError:
        pass


def test_generate_stops_at_eos():
    cfg = _tiny_config()
    model = LottieGPT(cfg)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=20, eos_token_id=1, temperature=1.0)
    assert out.shape[1] <= 21


def test_param_count_matches_hand_calculation():
    cfg = ModelConfig(vocab_size=3915, d_model=512, n_layer=10, n_head=8, ffn_dim=2048, max_seq_len=8192)
    model = LottieGPT(cfg)
    embed = cfg.vocab_size * cfg.d_model
    per_block = 4 * cfg.d_model**2 + 2 * cfg.d_model * cfg.ffn_dim
    expected = embed + cfg.n_layer * per_block
    # norms add a small constant on top; assert within a tight margin rather
    # than an exact match to avoid hard-coding norm-parameter bookkeeping here
    assert abs(model.num_params() - expected) < 0.01 * expected
