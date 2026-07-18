from __future__ import annotations

import torch

from lottiegpt.inference.validate_lottie import is_valid_lottie
from lottiegpt.model.transformer import LottieGPT
from lottiegpt.tokenizer.lottie_tokenizer import Vocab


@torch.no_grad()
def sample_animations(
    model: LottieGPT,
    vocab: Vocab,
    n_samples: int,
    max_new_tokens: int,
    device: str = "cpu",
    temperature: float = 0.9,
    top_k: int = 50,
) -> list[dict]:
    """Unconditional pretraining-stage sampling: start every sequence from
    BOS alone (there's no text prefix yet — that's the finetuning stage) and
    let the model generate a full animation. Returns the decoded dicts (not
    yet checked for validity — use is_valid_lottie separately)."""
    bos = vocab.special("BOS")
    idx = torch.full((n_samples, 1), bos, dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, eos_token_id=vocab.special("EOS"),
    )
    from lottiegpt.tokenizer.lottie_tokenizer import LottieTokenizer
    tok = LottieTokenizer(vocab)
    results = []
    for row in out.tolist():
        try:
            results.append(tok.decode(row))
        except Exception:
            results.append({})
    return results


def json_validity_rate(docs: list[dict]) -> float:
    if not docs:
        return 0.0
    return sum(1 for d in docs if is_valid_lottie(d)[0]) / len(docs)
