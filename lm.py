"""Tiny n-gram language models (count tables + Laplace) with FLOP cost proxies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NGramLM:
    """Character n-gram LM. order=2 → bigram (draft), order=3 → trigram (target)."""

    order: int
    V: int
    counts: np.ndarray  # shape (V,)*order  last axis = next token
    alpha: float = 0.5  # Laplace smoothing
    cost_per_token: float = 1.0  # FLOP proxy for one next-token lookup/sample

    def _ctx_key(self, ctx: List[int]) -> Tuple[int, ...]:
        need = self.order - 1
        if len(ctx) < need:
            # left-pad with 0 (first vocab id; space or 'a' — fine for toy)
            ctx = [0] * (need - len(ctx)) + list(ctx)
        return tuple(ctx[-need:])

    def probs(self, ctx: List[int]) -> np.ndarray:
        """P(next | context) with Laplace smoothing over the full vocab."""
        key = self._ctx_key(ctx)
        row = self.counts[key]
        return (row + self.alpha) / (row.sum() + self.alpha * self.V)

    def log_prob_seq(self, ids: List[int]) -> float:
        """Sum log P of tokens 1..end given preceding context (for perplexity)."""
        lp = 0.0
        for i in range(1, len(ids)):
            p = self.probs(ids[:i])
            lp += float(np.log(p[ids[i]] + 1e-12))
        return lp

    def perplexity(self, ids: List[int]) -> float:
        n = max(1, len(ids) - 1)
        return float(np.exp(-self.log_prob_seq(ids) / n))


def train_ngram(ids: List[int], order: int, V: int, alpha: float = 0.5,
                cost_per_token: float = 1.0) -> NGramLM:
    shape = (V,) * order
    counts = np.zeros(shape, dtype=np.float64)
    need = order - 1
    for i in range(need, len(ids)):
        idx = tuple(ids[i - need : i + 1])
        counts[idx] += 1.0
    return NGramLM(order=order, V=V, counts=counts, alpha=alpha, cost_per_token=cost_per_token)


def mix_probs(p_draft: np.ndarray, p_target: np.ndarray, mix: float) -> np.ndarray:
    """mix=1 → pure draft; mix=0 → identical to target (perfect draft)."""
    q = mix * p_draft + (1.0 - mix) * p_target
    return q / q.sum()


def sample_from(p: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(p), p=p))


def vanilla_decode(
    lm: NGramLM,
    prompt: List[int],
    n_tokens: int,
    rng: np.random.Generator,
) -> Tuple[List[int], Dict[str, float]]:
    """Autoregressive sample from a single LM; track FLOP proxy."""
    out = list(prompt)
    flops = 0.0
    for _ in range(n_tokens):
        p = lm.probs(out)
        out.append(sample_from(p, rng))
        flops += lm.cost_per_token
    return out, {"flops": flops, "target_forwards": float(n_tokens), "tokens_generated": float(n_tokens)}
