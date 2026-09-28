"""Speculative decoding (draft-then-verify) a la Leviathan / Chen et al."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from lm import NGramLM, mix_probs, sample_from


def _residual(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Normalised max(0, q - p) -- distribution used after a rejection."""
    r = np.maximum(q - p, 0.0)
    s = r.sum()
    if s <= 0:
        return q.copy()  # degenerate: fall back to target
    return r / s


def speculative_step(
    draft: NGramLM,
    target: NGramLM,
    prefix: List[int],
    gamma: int,
    rng: np.random.Generator,
    draft_mix: float = 1.0,
) -> Tuple[List[int], Dict[str, float]]:
    """One draft-verify round.

    Returns newly appended tokens (length in 1..gamma+1) and step stats.
    draft_mix blends draft probs toward target (1=pure draft, 0=identical).
    """
    # --- draft proposes gamma tokens ---
    draft_toks: List[int] = []
    draft_ps: List[np.ndarray] = []
    ctx = list(prefix)
    draft_flops = 0.0
    for _ in range(gamma):
        p_d = draft.probs(ctx)
        p_t = target.probs(ctx)
        p = mix_probs(p_d, p_t, draft_mix)
        tok = sample_from(p, rng)
        draft_toks.append(tok)
        draft_ps.append(p)
        ctx.append(tok)
        draft_flops += draft.cost_per_token

    # --- target verifies in one parallel forward (cost ~ one target forward) ---
    # We score each of the gamma draft positions + prepare bonus sample.
    target_ps: List[np.ndarray] = []
    verify_ctx = list(prefix)
    for i in range(gamma):
        target_ps.append(target.probs(verify_ctx))
        verify_ctx.append(draft_toks[i])
    # bonus position (if all accepted)
    bonus_p = target.probs(verify_ctx)
    # Parallel verify: charged as ONE target forward (the whole point of speculation)
    target_flops = target.cost_per_token

    accepted: List[int] = []
    n_accepted = 0
    rejected = False
    for i in range(gamma):
        tok = draft_toks[i]
        p = draft_ps[i][tok]
        q = target_ps[i][tok]
        # accept with min(1, q/p)
        ratio = 1.0 if p <= 0 else min(1.0, float(q / p))
        if rng.random() < ratio:
            accepted.append(tok)
            n_accepted += 1
        else:
            # sample residual from target at this position
            res = _residual(target_ps[i], draft_ps[i])
            accepted.append(sample_from(res, rng))
            rejected = True
            break

    if not rejected:
        # all gamma accepted -> sample one more from target
        accepted.append(sample_from(bonus_p, rng))

    stats = {
        "n_accepted_prefix": float(n_accepted),
        "n_emitted": float(len(accepted)),
        "all_accepted": 0.0 if rejected else 1.0,
        "draft_flops": draft_flops,
        "target_flops": target_flops,
        "flops": draft_flops + target_flops,
        "acceptance_rate": float(n_accepted / gamma) if gamma else 0.0,
    }
    return accepted, stats


def speculative_decode(
    draft: NGramLM,
    target: NGramLM,
    prompt: List[int],
    n_tokens: int,
    gamma: int,
    rng: np.random.Generator,
    draft_mix: float = 1.0,
) -> Tuple[List[int], Dict[str, float]]:
    """Generate ~n_tokens new tokens via repeated speculative steps."""
    out = list(prompt)
    total = {
        "flops": 0.0,
        "draft_flops": 0.0,
        "target_flops": 0.0,
        "n_steps": 0.0,
        "n_accepted_prefix": 0.0,
        "n_emitted": 0.0,
        "all_accepted_steps": 0.0,
        "tokens_generated": 0.0,
        "target_forwards": 0.0,
    }
    generated = 0
    while generated < n_tokens:
        new_toks, st = speculative_step(draft, target, out, gamma, rng, draft_mix)
        # trim if we would overshoot
        need = n_tokens - generated
        if len(new_toks) > need:
            new_toks = new_toks[:need]
        out.extend(new_toks)
        generated += len(new_toks)
        total["flops"] += st["flops"]
        total["draft_flops"] += st["draft_flops"]
        total["target_flops"] += st["target_flops"]
        total["n_steps"] += 1
        total["n_accepted_prefix"] += st["n_accepted_prefix"]
        total["n_emitted"] += st["n_emitted"]
        total["all_accepted_steps"] += st["all_accepted"]
        total["target_forwards"] += 1.0  # one verify per step
    total["tokens_generated"] = float(generated)
    total["mean_acceptance_rate"] = (
        total["n_accepted_prefix"] / (total["n_steps"] * gamma) if total["n_steps"] and gamma else 0.0
    )
    total["mean_accepted_per_step"] = (
        total["n_accepted_prefix"] / total["n_steps"] if total["n_steps"] else 0.0
    )
    total["mean_emitted_per_step"] = (
        total["n_emitted"] / total["n_steps"] if total["n_steps"] else 0.0
    )
    return out, total


def compare_distributions(
    draft: NGramLM,
    target: NGramLM,
    contexts: List[List[int]],
    draft_mix: float = 1.0,
) -> Dict[str, float]:
    """How well draft matches target: mean TV distance and KL(target||draft)."""
    tvs, kls = [], []
    for ctx in contexts:
        p_d = draft.probs(ctx)
        p_t = target.probs(ctx)
        p = mix_probs(p_d, p_t, draft_mix)
        tvs.append(0.5 * float(np.abs(p - p_t).sum()))
        # KL(q || p)
        mask = p_t > 1e-12
        kls.append(float((p_t[mask] * (np.log(p_t[mask] + 1e-12) - np.log(p[mask] + 1e-12))).sum()))
    return {
        "mean_tv": float(np.mean(tvs)) if tvs else 0.0,
        "mean_kl_target_draft": float(np.mean(kls)) if kls else 0.0,
    }
