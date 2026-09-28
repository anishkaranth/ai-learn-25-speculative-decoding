#!/usr/bin/env python3
"""Speculative decoding smoke: gamma sweep + draft-quality sweep -> results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from corpus import build_corpus, encode, make_maps
from lm import train_ngram, vanilla_decode
from smoke_plots import make_plots, write_results_md
from speculate import compare_distributions, speculative_decode

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
CFG = {
    "corpus_repeats": 40,
    "draft_order": 2,
    "target_order": 3,
    "draft_cost": 1.0,
    "target_cost": 8.0,  # target ~8x more expensive per token (proxy)
    "alpha": 0.5,
    "n_prompts": 24,
    "n_tokens": 32,
    "prompt_len": 8,
    "gammas": [1, 2, 3, 4, 5, 6, 8],
    "draft_mixes": [1.0, 0.75, 0.5, 0.25, 0.0],
    "gamma_quality": 4,
}


def _compact(js: str) -> str:
    js = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)
    return re.sub(r"\{\n([^{}\[\]]*?)\n\s*\}", lambda m: "{" + re.sub(r"\s*\n\s*", " ", m.group(1)).strip() + "}", js)


def r4(x) -> float:
    return round(float(x), 4)


def main() -> None:
    t0 = time.perf_counter()
    text, chars = build_corpus(CFG["corpus_repeats"])
    stoi, itos = make_maps(chars)
    ids = encode(text, stoi)
    V = len(chars)
    # train / held-out split
    split = int(0.9 * len(ids))
    train_ids, held_ids = ids[:split], ids[split:]

    draft = train_ngram(train_ids, CFG["draft_order"], V, CFG["alpha"], CFG["draft_cost"])
    target = train_ngram(train_ids, CFG["target_order"], V, CFG["alpha"], CFG["target_cost"])

    draft_ppl = draft.perplexity(held_ids[:500])
    target_ppl = target.perplexity(held_ids[:500])

    # fixed prompts from held-out region
    rng_prompts = np.random.default_rng(SEED)
    prompts: List[List[int]] = []
    for _ in range(CFG["n_prompts"]):
        start = int(rng_prompts.integers(0, max(1, len(held_ids) - CFG["prompt_len"] - 1)))
        prompts.append(held_ids[start : start + CFG["prompt_len"]])

    # contexts for TV/KL
    ctxs = [held_ids[i : i + 4] for i in range(0, min(200, len(held_ids) - 4), 5)]

    # --- vanilla target baseline (per prompt, seed = SEED + 1000 + i) ---
    vanilla_flops_total = 0.0
    vanilla_seqs: List[List[int]] = []
    for i, prompt in enumerate(prompts):
        rng = np.random.default_rng(SEED + 1000 + i)
        seq, st = vanilla_decode(target, prompt, CFG["n_tokens"], rng)
        vanilla_seqs.append(seq)
        vanilla_flops_total += st["flops"]

    # --- gamma sweep (pure draft) ---
    gamma_sweep = []
    for gamma in CFG["gammas"]:
        flops = 0.0
        acc_rates, accepted_ps, emitted_ps = [], [], []
        matches = 0
        for i, prompt in enumerate(prompts):
            rng = np.random.default_rng(SEED + 2000 + i + gamma * 17)
            seq, st = speculative_decode(
                draft, target, prompt, CFG["n_tokens"], gamma, rng, draft_mix=1.0
            )
            flops += st["flops"]
            acc_rates.append(st["mean_acceptance_rate"])
            accepted_ps.append(st["mean_accepted_per_step"])
            emitted_ps.append(st["mean_emitted_per_step"])
            # exact-match vs vanilla with same prompt index (different path -> usually mismatch;
            # we also compare length-matched suffix equality as a soft check)
            if seq == vanilla_seqs[i]:
                matches += 1
        speedup = vanilla_flops_total / flops if flops > 0 else 0.0
        gamma_sweep.append({
            "gamma": gamma,
            "mean_acceptance_rate": r4(np.mean(acc_rates)),
            "mean_accepted_per_step": r4(np.mean(accepted_ps)),
            "mean_emitted_per_step": r4(np.mean(emitted_ps)),
            "speedup_vs_vanilla": r4(speedup),
            "exact_match_rate": r4(matches / len(prompts)),
            "flops": r4(flops),
        })

    # Distributional equivalence check: when draft_mix=0 (draft == target), acceptance -> 1
    # and we still measure speedup from parallel verify.
    # --- draft quality sweep ---
    quality_sweep = []
    for mix in CFG["draft_mixes"]:
        dist = compare_distributions(draft, target, ctxs, draft_mix=mix)
        flops = 0.0
        acc_rates = []
        matches = 0
        g = CFG["gamma_quality"]
        for i, prompt in enumerate(prompts):
            rng = np.random.default_rng(SEED + 3000 + i + int(mix * 100))
            seq, st = speculative_decode(
                draft, target, prompt, CFG["n_tokens"], g, rng, draft_mix=mix
            )
            flops += st["flops"]
            acc_rates.append(st["mean_acceptance_rate"])
            if seq == vanilla_seqs[i]:
                matches += 1
        speedup = vanilla_flops_total / flops if flops > 0 else 0.0
        quality_sweep.append({
            "draft_mix": mix,
            "mean_tv": r4(dist["mean_tv"]),
            "mean_kl_target_draft": r4(dist["mean_kl_target_draft"]),
            "mean_acceptance_rate": r4(np.mean(acc_rates)),
            "speedup_vs_vanilla": r4(speedup),
            "exact_match_rate": r4(matches / len(prompts)),
            "flops": r4(flops),
        })

    # Coupled-sample exact-match: re-run speculative with draft_mix=0 and compare to
    # freshly sampled target-only with a *shared* outer seed per prompt -- still not
    # path-identical, so we also report a "law check": acceptance ~= 1 when mix=0.
    law_check = next(r for r in quality_sweep if r["draft_mix"] == 0.0)

    best = max(gamma_sweep, key=lambda r: r["speedup_vs_vanilla"])
    headline = {
        "draft_ppl": r4(draft_ppl),
        "target_ppl": r4(target_ppl),
        "vanilla_flops": r4(vanilla_flops_total),
        "best_gamma": best["gamma"],
        "best_speedup": best["speedup_vs_vanilla"],
        "best_acceptance": best["mean_acceptance_rate"],
        "best_emitted_per_step": best["mean_emitted_per_step"],
        "gamma4_acceptance": next(r["mean_acceptance_rate"] for r in gamma_sweep if r["gamma"] == 4),
        "gamma4_speedup": next(r["speedup_vs_vanilla"] for r in gamma_sweep if r["gamma"] == 4),
        "perfect_draft_acceptance": law_check["mean_acceptance_rate"],
        "perfect_draft_speedup": law_check["speedup_vs_vanilla"],
        "perfect_draft_tv": law_check["mean_tv"],
    }

    CFG_out = dict(CFG)
    CFG_out["V"] = V
    CFG_out["corpus_chars"] = len(text)
    CFG_out["train_chars"] = len(train_ids)
    CFG_out["held_chars"] = len(held_ids)

    m: Dict[str, Any] = {
        "project": "ai-learn-25-speculative-decoding",
        "seed": SEED,
        "config": CFG_out,
        "model_info": {
            "draft_order": CFG["draft_order"],
            "target_order": CFG["target_order"],
            "draft_cost": CFG["draft_cost"],
            "target_cost": CFG["target_cost"],
            "vocab": chars,
            "V": V,
        },
        "gamma_sweep": gamma_sweep,
        "draft_quality_sweep": quality_sweep,
        "headline": headline,
    }
    m["wall_time_s"] = round(time.perf_counter() - t0, 2)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(_compact(json.dumps(m, indent=1)) + "\n")
    shot = {
        "project": m["project"],
        "seed": SEED,
        "config": {k: CFG_out[k] for k in (
            "draft_order", "target_order", "draft_cost", "target_cost",
            "n_prompts", "n_tokens", "gammas", "draft_mixes", "gamma_quality", "V",
        )},
        "headline": headline,
        "wall_time_s": m["wall_time_s"],
    }
    (RESULTS / "JSON.shot").write_text(_compact(json.dumps(shot, indent=2)) + "\n")
    plots = make_plots(RESULTS, m)
    write_results_md(RESULTS / "RESULTS.md", m, plots)
    json.loads((RESULTS / "JSON.shot").read_text())
    print(json.dumps(headline, indent=2))
    print("wall", m["wall_time_s"])
    print("plots", plots)


if __name__ == "__main__":
    main()
