"""Matplotlib SVG plots + RESULTS.md writer for speculative-decoding smoke."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "ai-learn-25", "svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False})
COLS = ["#e76f51", "#e9c46a", "#2a9d8f", "#126782", "#8338ec"]


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    gamma_runs = m["gamma_sweep"]
    gammas = [r["gamma"] for r in gamma_runs]
    acc = [r["mean_acceptance_rate"] for r in gamma_runs]
    speed = [r["speedup_vs_vanilla"] for r in gamma_runs]
    emitted = [r["mean_emitted_per_step"] for r in gamma_runs]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(gammas, acc, "o-", color=COLS[0], label="acceptance rate")
    ax.set_xlabel("draft length gamma")
    ax.set_ylabel("mean acceptance rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Acceptance rate vs draft length gamma")
    ax.grid(True, alpha=0.3)
    names.append(_save(fig, out / "acceptance_vs_gamma.svg"))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(gammas, speed, "s-", color=COLS[2], label="FLOP speedup")
    ax.axhline(1.0, color="#555", ls="--", lw=1, label="vanilla (=1)")
    ax.set_xlabel("draft length gamma")
    ax.set_ylabel("speedup (vanilla FLOPs / speculative FLOPs)")
    ax.set_title("FLOP speedup vs draft length gamma")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    names.append(_save(fig, out / "speedup_vs_gamma.svg"))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(gammas, emitted, "^-", color=COLS[3], label="tokens / step")
    ax.plot(gammas, [g + 1 for g in gammas], ":", color="#999", label="gamma+1 (ideal)")
    ax.set_xlabel("draft length gamma")
    ax.set_ylabel("mean tokens emitted per step")
    ax.set_title("Tokens per speculative step vs gamma")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    names.append(_save(fig, out / "tokens_per_step_vs_gamma.svg"))

    mix_runs = m["draft_quality_sweep"]
    mixes = [r["draft_mix"] for r in mix_runs]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(mixes, [r["mean_acceptance_rate"] for r in mix_runs], "o-", color=COLS[0], label="acceptance")
    ax.plot(mixes, [r["speedup_vs_vanilla"] for r in mix_runs], "s-", color=COLS[2], label="speedup")
    ax.set_xlabel("draft_mix (1=pure weak draft, 0=identical to target)")
    ax.set_ylabel("metric")
    ax.set_title(f"Draft quality sweep (gamma={m['config']['gamma_quality']})")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    names.append(_save(fig, out / "draft_quality_sweep.svg"))

    return names


def write_results_md(path: Path, m: Dict[str, Any], plots: List[str]) -> None:
    h, cfg = m["headline"], m["config"]
    grow = "\n".join(
        f"| {r['gamma']} | {r['mean_acceptance_rate']} | {r['mean_accepted_per_step']} | "
        f"{r['mean_emitted_per_step']} | {r['speedup_vs_vanilla']} | {r['exact_match_rate']} | "
        f"{r['flops']} |"
        for r in m["gamma_sweep"]
    )
    qrow = "\n".join(
        f"| {r['draft_mix']} | {r['mean_tv']} | {r['mean_acceptance_rate']} | "
        f"{r['speedup_vs_vanilla']} | {r['exact_match_rate']} |"
        for r in m["draft_quality_sweep"]
    )
    plot_links = "\n".join(f"- ![{p}]({p})" for p in plots)
    txt = f"""# Results: ai-learn-25-speculative-decoding

Real output of `python run_smoke.py` (seed {m['seed']}, CPU, {m['wall_time_s']} s wall time).

## Setup
- Corpus: synthetic character phrases ({cfg['corpus_chars']} chars, vocab {cfg['V']}), repeats={cfg['corpus_repeats']}.
- **Draft** LM: order-{cfg['draft_order']} n-gram, cost_per_token={cfg['draft_cost']} (cheap).
- **Target** LM: order-{cfg['target_order']} n-gram, cost_per_token={cfg['target_cost']} (expensive).
- Speculative decoding: Leviathan/Chen accept/reject with `min(1, q/p)`, residual resample on rejection,
  bonus target sample when all gamma draft tokens are accepted. Parallel verify charged as **one** target forward.
- Generation: {cfg['n_prompts']} prompts x {cfg['n_tokens']} new tokens; gamma sweep {cfg['gammas']};
  draft_mix sweep {cfg['draft_mixes']} at gamma={cfg['gamma_quality']}.
- Exact-match: fraction of speculative sequences that equal a target-only decode under a **coupled** RNG
  protocol (same prompt seed stream for the target samples used in residual/bonus positions when drafts
  are rejected -- see `run_smoke.py`). Reported against independent target-only baselines with matched seeds.

## gamma sweep (pure draft, draft_mix=1)
| gamma | accept rate | accepted/step | emitted/step | FLOP speedup | exact-match vs target-only | speculative FLOPs |
|---|---|---|---|---|---|---|
{grow}

## Draft-quality sweep (gamma={cfg['gamma_quality']})
| draft_mix | mean TV(draft,target) | accept rate | FLOP speedup | exact-match |
|---|---|---|---|---|
{qrow}

## Headline
- Best gamma by speedup: **gamma={h['best_gamma']}** -> speedup **{h['best_speedup']}x**, acceptance **{h['best_acceptance']}**,
  emitted/step **{h['best_emitted_per_step']}**.
- Draft / target perplexity on held-out slice: draft **{h['draft_ppl']}**, target **{h['target_ppl']}**.
- Vanilla target FLOPs for {cfg['n_tokens']} tokens x {cfg['n_prompts']} prompts: **{h['vanilla_flops']}**.

## Plots
{plot_links}

## Notes
- FLOP speedup is a **cost-model** proxy (draft cheap, one parallel target verify per step), not wall-clock on a GPU.
- Exact distributional equivalence holds for the accept/reject math; sequence exact-match vs an independent
  target-only run is intentionally < 1 because sampling paths differ even when laws match.
"""
    path.write_text(txt, encoding="utf-8")
