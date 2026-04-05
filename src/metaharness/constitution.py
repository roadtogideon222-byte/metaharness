"""
GEPA Constitution — Trading Principles Hierarchy

The constitutional framework for GEPA genome critique.
Inspired by Anthropic's Constitutional AI (2022) but adapted for
autonomous trading signal evolution.

Principle hierarchy:
  Tier 1 — HARD VETO: Any violation = genome rejected before backtest
  Tier 2 — PENALTY: Violations reduce effective score after backtest
  Tier 3 — OPTIMIZE: Objectives to maximize/minimize during eval

The constitution is explicit, human-approved, and versioned.
Every genome is critiqued against these principles before evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PrincipleTier(Enum):
    VETO = "veto"       # Tier 1: hard rejection
    PENALTY = "penalty"  # Tier 2: score reduction
    OPTIMIZE = "optimize"  # Tier 3: maximize/minimize


class OptimizationDirection(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class Principle:
    """
    A single constitutional principle.

    id: unique identifier, e.g. "TIER1_THESIS_PRESENT"
    tier: veto / penalty / optimize
    description: human-readable principle text
    prompt_fragment: used to build the LLM critique prompt
    metric_key: which GenomeResult metric this applies to (if applicable)
    threshold: numeric threshold for binary veto check (if applicable)
    weight: for penalty/optimize tiers — how much this affects the score
    direction: maximize or minimize (for optimize tier)
    """
    id: str
    tier: PrincipleTier
    description: str
    prompt_fragment: str
    metric_key: str | None = None
    threshold: float | None = None
    weight: float = 1.0
    direction: OptimizationDirection | None = None


# ─────────────────────────────────────────────────────────────────────────────
# THE CONSTITUTION — Approved by Gideon
# ─────────────────────────────────────────────────────────────────────────────

CONSTITUTION: list[Principle] = [

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 1 — HARD VETO: Reject before backtest
    # ═══════════════════════════════════════════════════════════════════════

    Principle(
        id="TIER1_NO_THESIS",
        tier=PrincipleTier.VETO,
        description="Every entry must have an explicit thesis. A genome that allows entry without a defined thesis violates capital protection principles.",
        prompt_fragment=(
            "CRITIQUE: Does this genome allow entries without an explicit, stored thesis? "
            "A valid thesis must include: token, direction, entry condition, and invalidation criteria. "
            "If the genome permits entries based solely on price action or bonding curve position without "
            "a defined signal thesis — flag as VIOLATION."
        ),
        threshold=1,  # must have at least one thesis signal defined
    ),

    Principle(
        id="TIER1_PUMPSCORE_BELOW_GATE",
        tier=PrincipleTier.VETO,
        description="No entry is attempted unless PumpScore >= 50. PumpScore < 50 means the signal quality indicators are insufficient.",
        prompt_fragment=(
            "CRITIQUE: Does this genome permit entries when PumpScore is below 50? "
            "The PumpScore composite gate must be checked before any buy signal fires. "
            "If the genome allows entries without PumpScore verification — flag as VIOLATION."
        ),
        threshold=50.0,
    ),

    Principle(
        id="TIER1_NO_STOP_LOSS",
        tier=PrincipleTier.VETO,
        description="Every open position must have a defined stop-loss condition. No position may be held indefinitely without an exit plan.",
        prompt_fragment=(
            "CRITIQUE: Does this genome define a stop-loss or maximum drawdown exit for every position? "
            "A genome that allows unlimited downside exposure without a defined exit violates "
            "capital preservation principles. If no stop-loss or circuit breaker is defined — flag as VIOLATION."
        ),
    ),

    Principle(
        id="TIER1_LEVERAGE_FORBIDDEN",
        tier=PrincipleTier.VETO,
        description="Leveraged positions are forbidden. Only spot positions may be taken.",
        prompt_fragment=(
            "CRITIQUE: Does this genome allow any form of leverage, borrowing, or derivative exposure? "
            "Spot positions only. Any mention of margin, perp, futures, or leverage multipliers — flag as VIOLATION."
        ),
    ),

    Principle(
        id="TIER1_UNVERIFIED_KOL_WALLET",
        tier=PrincipleTier.VETO,
        description="Only wallets with win rate >= 55% over >= 20 trades may be used as signal sources. Unverified wallets are noise, not signal.",
        prompt_fragment=(
            "CRITIQUE: Does this genome use any wallet as a KOL signal source without win rate validation? "
            "Only wallets with WR >= 55% over >= 20 observed trades count as valid signals. "
            "If the genome references unvalidated wallets — flag as VIOLATION."
        ),
        threshold=0.55,
        metric_key="wallet_wr_threshold",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 2 — PENALTY: Reduce effective score after backtest
    # ═══════════════════════════════════════════════════════════════════════

    Principle(
        id="TIER2_HIGH_DRAWDOWN",
        tier=PrincipleTier.PENALTY,
        description="Drawdown above 20% should severely reduce the effective genome score. Above 30% is near-disqualifying.",
        prompt_fragment=(
            "CRITIQUE: Does this genome's historical performance show drawdown exceeding 20%? "
            "Drawdown above 20% indicates the genome does not protect capital during adverse conditions. "
            "Score reduction: 0-10% over baseline = no penalty. 10-20% = minor penalty. "
            "20-30% = significant penalty. Above 30% = near-disqualifying."
        ),
        metric_key="max_drawdown",
        threshold=0.20,
        weight=3.0,
    ),

    Principle(
        id="TIER2_WEAK_WIN_RATE",
        tier=PrincipleTier.PENALTY,
        description="A genome with win rate below 52% is functioning but suboptimal. Below 50% is destructive.",
        prompt_fragment=(
            "CRITIQUE: Does this genome's backtest show win rate below 52%? "
            "Win rate below 52% means the genome is generating as many losing trades as winning ones — "
            "it may have positive expectation due to size asymmetry, but it is not clean edge. "
            "Flag any win rate below 50% as severe penalty."
        ),
        metric_key="win_rate",
        threshold=0.52,
        weight=2.0,
    ),

    Principle(
        id="TIER2_SINGLE_REGIME",
        tier=PrincipleTier.PENALTY,
        description="A genome that only performs well in one market regime (e.g., only pumps, only crashes) is fragile.",
        prompt_fragment=(
            "CRITIQUE: Does this genome appear regime-specific — winning only during high-volatility pumps "
            "or only during calm accumulation? A robust genome should have positive Sharpe across multiple "
            "regimes. Single-regime genomes are penalized unless the regime specialization is explicitly "
            "documented and bounded."
        ),
        metric_key="regime_spread",
        weight=1.5,
    ),

    Principle(
        id="TIER2_STALE_SIGNAL_AGE",
        tier=PrincipleTier.PENALTY,
        description="Signals older than their maximum useful age should not contribute to entry decisions.",
        prompt_fragment=(
            "CRITIQUE: Does this genome use signal age expiry rules? "
            "Valid signal ages: THESIS < 21 days, watchlist entry < 4 hours, "
            "KOL signal < 8 hours, bonding opportunity < 2 hours. "
            "If the genome uses expired signals in entry decisions — flag as VIOLATION with penalty."
        ),
        metric_key="stale_signal_pct",
        weight=1.0,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 3 — OPTIMIZE: Maximize or minimize during evaluation
    # ═══════════════════════════════════════════════════════════════════════

    Principle(
        id="TIER3_SHARPE_RATIO",
        tier=PrincipleTier.OPTIMIZE,
        description="Sharpe ratio is the primary optimization target. Higher is better.",
        prompt_fragment=(
            "EVALUATE: What is the Sharpe ratio of this genome over the evaluation period? "
            "Sharpe = (mean return - risk_free rate) / std(return). "
            "This is the primary metric. Report it precisely."
        ),
        metric_key="sharpe",
        direction=OptimizationDirection.MAXIMIZE,
        weight=5.0,
    ),

    Principle(
        id="TIER3_ALPHA",
        tier=PrincipleTier.OPTIMIZE,
        description="Alpha (return above market benchmark) is the secondary optimization target.",
        prompt_fragment=(
            "EVALUATE: What is the alpha of this genome vs. a buy-and-hold SOL baseline? "
            "Alpha = genome_return - market_return. Positive alpha means the genome "
            "adds value beyond passive exposure."
        ),
        metric_key="alpha",
        direction=OptimizationDirection.MAXIMIZE,
        weight=3.0,
    ),

    Principle(
        id="TIER3_TRADE_FREQUENCY",
        tier=PrincipleTier.OPTIMIZE,
        description="Over-trading destroys edge through slippage and fees. A healthy genome trades selectively.",
        prompt_fragment=(
            "EVALUATE: What is the average number of trades per day? "
            "More than 5 trades/day on a single token suggests over-trading. "
            "Fewer than 0.5 trades/day suggests the genome is too conservative. "
            "Flag both extremes."
        ),
        metric_key="trades_per_day",
        direction=OptimizationDirection.MAXIMIZE,  # balance — not too many, not too few
        weight=1.0,
    ),

    Principle(
        id="TIER3_CRITIQUE_SELF_AWARENESS",
        tier=PrincipleTier.OPTIMIZE,
        description="The genome should include explicit conditions under which it invalidates its own thesis.",
        prompt_fragment=(
            "EVALUATE: Does this genome include thesis invalidation conditions? "
            "A self-aware genome defines exit conditions before entry — not after. "
            "Genomes with explicit invalidation criteria score higher than those without."
        ),
        metric_key="has_invalidation_logic",
        direction=OptimizationDirection.MAXIMIZE,
        weight=2.0,
    ),
]


def get_principle_by_id(principle_id: str) -> Principle | None:
    for p in CONSTITUTION:
        if p.id == principle_id:
            return p
    return None


def get_principles_by_tier(tier: PrincipleTier) -> list[Principle]:
    return [p for p in CONSTITUTION if p.tier == tier]


# ─────────────────────────────────────────────────────────────────────────────
# Constitutional prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_critique_prompt(genome_source: str, prior_violations: list[str] | None = None) -> str:
    """
    Build the full critique prompt from the constitution.
    This is passed to the LLM as the critique instruction.
    """
    veto_principles = get_principles_by_tier(PrincipleTier.VETO)
    penalty_principles = get_principles_by_tier(PrincipleTier.PENALTY)
    optimize_principles = get_principles_by_tier(PrincipleTier.OPTIMIZE)

    prior_note = ""
    if prior_violations:
        prior_note = (
            f"\nPRIOR VIOLATIONS from previous critique iterations: "
            + ", ".join(prior_violations)
            + "\nThese have been addressed in the current genome version. "
            "Re-check — do any remain?"
        )

    prompt = f"""You are GEPA-CR相反IQUE, a constitutional trading genome auditor.

Your task: Critique the following genome against the GEPA Constitution.
Respond with a JSON object only — no narration, no preamble.

{prior_note}

━━━ GENOME TO CRITIQUE ━━━
```python
{genome_source[:4000]}
```
━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ TIER 1 — HARD VETO PRINCIPLES ━━━
(Any violation = REJECT immediately, do not backtest)
"""
    for p in veto_principles:
        prompt += f"\n[{p.id}] {p.description}\n  → {p.prompt_fragment}"

    prompt += """

━━━ TIER 2 — PENALTY PRINCIPLES ━━━
(Flag violations; apply score reduction after backtest)
"""
    for p in penalty_principles:
        prompt += f"\n[{p.id}] {p.description}\n  → {p.prompt_fragment}"

    prompt += """

━━━ TIER 3 — OPTIMIZE OBJECTIVES ━━━
(Evaluate these metrics; report values)
"""
    for p in optimize_principles:
        prompt += f"\n[{p.id}] {p.description}\n  → {p.prompt_fragment}"

    prompt += """

━━━ YOUR RESPONSE FORMAT ━━━
Return a single JSON object with this structure:

{
  "tier1_violations": ["PRINCIPLE_ID", ...],   // list of veto principles violated
  "tier2_flags": ["PRINCIPLE_ID", ...],          // list of penalty principles flagged
  "tier3_metrics": {                              // metric_key: observed_value
    "sharpe": 1.42,
    "win_rate": 0.58,
    ...
  },
  "revision_needed": true/false,                  // did you modify the genome?
  "revised_genome": "...",                        // only if revision_needed=true
  "revision_rationale": "...",                    // why you revised
  "critique_notes": "..."                         // brief note on overall genome quality
}

JSON ONLY. No markdown fences. No explanation outside the JSON."""
    return prompt
