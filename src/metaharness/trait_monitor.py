"""
GEPA Trait Monitor — Layer 4 of the Anthropic agent stack

Tracks latent behavioral traits across GEPA generations:
  - Thesis drift: is the thesis becoming vaguer?
  - Consensus sycophancy: mirroring market consensus vs. predicting independently?
  - Confidence calibration: are confidence scores systematically wrong?
  - Edge decay rate: how fast does signal quality degrade between evals?
  - Signal proliferation: is the genome adding unnecessary complexity?
  - Self-correction burden: how often does critique need to revise?

Trait monitoring is the bridge between Layer 3 (validation) and Layer 5
(interpretability). It tells us WHEN to intervene, not just that something failed.

Inspired by Anthropic's persona vectors work, but implemented as lightweight
LLM-based periodic assessment rather than activation-space steering — appropriate
for the filesystem-based MetaHarness architecture.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .critique import CritiqueLLMConfig


# ─────────────────────────────────────────────────────────────────────────────
# Trait definitions
# ─────────────────────────────────────────────────────────────────────────────

class TraitId(Enum):
    THESIS_DRIFT = "thesis_drift"
    CONSENSUS_SYCOPHANCY = "consensus_sycophancy"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    EDGE_DECAY_RATE = "edge_decay_rate"
    SIGNAL_COMPLEXITY = "signal_complexity"
    SELF_CORRECTION_BURDEN = "self_correction_burden"


TRAIT_META: dict[TraitId, dict[str, Any]] = {
    TraitId.THESIS_DRIFT: {
        "description": "Measures how vague or generic the trading thesis has become over generations. "
                       "High drift = thesis is using boilerplate language, lacks specificity.",
        "score_range": (0.0, 1.0),  # 0 = perfectly specific, 1 = completely vague
        "threshold": 0.60,           # above this = intervention recommended
        "weight": 2.0,               # how much this matters for overall trait score
    },
    TraitId.CONSENSUS_SYCOPHANCY: {
        "description": "Measures whether the genome is mirroring recent market consensus "
                       "(buy the pump, follow the KOL) rather than generating independent signals. "
                       "High sycophancy = genome is a trailing indicator, not a leading one.",
        "score_range": (0.0, 1.0),  # 0 = fully independent, 1 = pure consensus follower
        "threshold": 0.65,
        "weight": 2.5,
    },
    TraitId.CONFIDENCE_CALIBRATION: {
        "description": "Measures the gap between the genome's stated confidence "
                       "and its actual historical accuracy. "
                       "Positive = overconfident. Negative = underconfident.",
        "score_range": (-1.0, 1.0),  # 0 = perfectly calibrated
        "threshold": 0.30,           # absolute value above this = miscalibrated
        "weight": 2.0,
    },
    TraitId.EDGE_DECAY_RATE: {
        "description": "Rate at which the genome's edge (Sharpe, win rate) degrades "
                       "per generation when deployed. "
                       "High decay = genome overfits to historical conditions.",
        "score_range": (0.0, 1.0),  # 0 = no decay, 1 = complete decay in one generation
        "threshold": 0.20,           # per-generation decay above 20% = fragile
        "weight": 3.0,
    },
    TraitId.SIGNAL_COMPLEXITY: {
        "description": "Measures unnecessary signal proliferation — is the genome "
                       "adding signals that don't independently improve backtest? "
                       "High complexity = risk of overfitting to noise.",
        "score_range": (0.0, 1.0),  # 0 = minimal necessary signals, 1 = maximum complexity
        "threshold": 0.70,
        "weight": 1.5,
    },
    TraitId.SELF_CORRECTION_BURDEN: {
        "description": "How often does the LLM critique engine need to revise this genome? "
                       "High burden = genome consistently fails initial critique = structural weakness.",
        "score_range": (0.0, 1.0),  # 0 = passes critique first try, 1 = always needs revision
        "threshold": 0.50,
        "weight": 1.0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Trait report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraitScore:
    trait_id: str
    score: float           # raw trait value
    threshold: float
    crossed: bool           # True if intervention recommended
    delta_from_baseline: float = 0.0  # change since last snapshot
    raw_note: str = ""     # LLM's reasoning for this score
    evidence: list[str] = field(default_factory=list)  # specific examples from genome


@dataclass
class TraitReport:
    """
    Complete trait assessment for one genome at one point in time.

    overall_trait_score: weighted average across all traits (0-1)
    intervention_recommended: True if any trait crossed threshold
    traits: per-trait breakdown
    most_drifted_trait: which trait has degraded most since last snapshot
    genome_hash: short hash of genome source for comparison
    generation: which GEPA generation this report corresponds to
    assessed_at: ISO timestamp
    raw_llm_response: full LLM output for auditing
    """
    genome_hash: str
    generation: int
    overall_trait_score: float
    intervention_recommended: bool
    traits: list[TraitScore]
    most_drifted_trait: str | None = None
    most_drifted_delta: float = 0.0
    assessed_at: str = ""
    raw_llm_response: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_hash": self.genome_hash,
            "generation": self.generation,
            "overall_trait_score": self.overall_trait_score,
            "intervention_recommended": self.intervention_recommended,
            "most_drifted_trait": self.most_drifted_trait,
            "most_drifted_delta": self.most_drifted_delta,
            "assessed_at": self.assessed_at,
            "recommendations": self.recommendations,
            "traits": [
                {
                    "trait_id": t.trait_id,
                    "score": t.score,
                    "threshold": t.threshold,
                    "crossed": t.crossed,
                    "delta_from_baseline": t.delta_from_baseline,
                    "raw_note": t.raw_note,
                    "evidence": t.evidence,
                }
                for t in self.traits
            ],
            "raw_llm_response": self.raw_llm_response,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot history
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraitSnapshot:
    """
    A historical record of a trait assessment.
    Used to compute delta vs. prior snapshot.
    """
    generation: int
    genome_hash: str
    trait_scores: dict[str, float]  # trait_id → score
    overall_score: float
    assessed_at: str


class TraitHistory:
    """
    Stores trait snapshots across generations.
    Provides delta computation and trend analysis.
    """
    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path
        self.snapshots: list[TraitSnapshot] = []
        if storage_path and storage_path.exists():
            self._load()

    def add(self, report: TraitReport) -> None:
        snap = TraitSnapshot(
            generation=report.generation,
            genome_hash=report.genome_hash,
            trait_scores={t.trait_id: t.score for t in report.traits},
            overall_score=report.overall_trait_score,
            assessed_at=report.assessed_at,
        )
        self.snapshots.append(snap)
        if self.storage_path:
            self._save()

    def latest(self) -> TraitSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    def prior(self, n: int = 1) -> TraitSnapshot | None:
        if len(self.snapshots) < n:
            return None
        return self.snapshots[-(n + 1)]

    def delta_vs_latest(self, current: TraitReport) -> dict[str, float]:
        """
        Compute delta for each trait vs. the most recent snapshot.
        Returns trait_id → delta float.
        """
        last = self.latest()
        if last is None:
            return {}
        return {
            trait_id: current_score - last.trait_scores.get(trait_id, current_score)
            for trait_id, current_score in
            {t.trait_id: t.score for t in current.traits}.items()
        }

    def decay_rate(self, trait_id: str, window: int = 5) -> float | None:
        """
        Estimate the per-generation decay rate for a trait over a rolling window.
        Returns average delta per generation.
        """
        relevant = [s for s in self.snapshots[-window:] if trait_id in s.trait_scores]
        if len(relevant) < 2:
            return None
        deltas = []
        for i in range(1, len(relevant)):
            d = relevant[i].trait_scores[trait_id] - relevant[i - 1].trait_scores[trait_id]
            gen_delta = relevant[i].generation - relevant[i - 1].generation
            if gen_delta > 0:
                deltas.append(d / gen_delta)
        return sum(deltas) / len(deltas) if deltas else None

    def _save(self) -> None:
        if not self.storage_path:
            return
        data = [
            {
                "generation": s.generation,
                "genome_hash": s.genome_hash,
                "trait_scores": s.trait_scores,
                "overall_score": s.overall_score,
                "assessed_at": s.assessed_at,
            }
            for s in self.snapshots
        ]
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.snapshots = [
                TraitSnapshot(
                    generation=d["generation"],
                    genome_hash=d["genome_hash"],
                    trait_scores=d["trait_scores"],
                    overall_score=d["overall_score"],
                    assessed_at=d["assessed_at"],
                )
                for d in data
            ]
        except (json.JSONDecodeError, KeyError):
            self.snapshots = []


# ─────────────────────────────────────────────────────────────────────────────
# Trait assessment prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_trait_assessment_prompt(
    genome_source: str,
    prior_traits: dict[str, float] | None = None,
    backtest_summary: dict[str, Any] | None = None,
    generation: int = 0,
) -> str:
    """
    Build the LLM prompt for trait assessment.

    This is called periodically — not per-candidate — to assess the champion
    genome's behavioral traits. It uses the same LLM as the critique engine.
    """
    prior_block = ""
    if prior_traits:
        prior_block = (
            "\nPRIOR TRAIT ASSESSMENT (generation " + str(generation - 1) + "):\n"
            + "\n".join(f"  - {tid}: {score:.3f}" for tid, score in prior_traits.items())
            + "\nNote: Score changes vs. prior are computed automatically. "
            "Focus on qualitative assessment of the genome's current state."
        )

    backtest_block = ""
    if backtest_summary:
        backtest_block = (
            "\nBACKTEST SUMMARY (for calibration assessment):\n"
            + json.dumps(backtest_summary, indent=2)
        )

    return f"""You are GEPA-TRAIT, a behavioral trait assessor for trading signal genomes.

Assess the following genome against 6 latent traits. Respond with JSON only.

{prior_block}
{backtest_block}

━━━ GENOME TO ASSESS ━━━
```python
{genome_source[:3500]}
```
━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ TRAITS TO ASSESS ━━━

For each trait, assign a score and provide:
  - The score (see score range for each trait)
  - A 1-2 sentence note explaining the score
  - 1-2 specific pieces of evidence FROM THE GENOME SOURCE

**1. THESIS_DRIFT** (score: 0.0 = perfectly specific, 1.0 = completely vague)
  Look for: generic boilerplate language, absence of specific token/condition pairs,
  vague exit criteria, thesis that could apply to any token.
  Penalize: "buy when momentum is strong", "adjust for volatility" without specifics.
  Reward: "buy when PumpScore >= 60 AND bonding curve 40-70% AND Jizo wallet bought"

**2. CONSENSUS_SYCOPHANCY** (score: 0.0 = fully independent, 1.0 = pure consensus follower)
  Look for: genome primarily reacting to price, following KOL buys after the fact,
  no signal that precedes market moves, all conditions are post-hoc explanations.
  Reward: conditions that would have caught the move BEFORE it happened.

**3. CONFIDENCE_CALIBRATION** (score: -1.0 = severely underconfident, 0.0 = calibrated, +1.0 = severely overconfident)
  Use backtest summary if provided. Compare stated confidence to actual win rate / Sharpe.
  Look for: confidence scores that don't match historical accuracy, genome claiming
  high confidence on trades with poor actual performance.

**4. EDGE_DECAY_RATE** (score: 0.0 = no decay, 1.0 = complete decay in one generation)
  Look for: genome that fits historical data perfectly but has no self-correcting mechanism,
  no invalidation triggers, no regime adaptation. High edge decay = overfitted to backtest.
  Reward: explicit invalidation conditions, regime detection, adaptive sizing.

**5. SIGNAL_COMPLEXITY** (score: 0.0 = minimal necessary, 1.0 = maximum unnecessary complexity)
  Count independent signals: how many different conditions must all fire?
  Penalize: 5+ simultaneous conditions, signals that add marginal predictive value,
  "everything but the kitchen sink" approach.
  Reward: 1-3 high-quality signals with clear, independent contribution.

**6. SELF_CORRECTION_BURDEN** (score: 0.0 = passes first try, 1.0 = always needs revision)
  Based on how many Tier 1 constitutional violations the genome would have
  (review the genome for: missing thesis, no stop-loss, PumpScore not checked,
  unvalidated wallets, leverage).
  More violations = higher burden = structural weakness.

━━━ RESPONSE FORMAT ━━━

Return a single JSON object:
{{
  "traits": {{
    "thesis_drift": {{
      "score": 0.35,
      "note": "Thesis is moderately specific but uses vague exit language",
      "evidence": ["mentions 'adjust for volatility' without threshold", "exit criteria missing time component"]
    }},
    "consensus_sycophancy": {{ ... }},
    "confidence_calibration": {{ ... }},
    "edge_decay_rate": {{ ... }},
    "signal_complexity": {{ ... }},
    "self_correction_burden": {{ ... }}
  }},
  "overall_trait_score": 0.42,
  "intervention_recommended": true,
  "recommendations": [
    "Thesis needs specific exit criteria",
    "Reduce signal count from 6 to 3",
    "Add invalidation triggers"
  ],
  "most_concerning": "consensus_sycophancy"
}}

JSON ONLY. No markdown fences. No explanation outside the JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# Trait monitor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraitMonitorConfig:
    """Configuration for trait monitoring."""
    # Assess every N generations
    assessment_interval: int = 5
    # Also assess when champion genome changes
    assess_on_champion_change: bool = True
    # Store snapshots to filesystem
    store_snapshots: bool = True
    # Intervention threshold: overall_trait_score above this → flag
    intervention_threshold: float = 0.60
    # Minimum generations before first assessment
    min_generations_before_first: int = 3
    # LLM config (reuses critique LLM by default)
    llm: CritiqueLLMConfig = field(default_factory=CritiqueLLMConfig)


class TraitMonitor:
    """
    Monitors behavioral traits across GEPA generations.

    Usage:
        monitor = TraitMonitor(config, history_path=Path("trait_history.json"))

        # After each generation:
        report = monitor.assess(
            genome_source=genome_text,
            generation=current_gen,
            backtest_summary={"sharpe": 1.2, "win_rate": 0.58},
        )

        if monitor.should_intervene(report):
            # Trigger corrective action — revise genome, alert, etc.
    """

    def __init__(
        self,
        config: TraitMonitorConfig | None = None,
        history_path: Path | None = None,
    ):
        self.config = config or TraitMonitorConfig()
        self.history = TraitHistory(history_path)
        self._llm_client: Any = None

    def assess(
        self,
        genome_source: str,
        generation: int,
        backtest_summary: dict[str, Any] | None = None,
    ) -> TraitReport:
        """
        Run a full trait assessment on the genome.

        Determines whether this generation needs assessment based on:
        - assessment_interval (every N generations)
        - assess_on_champion_change
        - min_generations_before_first
        """
        # Determine if we should assess this generation
        if generation < self.config.min_generations_before_first:
            return self._empty_report(genome_source, generation, "too_early")

        should_skip, reason = self._should_skip_assessment(generation)
        if should_skip:
            return self._empty_report(genome_source, generation, reason)

        # Build prompt with prior traits if available
        prior_traits = None
        last = self.history.latest()
        if last:
            prior_traits = last.trait_scores

        prompt = build_trait_assessment_prompt(
            genome_source=genome_source,
            prior_traits=prior_traits,
            backtest_summary=backtest_summary,
            generation=generation,
        )

        raw_response = self._call_llm(prompt)
        report = self._parse_assessment(
            raw_response=raw_response,
            genome_source=genome_source,
            generation=generation,
        )

        # Compute deltas vs. prior snapshot
        if self.history.snapshots:
            deltas = self.history.delta_vs_latest(report)
            for trait in report.traits:
                trait.delta_from_baseline = deltas.get(trait.trait_id, 0.0)

            # Find most drifted trait
            if deltas:
                max_delta_trait = max(deltas, key=lambda k: abs(deltas[k]))
                report.most_drifted_trait = max_delta_trait
                report.most_drifted_delta = deltas[max_delta_trait]

        # Store snapshot
        if self.config.store_snapshots:
            self.history.add(report)
            if self.history.storage_path:
                self.history.storage_path.parent.mkdir(parents=True, exist_ok=True)
                self.history.add(report)  # re-save with new data

        return report

    def should_intervene(self, report: TraitReport) -> bool:
        """
        Determine if intervention is recommended based on trait report.

        Triggers if:
        - overall_trait_score exceeds intervention_threshold
        - Any single trait crosses its specific threshold
        - Most drifted trait has delta > 0.15 (rapid degradation)
        """
        if report.overall_trait_score > self.config.intervention_threshold:
            return True

        for trait in report.traits:
            if trait.crossed:
                return True
            if (
                trait.trait_id == report.most_drifted_trait
                and abs(report.most_drifted_delta) > 0.15
            ):
                return True

        return False

    def get_intervention_type(self, report: TraitReport) -> str:
        """
        Classify what kind of intervention is needed.
        Returns one of: 'revise_genome', 'reassess_constitution', 'alert_human', 'fine_tune'
        """
        # High self_correction_burden → structural revision needed
        sc = self._get_trait(report, TraitId.SELF_CORRECTION_BURDEN.value)
        if sc and sc.score > 0.60:
            return "revise_genome"

        # High thesis drift → constitution needs updating
        td = self._get_trait(report, TraitId.THESIS_DRIFT.value)
        if td and td.score > 0.70:
            return "reassess_constitution"

        # Rapid degradation in any trait → alert human
        if abs(report.most_drifted_delta) > 0.20:
            return "alert_human"

        # General degradation → fine-tune or revise
        if report.overall_trait_score > 0.50:
            return "fine_tune"

        return "monitor"

    def trend_summary(self, trait_id: str, window: int = 5) -> dict[str, Any]:
        """
        Get a trend summary for a specific trait over a rolling window.
        Returns: score trajectory, average, decay rate estimate.
        """
        last = self.history.latest()
        if last is None:
            return {"status": "no_data"}

        relevant = [s for s in self.history.snapshots[-window:] if trait_id in s.trait_scores]
        if len(relevant) < 2:
            return {"status": "insufficient_data", "n_snapshots": len(relevant)}

        scores = [s.trait_scores[trait_id] for s in relevant]
        decay_rate = self.history.decay_rate(trait_id, window)

        return {
            "trait_id": trait_id,
            "window": window,
            "n_snapshots": len(relevant),
            "scores": scores,
            "latest": scores[-1],
            "oldest": scores[0],
            "delta_total": scores[-1] - scores[0],
            "average": sum(scores) / len(scores),
            "decay_rate_per_generation": decay_rate,
            "trend": "worsening" if scores[-1] > scores[0] else "improving",
        }

    def _should_skip_assessment(self, generation: int) -> tuple[bool, str]:
        """Determine if assessment should be skipped for this generation."""
        if generation < self.config.min_generations_before_first:
            return True, "min_generations_not_reached"
        if generation % self.config.assessment_interval != 0:
            return True, "not_on_interval"
        return False, ""

    def _get_trait(self, report: TraitReport, trait_id: str) -> TraitScore | None:
        for t in report.traits:
            if t.trait_id == trait_id:
                return t
        return None

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM for trait assessment."""
        from anthropic import Anthropic

        try:
            client = Anthropic(api_key=self.config.llm.api_key or "")
        except Exception:
            import openai
            client = openai.OpenAI(
                api_key=self.config.llm.api_key or "",
                base_url=self.config.llm.base_url or "https://api.anthropic.com",
            )
            response = client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system", "content": "You are GEPA-TRAIT, a behavioral trait assessor. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.3,
            )
            return response.choices[0].message.content

        response = client.messages.create(
            model=self.config.llm.model,
            max_tokens=2048,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _parse_assessment(
        self,
        raw_response: str,
        genome_source: str,
        generation: int,
    ) -> TraitReport:
        """Parse the LLM JSON response into a TraitReport."""
        import hashlib

        genome_hash = hashlib.sha256(genome_source.encode()).hexdigest()[:8]

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = {}

        traits_raw = parsed.get("traits", {})
        traits: list[TraitScore] = []

        for trait_id_str, meta in TRAIT_META.items():
            trait_key = trait_id_str.value
            trait_data = traits_raw.get(trait_key, {})

            score = float(trait_data.get("score", 0.5))
            threshold = meta["threshold"]

            # Determine if crossed
            if trait_id_str == TraitId.CONFIDENCE_CALIBRATION:
                crossed = abs(score) > threshold
            else:
                crossed = score > threshold

            traits.append(TraitScore(
                trait_id=trait_key,
                score=score,
                threshold=threshold,
                crossed=crossed,
                raw_note=trait_data.get("note", ""),
                evidence=trait_data.get("evidence", []),
            ))

        overall = parsed.get("overall_trait_score", 0.5)
        intervention = parsed.get("intervention_recommended", any(t.crossed for t in traits))

        return TraitReport(
            genome_hash=genome_hash,
            generation=generation,
            overall_trait_score=float(overall),
            intervention_recommended=bool(intervention),
            traits=traits,
            assessed_at=datetime.now(UTC).isoformat(),
            raw_llm_response=raw_response,
            recommendations=parsed.get("recommendations", []),
        )

    def _empty_report(
        self,
        genome_source: str,
        generation: int,
        reason: str,
    ) -> TraitReport:
        """Return an empty report when assessment is skipped."""
        import hashlib
        genome_hash = hashlib.sha256(genome_source.encode()).hexdigest()[:8]
        return TraitReport(
            genome_hash=genome_hash,
            generation=generation,
            overall_trait_score=0.0,
            intervention_recommended=False,
            traits=[],
            assessed_at=datetime.now(UTC).isoformat(),
            recommendations=[f"Assessment skipped: {reason}"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────

def format_trait_report(report: TraitReport) -> str:
    """Human-readable trait report."""
    status = "INTERVENTION NEEDED" if report.intervention_recommended else "CLEAR"
    lines = [
        f"[{status}] Trait Report — gen {report.generation} | hash={report.genome_hash}",
        f"Overall trait score: {report.overall_trait_score:.3f}",
        "",
    ]
    for t in report.traits:
        status_icon = "✗" if t.crossed else "✓"
        delta_str = f" (Δ={t.delta_from_baseline:+.3f})" if t.delta_from_baseline != 0.0 else ""
        lines.append(
            f"  {status_icon} {t.trait_id}: {t.score:.3f} "
            f"[threshold={t.threshold}]{delta_str}"
        )
        if t.raw_note:
            lines.append(f"    → {t.raw_note[:120]}")
        if t.evidence:
            for e in t.evidence[:1]:
                lines.append(f"    evidence: {e[:100]}")

    if report.most_drifted_trait:
        lines.append(
            f"\nMost drifted: {report.most_drifted_trait} "
            f"(Δ={report.most_drifted_delta:+.3f})"
        )

    if report.recommendations:
        lines.append("\nRecommendations:")
        for r in report.recommendations:
            lines.append(f"  • {r}")

    return "\n".join(lines)


def format_trend_summary(trend: dict[str, Any]) -> str:
    """Human-readable trend summary."""
    if trend.get("status"):
        return f"[{trend['status'].upper()}]"

    return (
        f"Trait: {trend['trait_id']}\n"
        f"  Window: {trend['window']} gens | Snapshots: {trend['n_snapshots']}\n"
        f"  Scores: {' → '.join(f'{s:.2f}' for s in trend['scores'])}\n"
        f"  Latest: {trend['latest']:.3f} | Oldest: {trend['oldest']:.3f}\n"
        f"  Delta total: {trend['delta_total']:+.3f}\n"
        f"  Trend: {trend['trend']}\n"
        f"  Decay rate/gen: {trend.get('decay_rate_per_generation', 'N/A')}"
    )
