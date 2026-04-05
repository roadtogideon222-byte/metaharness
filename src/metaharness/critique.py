"""
GEPA Constitutional Critique Engine

Implements the CAI-inspired critique-revise loop for GEPA trading genomes.
The critique layer sits between genome proposal and backtest evaluation —
filtering invalid genomes before expensive eval, and producing structured
feedback for the preference model.

Flow:
  Genome Proposed
      ↓
  critiqueGenome()      ← LLM critique against constitution
      ↓
  applyVeto()         ← Tier 1 violations → reject immediately
      ↓
  backtestGenome()   ← Only if veto passed
      ↓
  applyPenalties()   ← Tier 2 flags → reduce effective score
  applyOptimize()     ← Tier 3 metrics → update preference model
      ↓
  StoreResult()
"""
from __future__ import annotations

import enum
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constitution import (
    CONSTITUTION,
    Principle,
    PrincipleTier,
    OptimizationDirection,
    build_critique_prompt,
    get_principles_by_tier,
)
from .models import EvaluationResult


# ─────────────────────────────────────────────────────────────────────────────
# Critique result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Tier1Violation:
    principle_id: str
    description: str
    severity: str = "hard_veto"  # hard_veto | near_veto
    raw_finding: str = ""


@dataclass
class Tier2Flag:
    principle_id: str
    description: str
    observed_value: float | None = None
    threshold: float | None = None
    penalty_fraction: float = 0.0  # fraction of score to deduct


@dataclass
class Tier3Metric:
    principle_id: str
    metric_key: str
    observed_value: float | None = None
    direction: OptimizationDirection | None = None
    weight: float = 1.0
    raw_note: str = ""


@dataclass
class CritiqueResult:
    """
    Structured output from a single critique pass.

    veto_passed: True if no Tier 1 violations (genome is eligible for backtest)
    tier1_violations: list of Tier 1 violations found
    tier2_flags: list of Tier 2 penalties flagged
    tier3_metrics: list of Tier 3 optimization metrics measured
    revision_needed: True if the LLM proposed a genome revision
    revised_genome: the revised genome source (if revision_needed=True)
    revision_rationale: why the LLM revised
    critique_notes: free-text notes from the LLM
    raw_llm_response: the raw LLM output (for debugging/auditing)
    critique_passed_at: timestamp
    iteration: which critique iteration this is (1 = first pass)
    """
    veto_passed: bool
    tier1_violations: list[Tier1Violation] = field(default_factory=list)
    tier2_flags: list[Tier2Flag] = field(default_factory=list)
    tier3_metrics: list[Tier3Metric] = field(default_factory=list)
    revision_needed: bool = False
    revised_genome: str = ""
    revision_rationale: str = ""
    critique_notes: str = ""
    raw_llm_response: str = ""
    critique_passed_at: str = ""
    iteration: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "veto_passed": self.veto_passed,
            "tier1_violations": [_dataclass_to_dict(v) for v in self.tier1_violations],
            "tier2_flags": [_dataclass_to_dict(f) for f in self.tier2_flags],
            "tier3_metrics": [_dataclass_to_dict(m) for m in self.tier3_metrics],
            "revision_needed": self.revision_needed,
            "revised_genome": self.revised_genome,
            "revision_rationale": self.revision_rationale,
            "critique_notes": self.critique_notes,
            "raw_llm_response": self.raw_llm_response,
            "critique_passed_at": self.critique_passed_at,
            "iteration": self.iteration,
        }


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialization helper
# ─────────────────────────────────────────────────────────────────────────────

def _dataclass_to_dict(obj: Any) -> Any:
    """
    Serialize a dataclass instance to a dict, converting enum values to strings.
    Fixes: dataclasses.asdict() does not handle enum serialization.
    Uses __dataclass_fields__ to detect dataclass instances without
    calling is_dataclass() from the module namespace.
    """
    if isinstance(obj, enum.Enum):
        return obj.value
    if getattr(obj, '__dataclass_fields__', None) is not None:
        result = {}
        for key in obj.__dataclass_fields__:
            value = getattr(obj, key)
            result[key] = _dataclass_to_dict(value)
        return result
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(item) for item in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# LLM critique runner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CritiqueLLMConfig:
    """Configuration for the LLM used for critique."""
    model: str = "claude-sonnet-4-7-2025"
    provider: str = "anthropic"
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.3  # Low temp for consistent structured output
    timeout_seconds: float = 60.0


@dataclass
class CritiqueConfig:
    """Configuration for the critique engine."""
    max_iterations: int = 3  # Max revise-critique loops before accepting
    revision_temperature: float = 0.7  # Higher temp for revision generation
    llm: CritiqueLLMConfig = field(default_factory=CritiqueLLMConfig)
    store_critiques: bool = True  # Write critiques to filesystem


class ConstitutionalCritiqueEngine:
    """
    Runs the CAI-inspired critique loop on a proposed genome.

    Usage:
        engine = ConstitutionalCritiqueEngine(config)
        result = engine.critique(genome_source="def signal():\n    ...")
        if result.veto_passed:
            # proceed to backtest
        else:
            # reject genome
    """

    def __init__(self, config: CritiqueConfig | None = None):
        self.config = config or CritiqueConfig()
        self._client: Any = None

    def critique(
        self,
        genome_source: str,
        prior_violations: list[str] | None = None,
        iteration: int = 1,
    ) -> CritiqueResult:
        """
        Run one critique pass on the genome.

        Calls the LLM with the constitutional prompt, parses the JSON response,
        and returns a structured CritiqueResult.
        """
        prompt = build_critique_prompt(genome_source, prior_violations)

        raw_response = self._call_llm(prompt)

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code fences
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError:
                    parsed = {}

        result = self._parse_critique_response(
            parsed=parsed,
            raw_llm_response=raw_response,
            genome_source=genome_source,
            iteration=iteration,
        )
        return result

    def critique_with_revision(
        self,
        genome_source: str,
        max_iterations: int | None = None,
    ) -> CritiqueResult:
        """
        Run the full CAI-inspired critique-revise loop.

        Iterates: critique → if veto failed and revision_needed → revise → critique again
        Stops when veto passes, max iterations reached, or no revision is needed.
        """
        max_iters = max_iterations or self.config.max_iterations
        current_genome = genome_source
        all_prior_violations: list[str] = []
        last_result: CritiqueResult | None = None

        for iteration in range(1, max_iters + 1):
            result = self.critique(
                genome_source=current_genome,
                prior_violations=all_prior_violations if all_prior_violations else None,
                iteration=iteration,
            )
            last_result = result

            if result.veto_passed:
                # Veto passed — genome is eligible for backtest
                return result

            # Veto failed
            if not result.revision_needed:
                # No revision proposed — we can't fix the violations, accept as-is
                # (this genome is rejected but we return honestly)
                return result

            # LLM proposed a revision — apply it
            if result.revised_genome and result.revised_genome.strip():
                current_genome = result.revised_genome
                # Track which violations need to be re-checked
                all_prior_violations.extend(
                    v.principle_id for v in result.tier1_violations
                )
            else:
                # Revision flagged but no actual revision text
                return result

        # Max iterations reached
        return last_result or CritiqueResult(
            veto_passed=False,
            critique_notes=f"Max iterations ({max_iters}) reached without veto passing",
            iteration=max_iters,
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM for critique."""
        client = self._get_client()
        cfg = self.config.llm

        try:
            if cfg.provider == "anthropic":
                if hasattr(client, "messages"):
                    # Anthropic SDK
                    response = client.messages.create(
                        model=cfg.model,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        messages=[{"role": "user", "content": prompt}],
                        timeout=cfg.timeout_seconds,
                    )
                    return response.content[0].text
                else:
                    # OpenAI-compatible endpoint with Anthropic model
                    response = client.chat.completions.create(
                        model=cfg.model,
                        messages=[
                            {"role": "system", "content": "You are GEPA-CRITO, a precise trading genome auditor. Respond ONLY with valid JSON."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                    )
                    return response.choices[0].message.content

            else:
                # OpenAI or Ollama
                response = client.chat.completions.create(
                    model=cfg.model,
                    messages=[
                        {"role": "system", "content": "You are GEPA-CRITO, a precise trading genome auditor. Respond ONLY with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                )
                return response.choices[0].message.content

        except Exception as e:
            # On LLM failure, return a "cannot critique" result
            return json.dumps({
                "tier1_violations": [],
                "tier2_flags": [],
                "tier3_metrics": {},
                "revision_needed": False,
                "revised_genome": "",
                "revision_rationale": f"LLM call failed: {str(e)}",
                "critique_notes": f"CRITIQUE FAILED — LLM unavailable: {str(e)}",
            })

    def _get_client(self) -> Any:
        """Lazy LLM client initialization."""
        if self._client is not None:
            return self._client

        cfg = self.config.llm
        if cfg.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic(
                    api_key=cfg.api_key or "",
                    base_url=cfg.base_url,
                )
            except ImportError:
                import openai
                self._client = openai.OpenAI(
                    api_key=cfg.api_key or "",
                    base_url=cfg.base_url or "https://api.anthropic.com",
                )
        elif cfg.provider == "openai":
            import openai
            self._client = openai.OpenAI(
                api_key=cfg.api_key or "",
                base_url=cfg.base_url,
            )
        elif cfg.provider == "ollama":
            import openai
            self._client = openai.OpenAI(
                api_key="ollama",
                base_url=cfg.base_url or "http://localhost:11434/v1",
            )
        return self._client

    def _parse_critique_response(
        self,
        parsed: dict[str, Any],
        raw_llm_response: str,
        genome_source: str,
        iteration: int,
    ) -> CritiqueResult:
        """Parse the LLM JSON response into a CritiqueResult."""

        # Parse Tier 1 violations
        tier1_violations: list[Tier1Violation] = []
        for pid in parsed.get("tier1_violations", []):
            p = self._get_principle(pid)
            tier1_violations.append(Tier1Violation(
                principle_id=pid,
                description=p.description if p else f"Unknown principle: {pid}",
                severity="hard_veto",
                raw_finding=parsed.get("tier1_notes", ""),
            ))

        # Parse Tier 2 flags
        tier2_flags: list[Tier2Flag] = []
        for fid in parsed.get("tier2_flags", []):
            p = self._get_principle(fid)
            tier2_flags.append(Tier2Flag(
                principle_id=fid,
                description=p.description if p else f"Unknown principle: {fid}",
                observed_value=parsed.get("tier2_values", {}).get(fid),
                threshold=p.threshold if p else None,
            ))

        # Parse Tier 3 metrics
        tier3_metrics: list[Tier3Metric] = []
        raw_metrics = parsed.get("tier3_metrics", {})
        if isinstance(raw_metrics, dict):
            for key, value in raw_metrics.items():
                p = self._get_principle_by_metric(key)
                if p:
                    tier3_metrics.append(Tier3Metric(
                        principle_id=p.id,
                        metric_key=key,
                        observed_value=float(value) if value is not None else None,
                        direction=p.direction,
                        weight=p.weight,
                    ))

        return CritiqueResult(
            veto_passed=len(tier1_violations) == 0,
            tier1_violations=tier1_violations,
            tier2_flags=tier2_flags,
            tier3_metrics=tier3_metrics,
            revision_needed=bool(parsed.get("revision_needed", False)),
            revised_genome=parsed.get("revised_genome", ""),
            revision_rationale=parsed.get("revision_rationale", ""),
            critique_notes=parsed.get("critique_notes", ""),
            raw_llm_response=raw_llm_response,
            critique_passed_at=datetime.now(UTC).isoformat(),
            iteration=iteration,
        )

    def _get_principle(self, principle_id: str) -> Principle | None:
        from .constitution import get_principle_by_id
        return get_principle_by_id(principle_id)

    def _get_principle_by_metric(self, metric_key: str) -> Principle | None:
        for p in CONSTITUTION:
            if p.metric_key == metric_key:
                return p
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Score modifiers (applied after backtest)
# ─────────────────────────────────────────────────────────────────────────────

def compute_penalty_fraction(flag: Tier2Flag) -> float:
    """
    Compute the penalty fraction for a Tier 2 flag.
    This is applied to the raw backtest score.
    """
    if flag.observed_value is None or flag.threshold is None:
        return 0.0

    p = flag.principle_id
    weight = 1.0

    # Look up the principle's weight
    for princ in CONSTITUTION:
        if princ.id == p:
            weight = princ.weight
            break

    # Compute how far over the threshold
    ratio = abs(flag.observed_value) / max(abs(flag.threshold), 1e-9)

    if ratio <= 1.0:
        return 0.0  # within threshold

    # exponential penalty: the further over, the worse
    if ratio > 3.0:
        return min(1.0, weight * 0.5)  # max 50% penalty for extreme violations
    elif ratio > 2.0:
        return min(1.0, weight * 0.25)
    elif ratio > 1.5:
        return min(1.0, weight * 0.15)
    else:
        return min(1.0, weight * 0.08)


def apply_critique_penalties(
    base_score: float,
    critique: CritiqueResult,
) -> float:
    """
    Apply Tier 2 penalty reductions to a backtest score.

    Tier 1 violations should already have caused rejection before this is called.
    This only applies to Tier 2 flags.

    Penalty is multiplicative, not additive — penalties compound.
    """
    if not critique.veto_passed:
        return 0.0  # already rejected

    if not critique.tier2_flags:
        return base_score  # no penalties

    effective_score = base_score
    total_penalty = 0.0

    for flag in critique.tier2_flags:
        penalty = compute_penalty_fraction(flag)
        total_penalty += penalty

    # Compound: (1 - total_penalty_1) * (1 - total_penalty_2) * ...
    # Cap total penalty at 80%
    total_penalty = min(total_penalty, 0.80)
    effective_score = base_score * (1.0 - total_penalty)

    return max(0.0, effective_score)


def score_from_tier3_metrics(
    metrics: list[Tier3Metric],
    base_score: float,
) -> float:
    """
    Score a genome based on Tier 3 optimization metrics.

    Each metric contributes (weight * normalized_value) to the score.
    Sharpe and alpha are normalized against a baseline.
    """
    if not metrics:
        return base_score

    total_adjustment = 0.0

    for m in metrics:
        if m.observed_value is None:
            continue

        # Sharpe: normalize. Sharpe of 1.0 = baseline. 2.0 = excellent. 0.0 = poor
        if m.metric_key == "sharpe":
            normalized = max(-1.0, min(3.0, (m.observed_value - 0.5) / 1.5))
            total_adjustment += m.weight * normalized * 0.2

        # Win rate: 0.50 = baseline. 0.60 = excellent
        elif m.metric_key == "win_rate":
            normalized = max(-1.0, min(1.0, (m.observed_value - 0.50) / 0.15))
            total_adjustment += m.weight * normalized * 0.1

        # Alpha: normalize against 0 baseline
        elif m.metric_key == "alpha":
            normalized = max(-1.0, min(2.0, m.observed_value / 0.10))
            total_adjustment += m.weight * normalized * 0.15

        # Trades per day: penalize both extremes
        elif m.metric_key == "trades_per_day":
            ideal = 2.0  # 2 trades/day is healthy
            deviation = abs(m.observed_value - ideal) / ideal
            normalized = -deviation  # negative adjustment for deviation
            total_adjustment += m.weight * normalized * 0.05

    return base_score + total_adjustment


# ─────────────────────────────────────────────────────────────────────────────
# Integration helpers
# ─────────────────────────────────────────────────────────────────────────────

def veto_summary(critique: CritiqueResult) -> str:
    """Human-readable veto summary."""
    if critique.veto_passed:
        return "✓ VETO PASSED — eligible for backtest"

    lines = ["✗ VETO FAILED — Tier 1 violations:"]
    for v in critique.tier1_violations:
        lines.append(f"  • {v.principle_id}: {v.description[:80]}")
    return "\n".join(lines)


def full_critique_summary(critique: CritiqueResult) -> str:
    """Human-readable full critique summary."""
    lines = [veto_summary(critique), ""]

    if critique.tier2_flags:
        lines.append(f"Tier 2 flags ({len(critique.tier2_flags)}):")
        for f in critique.tier2_flags:
            val_str = f"{f.observed_value:.3f}" if f.observed_value is not None else "N/A"
            lines.append(f"  • {f.principle_id}: {val_str}")
        lines.append("")

    if critique.tier3_metrics:
        lines.append(f"Tier 3 metrics ({len(critique.tier3_metrics)}):")
        for m in critique.tier3_metrics:
            val_str = f"{m.observed_value:.3f}" if m.observed_value is not None else "N/A"
            lines.append(f"  • {m.metric_key}: {val_str} ({m.direction.value if m.direction else 'N/A'})")
        lines.append("")

    if critique.revision_needed:
        lines.append(f"LLM revision: YES — rationale: {critique.revision_rationale[:100]}")
    else:
        lines.append("LLM revision: NO")

    if critique.critique_notes:
        lines.append(f"Notes: {critique.critique_notes[:200]}")

    return "\n".join(lines)
