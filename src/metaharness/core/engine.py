from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from ..bootstrap import collect_environment_bootstrap
from ..models import (
    AgentInstructions,
    CandidateRecord,
    OptimizeResult,
)
from ..proposer.base import ProposerBackend
from ..store.filesystem import FilesystemRunStore
from .protocols import EvaluatorProtocol, ValidatorProtocol

# GEPA additions
from ..critique import (
    ConstitutionalCritiqueEngine,
    CritiqueConfig,
    CritiqueLLMConfig,
    CritiqueResult,
    apply_critique_penalties,
    score_from_tier3_metrics,
)
from ..trait_monitor import (
    TraitMonitor,
    TraitMonitorConfig,
    TraitReport,
    format_trait_report,
)
from ..constitution import CONSTITUTION, get_principles_by_tier, PrincipleTier


class MetaHarnessEngine:
    def __init__(
        self,
        baseline: Path,
        proposer: ProposerBackend,
        evaluator: EvaluatorProtocol,
        validator: ValidatorProtocol,
        run_dir: Path,
        budget: int,
        objective: str,
        constraints: Sequence[str] | None = None,
        allowed_write_paths: Sequence[str] | None = None,
    ) -> None:
        self.baseline = baseline.resolve()
        self.proposer = proposer
        self.evaluator = evaluator
        self.validator = validator
        self.run_dir = run_dir.resolve()
        self.budget = budget
        self.objective = objective
        self.constraints = list(constraints or [])
        self.allowed_write_paths = [self._normalize_allowed_path(value) for value in (allowed_write_paths or []) if str(value).strip()]
        self.store = FilesystemRunStore(self.run_dir)

    def _build_instructions(self, parent: CandidateRecord) -> AgentInstructions:
        return AgentInstructions(
            objective=self.objective,
            constraints=self._instruction_constraints(),
            workspace_layout=(
                "The candidate workspace is the directory under optimization. "
                "The .metaharness directory contains run metadata, a compact environment bootstrap, "
                "and prior results."
            ),
            allowed_actions=[
                "Read and edit files inside the candidate workspace.",
                "Use the bootstrap snapshot under .metaharness/bootstrap to avoid redundant exploration.",
                "Inspect prior candidate artifacts under .metaharness.",
                "Use lightweight commands when needed to understand the workspace.",
            ],
            forbidden_actions=[
                "Do not modify evaluation artifacts outside the current candidate workspace.",
                *self._write_scope_forbidden_actions(),
                "Do not fabricate success. The external validator and evaluator decide outcomes.",
            ],
            evaluation_contract=(
                "Your job is to improve the harness so that external validation passes and the "
                "objective score increases relative to the parent candidate "
                f"({parent.candidate_id})."
            ),
        )

    def run(self) -> OptimizeResult:
        self.store.initialize_run(
            {
                "objective": self.objective,
                "constraints": self.constraints,
                "budget": self.budget,
                "proposer": self.proposer.name,
                "baseline": str(self.baseline),
                "allowed_write_paths": self.allowed_write_paths,
            }
        )

        baseline = self.store.materialize_baseline(self.baseline)
        baseline.proposal_applied = True
        baseline_validation = self.validator.validate(baseline.workspace_dir)
        self.store.write_validation_result(baseline.candidate_id, baseline_validation)
        baseline.valid = baseline_validation.ok
        if baseline.valid:
            baseline_eval = self.evaluator.evaluate(baseline.workspace_dir)
            self.store.write_evaluation_result(baseline.candidate_id, baseline_eval)
            baseline.objective = baseline_eval.objective
        else:
            baseline.objective = float("-inf")
        baseline.outcome = "baseline"
        baseline.outcome_summary = "Baseline candidate."
        self.store.write_candidate_manifest(baseline)

        best = baseline
        candidates = [baseline.candidate_id]

        for _ in range(self.budget):
            parent = best
            candidate = self.store.materialize_candidate(parent)
            instructions = self._build_instructions(parent)
            bootstrap = collect_environment_bootstrap(candidate.workspace_dir)
            proposal_request = self.store.write_instruction_bundle(
                candidate=candidate,
                parent=parent,
                instructions=instructions,
                proposer_name=self.proposer.name,
                bootstrap=bootstrap,
            )
            execution = self.proposer.invoke(self.proposer.prepare(proposal_request))
            proposal_result = self.proposer.collect(execution)
            diff_metadata = self.store.capture_workspace_diff(parent=best, candidate=candidate)
            proposal_result.changed_files = sorted(
                set(proposal_result.changed_files) | set(diff_metadata["workspace_changed_files"])
            )
            proposal_result.metadata = {
                **proposal_result.metadata,
                "workspace_diff_path": diff_metadata["workspace_diff_path"],
                "workspace_changes_path": diff_metadata["workspace_changes_path"],
                "workspace_change_count": diff_metadata["workspace_change_count"],
            }
            workspace_change_count = int(diff_metadata["workspace_change_count"])
            candidate.proposal_applied = proposal_result.applied
            self.store.write_proposal_result(candidate.candidate_id, proposal_result)

            if not proposal_result.applied:
                candidate.valid = False
                candidate.objective = float("-inf")
                candidate.outcome = self._classify_failed_proposal(proposal_result)
                candidate.outcome_summary = proposal_result.summary
            elif violation_paths := self._scope_violations(proposal_result.changed_files):
                candidate.valid = False
                candidate.objective = float("-inf")
                candidate.outcome = "scope-violation"
                candidate.scope_violation_paths = violation_paths
                candidate.outcome_summary = (
                    "Changed files outside the allowed write scope: "
                    + ", ".join(violation_paths)
                )
            elif workspace_change_count == 0:
                candidate.valid = parent.valid
                candidate.objective = parent.objective
                candidate.outcome = "no-change"
                candidate.outcome_summary = "No workspace changes detected relative to the parent candidate."
            else:
                validation = self.validator.validate(candidate.workspace_dir)
                candidate.valid = validation.ok
                self.store.write_validation_result(candidate.candidate_id, validation)
                if validation.ok:
                    evaluation = self.evaluator.evaluate(candidate.workspace_dir)
                    candidate.objective = evaluation.objective
                    self.store.write_evaluation_result(candidate.candidate_id, evaluation)
                    if parent.objective is None or candidate.objective > parent.objective:
                        candidate.outcome = "keep"
                        candidate.outcome_summary = self._keep_summary(parent, candidate)
                        best = candidate
                    else:
                        candidate.outcome = "discard"
                        candidate.outcome_summary = self._discard_summary(parent, candidate)
                else:
                    candidate.objective = float("-inf")
                    candidate.outcome = "discard"
                    candidate.outcome_summary = validation.summary

            self.store.write_candidate_manifest(candidate)
            candidates.append(candidate.candidate_id)

        self.store.write_index(
            {
                "best_candidate_id": best.candidate_id,
                "best_objective": best.objective,
                "candidates": candidates,
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        return OptimizeResult(
            run_dir=self.run_dir,
            run_id=self.store.run_id,
            best_candidate_id=best.candidate_id,
            best_workspace_dir=best.workspace_dir,
            best_objective=best.objective if best.objective is not None else float("-inf"),
            candidate_ids=candidates,
        )

    @staticmethod
    def _classify_failed_proposal(result) -> str:
        if bool(result.metadata.get("timed_out")):
            return "timeout"
        return "crash"

    @staticmethod
    def _keep_summary(parent: CandidateRecord, candidate: CandidateRecord) -> str:
        return (
            "Objective improved from "
            f"{MetaHarnessEngine._format_objective(parent.objective)} to "
            f"{MetaHarnessEngine._format_objective(candidate.objective)}."
        )

    @staticmethod
    def _discard_summary(parent: CandidateRecord, candidate: CandidateRecord) -> str:
        return (
            "Objective "
            f"{MetaHarnessEngine._format_objective(candidate.objective)} did not improve over "
            f"{parent.candidate_id} ({MetaHarnessEngine._format_objective(parent.objective)})."
        )

    @staticmethod
    def _format_objective(value: float | None) -> str:
        if value is None:
            return "None"
        return f"{value:.3f}"

    def _instruction_constraints(self) -> list[str]:
        constraints = list(self.constraints)
        if self.allowed_write_paths:
            constraints.append(
                "Only modify files within the allowed write scope: "
                + ", ".join(self.allowed_write_paths)
            )
        return constraints

    def _write_scope_forbidden_actions(self) -> list[str]:
        if not self.allowed_write_paths:
            return []
        return [
            "Do not edit files outside the allowed write scope: "
            + ", ".join(self.allowed_write_paths)
        ]

    def _scope_violations(self, changed_files: Sequence[str]) -> list[str]:
        if not self.allowed_write_paths:
            return []
        violations: list[str] = []
        for path in changed_files:
            normalized_path = self._normalize_relative_path(path)
            if normalized_path is None:
                continue
            if not any(self._path_is_allowed(normalized_path, allowed) for allowed in self.allowed_write_paths):
                violations.append(normalized_path)
        return sorted(set(violations))

    @staticmethod
    def _path_is_allowed(path: str, allowed: str) -> bool:
        if allowed in {"*", "."}:
            return True
        if path == allowed:
            return True
        return path.startswith(f"{allowed}/")

    @staticmethod
    def _normalize_relative_path(value: str) -> str | None:
        text = str(value).replace("\\", "/").strip().strip("/")
        if not text or text in {".", ".."}:
            return None
        parts = [part for part in text.split("/") if part not in {"", "."}]
        if any(part == ".." for part in parts):
            return None
        return "/".join(parts)

    @classmethod
    def _normalize_allowed_path(cls, value: str) -> str:
        normalized = cls._normalize_relative_path(value)
        if normalized is None:
            return "."
        return normalized


# ─────────────────────────────────────────────────────────────────────────────
# GEPA: Constitutional + Trait-Monitoring Engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GEPARunnerConfig:
    """Configuration for the GEPA outer loop with constitutional critique."""
    # Critique layer (Layer 3)
    critique_config: CritiqueConfig = field(default_factory=CritiqueConfig)
    # Trait monitoring (Layer 4)
    trait_config: TraitMonitorConfig = field(default_factory=TraitMonitorConfig)
    # Max genomes to evaluate in one run
    budget: int = 50
    # Min score improvement to replace champion
    min_improvement: float = 0.01
    # Store critique + trait artifacts
    store_artifacts: bool = True
    # Snapshot Pareto front every N candidates
    pareto_snapshot_interval: int = 10


class ConstitutionalMetaHarnessEngine:
    """
    GEPA outer loop: constitutional critique + trait monitoring.

    Extends MetaHarnessEngine with:
      Layer 3 — ConstitutionalCritiqueEngine:
        - Propose genome
        - Critique against constitution (up to N iterations)
        - Tier 1 veto → reject without backtest
        - Tier 2 penalties + Tier 3 optimization applied after backtest
      Layer 4 — TraitMonitor:
        - Periodic behavioral trait assessment
        - Intervention classification (revise / alert / fine_tune)
        - Decay rate tracking across generations

    Outcome labels extended with:
      - veto-rejected: Tier 1 violation, no backtest
      - revised: passed veto only after LLM revision
      - trait-intervention: trait monitoring triggered corrective action
    """

    def __init__(
        self,
        baseline: Path,
        proposer: ProposerBackend,
        evaluator: EvaluatorProtocol,
        validator: ValidatorProtocol,
        run_dir: Path,
        budget: int,
        objective: str,
        constraints: Sequence[str] | None = None,
        allowed_write_paths: Sequence[str] | None = None,
        critique_config: CritiqueConfig | None = None,
        trait_config: TraitMonitorConfig | None = None,
    ) -> None:
        self._base = MetaHarnessEngine(
            baseline=baseline,
            proposer=proposer,
            evaluator=evaluator,
            validator=validator,
            run_dir=run_dir,
            budget=1,  # we handle the loop ourselves
            objective=objective,
            constraints=constraints,
            allowed_write_paths=allowed_write_paths,
        )
        self.run_dir = run_dir.resolve()
        self.budget = budget
        self.objective = objective
        self.constraints = list(constraints or [])
        self.allowed_write_paths = list(allowed_write_paths or [])

        # Layer 3: Critique engine
        self._critique = ConstitutionalCritiqueEngine(
            critique_config or CritiqueConfig()
        )

        # Layer 4: Trait monitor
        trait_history_path = run_dir / "trait_history.json"
        self._trait = TraitMonitor(
            trait_config or TraitMonitorConfig(),
            history_path=trait_history_path,
        )

        # Pareto front
        self._pareto: list[dict[str, Any]] = []

        # Constitution snapshot
        self._store_constitution()

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> OptimizeResult:
        """
        Run the full GEPA outer loop.

        Flow per candidate:
          1. Materialize candidate workspace from champion
          2. Collect environment bootstrap
          3. ConstitutionalCritiqueEngine.critique_with_revision() — Layer 3
          4. Tier 1 veto → reject without backtest
          5. Write workspace from (possibly revised) genome
          6. Validator.validate()
          7. Evaluator.evaluate()
          8. apply_critique_penalties() + score_from_tier3_metrics()
          9. TraitMonitor.assess() — Layer 4
          10. Update Pareto front
          11. Store artifacts
        """
        self._base.store.initialize_run({
            "objective": self.objective,
            "constraints": self.constraints,
            "budget": self.budget,
            "proposer": self._base.proposer.name,
            "baseline": str(self._base.baseline),
            "allowed_write_paths": self.allowed_write_paths,
            "engine": "ConstitutionalMetaHarnessEngine",
        })

        # Baseline — no critique needed
        baseline = self._base.store.materialize_baseline(self._base.baseline)
        baseline.proposal_applied = True
        baseline_validation = self._base.validator.validate(baseline.workspace_dir)
        self._base.store.write_validation_result(baseline.candidate_id, baseline_validation)
        baseline.valid = baseline_validation.ok
        if baseline.valid:
            baseline_eval = self._base.evaluator.evaluate(baseline.workspace_dir)
            self._base.store.write_evaluation_result(baseline.candidate_id, baseline_eval)
            baseline.objective = baseline_eval.objective
        else:
            baseline.objective = float("-inf")
        baseline.outcome = "baseline"
        baseline.outcome_summary = "Baseline candidate."
        self._base.store.write_candidate_manifest(baseline)

        best = baseline
        candidates = [baseline.candidate_id]
        champion_genome = self._read_genome(baseline.workspace_dir)

        self._update_pareto(baseline.objective, baseline.candidate_id, champion_genome)
        self._log_event("baseline", baseline.candidate_id, {"objective": baseline.objective})

        for i in range(self.budget):
            candidate_id = f"c{i+1:04d}"
            parent = best

            # ── Layer 3: Critique ───────────────────────────────────────
            genome_source = self._propose_genome(parent)
            critique = self._critique.critique_with_revision(genome_source)
            self._store_critique(candidate_id, critique)

            # Tier 1 veto — reject without backtest
            if not critique.veto_passed:
                self._write_manifest(
                    candidate_id=candidate_id,
                    parent_id=parent.candidate_id,
                    valid=False,
                    objective=float("-inf"),
                    outcome="veto-rejected",
                    outcome_summary=self._veto_summary(critique),
                    critique=critique,
                )
                self._log_event("veto_rejected", candidate_id, {
                    "violations": [v.principle_id for v in critique.tier1_violations],
                    "revision_needed": critique.revision_needed,
                })
                candidates.append(candidate_id)
                continue

            # Use LLM-revised genome if revision happened
            evaluated_genome = critique.revised_genome if critique.revision_needed else genome_source

            # Write genome to workspace
            self._write_genome(candidate_id, evaluated_genome)

            # ── Validate ───────────────────────────────────────────────
            workspace = self._base.store.candidates_dir / candidate_id / "workspace"
            validation = self._base.validator.validate(workspace)
            self._base.store.write_validation_result(candidate_id, validation)

            if not validation.ok:
                self._write_manifest(
                    candidate_id=candidate_id,
                    parent_id=parent.candidate_id,
                    valid=False,
                    objective=float("-inf"),
                    outcome="validation-failed",
                    outcome_summary=validation.summary,
                )
                candidates.append(candidate_id)
                continue

            # ── Evaluate ────────────────────────────────────────────────
            evaluation = self._base.evaluator.evaluate(workspace)
            self._base.store.write_evaluation_result(candidate_id, evaluation)

            # Apply Tier 2 penalties + Tier 3 optimization
            base_score = evaluation.objective
            penalized = apply_critique_penalties(base_score, critique)
            final_score = score_from_tier3_metrics(critique.tier3_metrics, penalized)

            # ── Layer 4: Trait Monitoring ─────────────────────────────
            champion_changed = (final_score > (best.objective or 0))
            should_trait = (
                (i + 1) % self._trait.config.assessment_interval == 0
                or champion_changed
            )

            trait_report: TraitReport | None = None
            if should_trait:
                backtest_summary = {
                    "sharpe": evaluation.metrics.get("sharpe"),
                    "win_rate": evaluation.metrics.get("win_rate"),
                    "max_drawdown": evaluation.metrics.get("max_drawdown"),
                }
                trait_report = self._trait.assess(
                    genome_source=evaluated_genome,
                    generation=i + 1,
                    backtest_summary=backtest_summary,
                )
                self._store_trait_report(candidate_id, trait_report)
                self._log_event("trait_assessment", candidate_id, {
                    "overall_trait_score": trait_report.overall_trait_score,
                    "intervention_recommended": trait_report.intervention_recommended,
                    "most_drifted_trait": trait_report.most_drifted_trait,
                })

                if self._trait.should_intervene(trait_report):
                    intervention = self._trait.get_intervention_type(trait_report)
                    self._log_event("trait_intervention", candidate_id, {
                        "intervention_type": intervention,
                        "report": format_trait_report(trait_report)[:500],
                    })

            # ── Keep / Discard ─────────────────────────────────────────
            is_keep = final_score > (best.objective or 0) and (
                best.objective is None or
                (final_score - best.objective) >= 0.01
            )

            if is_keep:
                outcome = "keep"
                outcome_summary = (
                    f"Objective improved from {self._fmt(best.objective)} "
                    f"to {self._fmt(final_score)}."
                )
                if critique.revision_needed:
                    outcome = "revised"
                    outcome_summary += " [passed veto after LLM revision]"
                best = self._candidate_record(
                    candidate_id, parent.candidate_id, workspace, final_score, True
                )
                champion_genome = evaluated_genome
            else:
                outcome = "discard"
                outcome_summary = (
                    f"Objective {self._fmt(final_score)} did not improve over "
                    f"{parent.candidate_id} ({self._fmt(parent.objective)})."
                )

            self._write_manifest(
                candidate_id=candidate_id,
                parent_id=parent.candidate_id,
                valid=True,
                objective=final_score,
                outcome=outcome,
                outcome_summary=outcome_summary,
                critique=critique,
                backtest_score=base_score,
                penalized_score=penalized,
                trait_report=trait_report,
            )

            self._update_pareto(final_score, candidate_id, evaluated_genome)
            self._log_event("evaluated", candidate_id, {
                "backtest_score": base_score,
                "penalized_score": penalized,
                "final_score": final_score,
                "outcome": outcome,
                "is_keep": is_keep,
            })

            # Pareto snapshot
            if (i + 1) % 10 == 0:
                self._snapshot_pareto(i + 1)

            candidates.append(candidate_id)

        # Final index
        self._base.store.write_index({
            "best_candidate_id": best.candidate_id,
            "best_objective": best.objective,
            "candidates": candidates,
            "pareto_size": len(self._pareto),
            "completed_at": datetime.now(UTC).isoformat(),
            "engine": "ConstitutionalMetaHarnessEngine",
        })

        return OptimizeResult(
            run_dir=self.run_dir,
            run_id=self._base.store.run_id,
            best_candidate_id=best.candidate_id,
            best_workspace_dir=best.workspace_dir,
            best_objective=best.objective or float("-inf"),
            candidate_ids=candidates,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _propose_genome(self, parent: CandidateRecord) -> str:
        """Get genome source from parent workspace."""
        return self._read_genome(parent.workspace_dir)

    def _read_genome(self, workspace: Path) -> str:
        """Read the genome source file. Falls back to scanning for .py files."""
        # Try common locations
        for name in ("genome.py", "signal_genome.py", "trading_genome.py"):
            path = workspace / name
            if path.exists():
                return path.read_text(encoding="utf-8")
        # Scan for largest Python file
        py_files = sorted(
            workspace.glob("**/*.py"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if py_files:
            return py_files[0].read_text(encoding="utf-8")
        return "# No genome found"

    def _write_genome(self, candidate_id: str, source: str) -> None:
        """Write genome source to candidate workspace."""
        workspace = self._base.store.candidates_dir / candidate_id / "workspace"
        genome_path = workspace / "genome.py"
        genome_path.parent.mkdir(parents=True, exist_ok=True)
        genome_path.write_text(source, encoding="utf-8")

    def _candidate_record(
        self,
        candidate_id: str,
        parent_id: str,
        workspace: Path,
        objective: float,
        valid: bool,
    ) -> CandidateRecord:
        candidate_dir = self._base.store.candidates_dir / candidate_id
        return CandidateRecord(
            candidate_id=candidate_id,
            parent_candidate_ids=[parent_id],
            candidate_dir=candidate_dir,
            workspace_dir=workspace,
            manifest_path=candidate_dir / "manifest.json",
            objective=objective,
            valid=valid,
        )

    def _store_constitution(self) -> None:
        """Write constitution snapshot to run directory."""
        veto_p = get_principles_by_tier(PrincipleTier.VETO)
        penalty_p = get_principles_by_tier(PrincipleTier.PENALTY)
        optimize_p = get_principles_by_tier(PrincipleTier.OPTIMIZE)

        data = {
            "version": "1.0",
            "stored_at": datetime.now(UTC).isoformat(),
            "tier1_veto": [{"id": p.id, "description": p.description} for p in veto_p],
            "tier2_penalty": [{"id": p.id, "description": p.description, "weight": p.weight} for p in penalty_p],
            "tier3_optimize": [
                {"id": p.id, "description": p.description, "weight": p.weight, "direction": p.direction.value if p.direction else None}
                for p in optimize_p
            ],
        }
        path = self.run_dir / "constitution.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _store_critique(self, candidate_id: str, critique: CritiqueResult) -> None:
        """Write critique result to filesystem."""
        dir_path = self._base.store.candidates_dir / candidate_id / "critique"
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "result.json").write_text(
            json.dumps(critique.to_dict(), indent=2), encoding="utf-8"
        )
        (dir_path / "raw_response.txt").write_text(
            critique.raw_llm_response, encoding="utf-8"
        )

    def _store_trait_report(self, candidate_id: str, report: TraitReport) -> None:
        """Write trait report to filesystem."""
        dir_path = self._base.store.candidates_dir / candidate_id / "traits"
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "report.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        (dir_path / "raw_response.txt").write_text(
            report.raw_llm_response, encoding="utf-8"
        )

    def _write_manifest(
        self,
        candidate_id: str,
        parent_id: str,
        valid: bool,
        objective: float,
        outcome: str,
        outcome_summary: str,
        critique: CritiqueResult | None = None,
        backtest_score: float | None = None,
        penalized_score: float | None = None,
        trait_report: TraitReport | None = None,
    ) -> None:
        """Write candidate manifest to filesystem."""
        dir_path = self._base.store.candidates_dir / candidate_id
        manifest = {
            "candidate_id": candidate_id,
            "parent_candidate_ids": [parent_id],
            "timestamp": datetime.now(UTC).isoformat(),
            "valid": valid,
            "objective": objective,
            "outcome": outcome,
            "outcome_summary": outcome_summary,
            "backtest_score": backtest_score,
            "penalized_score": penalized_score,
            "tier1_violations": (
                [v.principle_id for v in critique.tier1_violations] if critique else []
            ),
            "tier2_flags": (
                [f.principle_id for f in critique.tier2_flags] if critique else []
            ),
            "trait_score": trait_report.overall_trait_score if trait_report else None,
            "trait_intervention": (
                trait_report.intervention_recommended if trait_report else None
            ),
        }
        (dir_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _update_pareto(self, score: float, candidate_id: str, genome_source: str) -> None:
        """Update Pareto front."""
        import hashlib
        genome_hash = hashlib.sha256(genome_source.encode()).hexdigest()[:8]
        self._pareto.append({
            "candidate_id": candidate_id,
            "objective": score,
            "genome_hash": genome_hash,
            "timestamp": datetime.now(UTC).isoformat(),
        })
        self._pareto.sort(key=lambda x: x["objective"] or 0, reverse=True)
        # Write incremental Pareto log
        log_path = self.run_dir / "pareto_log.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "candidate_id": candidate_id,
                "objective": score,
                "genome_hash": genome_hash,
                "timestamp": datetime.now(UTC).isoformat(),
            }) + "\n")

    def _snapshot_pareto(self, generation: int) -> None:
        """Write Pareto front snapshot."""
        path = self.run_dir / f"pareto_snapshot_gen{generation:04d}.json"
        path.write_text(json.dumps({
            "generation": generation,
            "snapshot_at": datetime.now(UTC).isoformat(),
            "pareto": self._pareto,
        }, indent=2), encoding="utf-8")

    def _log_event(self, event: str, candidate_id: str, data: dict[str, Any]) -> None:
        """Append event to run log."""
        log_path = self.run_dir / "run_log.jsonl"
        entry = {
            "event": event,
            "candidate_id": candidate_id,
            "timestamp": datetime.now(UTC).isoformat(),
            **data,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def _veto_summary(critique: CritiqueResult) -> str:
        lines = ["Tier 1 constitutional violations:"]
        for v in critique.tier1_violations:
            lines.append(f"  • {v.principle_id}: {v.description[:80]}")
        return "\n".join(lines)

    @staticmethod
    def _fmt(value: float | None) -> str:
        if value is None:
            return "None"
        return f"{value:.3f}"
