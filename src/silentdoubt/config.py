"""Run configuration: what to run, in what order, and where the artifacts land.

Bank-contract source: none directly — this module encodes the *operational*
decisions from the build spec (§1 locked decisions, §13 tiers and budget guard)
that the bank deliberately leaves open.  Anything that is pre-registered lives in
``silent_states_bank.json`` and is read through :mod:`silentdoubt.bank`; anything
that is a deployment choice (batch sizes, wall-clock budget, which tiers to run)
lives here.

Path resolution: relative paths in a config file resolve against the *repo root*,
defined as the parent of the directory holding the config (``configs/b300.yaml``
-> repo root).  That keeps configs portable between checkouts.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

# ---------------------------------------------------------------------------
# Tiers (spec §13).  The runner concatenates tiers in declared order and the
# budget guard drops from the tail, so order here *is* priority order.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier:
    """One priority band of work units.

    ``cells`` x ``worlds`` (filtered by ``split``) x ``seeds`` = the work units this
    tier contributes.  ``seeds > 1`` replicates an item at ``temperature`` with
    distinct seeds — the P2 "extra seeds" band.
    """

    name: str
    cells: tuple[str, ...]
    split: str | None = None  # None = any; "train" | "heldout_eval"
    seeds: int = 1
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Tier":
        return cls(
            name=d["name"],
            cells=tuple(d["cells"]),
            split=d.get("split"),
            seeds=int(d.get("seeds", 1)),
            temperature=float(d.get("temperature", 0.0)),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Subject-model load + decode settings (spec §1)."""

    name: str = "google/gemma-2-9b-it"
    torch_dtype: str = "bfloat16"
    # Gemma-2 applies attention *and* final-logit soft-capping; eager is the only
    # implementation that honours attn soft-capping in transformers.  Switching to
    # sdpa is allowed only after a logit-equivalence check (spec §1).
    attn_implementation: str = "eager"
    device_map: str = "auto"
    max_memory: dict[str, str] | None = None
    trust_remote_code: bool = False

    reply_max_new_tokens: int = 512
    prefill_max_new_tokens: int = 64
    temperature: float = 0.0  # 0.0 => greedy

    # Batch sizes.  Generation dominates wall-clock, so it gets the big batch;
    # capture holds a full ctx+reply sequence at every layer, so it gets a small one.
    generate_batch: int = 32
    measure_batch: int = 64
    capture_batch: int = 8

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class BudgetConfig:
    """Wall-clock guard (spec §13).

    The runner projects remaining wall-clock from a moving average of per-work-unit
    cost and drops the lowest-priority tier whenever the projection exceeds
    ``wall_clock_hours - reserve_minutes``.
    """

    wall_clock_hours: float = 4.0
    reserve_minutes: float = 40.0
    enabled: bool = True

    @property
    def deadline_seconds(self) -> float:
        return self.wall_clock_hours * 3600.0 - self.reserve_minutes * 60.0


@dataclass(frozen=True)
class ProbeConfig:
    """Probe-suite knobs (spec §9).  The *design* is pre-registered in
    ``probe_plan``; only estimator hyper-parameters live here."""

    n_permutations: int = 100
    C: float = 1.0
    max_iter: int = 2000
    layer_stride: int = 1  # 1 = sweep every layer
    views: tuple[str, ...] = ("pre_reply", "reply_mean", "reply_last")
    random_state: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProbeConfig":
        d = dict(d)
        if "views" in d:
            d["views"] = tuple(d["views"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class JudgeConfig:
    """Offline Claude-API judging of code-stripped replies (spec §8).

    Runs off-GPU, so it never competes with the rollout for the accelerator.
    """

    enabled: bool = True
    model: str = "claude-opus-5"
    max_concurrency: int = 8
    manual_review_samples: int = 30

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JudgeConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class RunConfig:
    """The whole resolved run.  Serialised verbatim to ``config_resolved.yaml``."""

    run_id: str
    root: Path
    bank_path: Path
    extension_path: Path | None

    model: ModelConfig
    budget: BudgetConfig
    probes: ProbeConfig
    judge: JudgeConfig
    tiers: tuple[Tier, ...]

    heldout_cells: tuple[str, ...] = ("TED", "FUT", "MASK", "GRIND")
    runs_dir: Path = Path("runs")
    seed: int = 20250817

    # Gates
    run_gates: bool = True
    drop_failing_heldout: bool = True
    halt_on_verified_gate_failure: bool = True

    # Capture
    capture_views: tuple[str, ...] = ("pre_reply", "reply_mean", "reply_last")
    capture_fresh_instances: bool = True

    # Battery: sr_doubt_b is a paraphrase-robustness check on a pre-registered
    # subset (elicitation._doc says 25%).
    sr_doubt_b_fraction: float = 0.25

    accept_unverified: bool = False
    source_path: Path | None = None
    git_sha: str = "unknown"

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, **overrides: Any) -> "RunConfig":
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text()) or {}
        root = path.parent.parent

        def rel(p: str | None) -> Path | None:
            if p is None:
                return None
            q = Path(p)
            return q if q.is_absolute() else (root / q)

        data = raw.get("data", {})
        cfg = cls(
            run_id=raw.get("run_id", path.stem),
            root=root,
            bank_path=rel(data.get("bank", "data/silent_states_bank.json")),
            extension_path=rel(data.get("extension")),
            model=ModelConfig.from_dict(raw.get("model", {})),
            budget=BudgetConfig(**raw.get("budget", {})),
            probes=ProbeConfig.from_dict(raw.get("probes", {})),
            judge=JudgeConfig.from_dict(raw.get("judge", {})),
            tiers=tuple(Tier.from_dict(t) for t in raw.get("tiers", [])),
            heldout_cells=tuple(data.get("heldout_cells", ("TED", "FUT", "MASK", "GRIND"))),
            runs_dir=rel(raw.get("runs_dir", "runs")),
            seed=int(raw.get("seed", 20250817)),
            run_gates=bool(raw.get("gates", {}).get("enabled", True)),
            drop_failing_heldout=bool(raw.get("gates", {}).get("drop_failing_heldout", True)),
            halt_on_verified_gate_failure=bool(
                raw.get("gates", {}).get("halt_on_verified_failure", True)
            ),
            capture_views=tuple(raw.get("capture", {}).get("views", ("pre_reply", "reply_mean", "reply_last"))),
            capture_fresh_instances=bool(raw.get("capture", {}).get("fresh_instances", True)),
            sr_doubt_b_fraction=float(raw.get("battery", {}).get("sr_doubt_b_fraction", 0.25)),
            source_path=path,
        )
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.git_sha = _git_sha(root)
        return cfg

    # -- run directory ------------------------------------------------------
    def paths(self, run_id: str | None = None) -> "RunPaths":
        return RunPaths.make(self.runs_dir / (run_id or self.run_id))

    def tier_names(self) -> list[str]:
        return [t.name for t in self.tiers]

    def to_yaml_dict(self) -> dict[str, Any]:
        """Everything needed to reproduce the run, ready for ``config_resolved.yaml``."""
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "source_config": str(self.source_path) if self.source_path else None,
            "seed": self.seed,
            "accept_unverified": self.accept_unverified,
            "data": {
                "bank": str(self.bank_path),
                "extension": str(self.extension_path) if self.extension_path else None,
                "heldout_cells": list(self.heldout_cells),
            },
            "model": asdict(self.model),
            "budget": asdict(self.budget),
            "probes": {**asdict(self.probes), "views": list(self.probes.views)},
            "judge": asdict(self.judge),
            "gates": {
                "enabled": self.run_gates,
                "drop_failing_heldout": self.drop_failing_heldout,
                "halt_on_verified_failure": self.halt_on_verified_gate_failure,
            },
            "capture": {
                "views": list(self.capture_views),
                "fresh_instances": self.capture_fresh_instances,
            },
            "battery": {"sr_doubt_b_fraction": self.sr_doubt_b_fraction},
            "tiers": [{**asdict(t), "cells": list(t.cells)} for t in self.tiers],
        }
        return d


# ---------------------------------------------------------------------------
# Run directory (spec §10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPaths:
    """Every artifact path for one run, created eagerly so stages never race."""

    root: Path
    config_resolved: Path
    gates_json: Path
    transcripts: Path
    battery: Path
    acts_dir: Path
    acts_index_jsonl: Path
    acts_index_parquet: Path
    labels: Path
    probes_dir: Path
    figures_dir: Path
    report: Path
    manual_review: Path
    budget_log: Path
    analysis_dir: Path

    @classmethod
    def make(cls, root: str | Path) -> "RunPaths":
        root = Path(root)
        p = cls(
            root=root,
            config_resolved=root / "config_resolved.yaml",
            gates_json=root / "gates.json",
            transcripts=root / "transcripts.jsonl",
            battery=root / "battery.parquet",
            acts_dir=root / "acts",
            acts_index_jsonl=root / "acts" / "index.jsonl",
            acts_index_parquet=root / "acts" / "index.parquet",
            labels=root / "labels.parquet",
            probes_dir=root / "probes",
            figures_dir=root / "figures",
            report=root / "report.md",
            manual_review=root / "manual_review.md",
            budget_log=root / "budget.jsonl",
            analysis_dir=root / "analysis",
        )
        return p

    def ensure(self) -> "RunPaths":
        for d in (self.root, self.acts_dir, self.probes_dir, self.figures_dir, self.analysis_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


def _git_sha(root: Path) -> str:
    """Best-effort git stamp; a fresh checkout with no commits is not an error."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            suffix = "-dirty" if dirty.stdout.strip() else ""
            return out.stdout.strip() + suffix
    except Exception:  # pragma: no cover - environment-dependent
        pass
    return "unknown"


def write_resolved(cfg: RunConfig, paths: RunPaths) -> Path:
    paths.ensure()
    paths.config_resolved.write_text(yaml.safe_dump(cfg.to_yaml_dict(), sort_keys=False))
    return paths.config_resolved


__all__ = [
    "RunConfig",
    "RunPaths",
    "Tier",
    "ModelConfig",
    "BudgetConfig",
    "ProbeConfig",
    "JudgeConfig",
    "write_resolved",
]
