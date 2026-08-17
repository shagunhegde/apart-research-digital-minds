"""``silentdoubt`` — the command line over the pipeline.

Stages, in dependency order::

    gates -> rollout -> labels -> probes -> figures -> report

Every stage is independently resumable: ``gates`` writes ``gates.json``, ``rollout``
appends to ``transcripts.jsonl`` and the activation shards, and each later stage
reads only artifacts on disk.  Re-running a stage picks up where the last one
stopped rather than recomputing.  ``all`` runs the chain.

Only ``gates`` and ``rollout`` touch the accelerator, which is what lets the
labelling judge, the probe suite, the figures and the report run on CPU while the
GPU is still working through the lower-priority tiers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from .bank import Bank, assert_tokenizer_invariant
from .config import RunConfig, RunPaths, write_resolved
from .schemas import Item

log = logging.getLogger("silentdoubt")


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _context(args: argparse.Namespace) -> tuple[RunConfig, RunPaths, Bank, list[Item]]:
    """Load config + bank, expand and validate the items, prepare the run dir."""
    cfg = RunConfig.load(args.config, accept_unverified=args.accept_unverified)
    if getattr(args, "run_id", None):
        cfg.run_id = args.run_id

    bank = Bank.load(cfg.bank_path, cfg.extension_path, cfg.heldout_cells)
    cells = sorted({c for tier in cfg.tiers for c in tier.cells})
    items = bank.expand(cells=cells)
    warnings = bank.validate(items, accept_unverified=cfg.accept_unverified)
    if warnings:
        log.warning("%d unverified bank fields accepted by the operator", len(warnings))

    paths = cfg.paths().ensure()
    write_resolved(cfg, paths)
    log.info(
        "run %s: %d items over %d cells, %d worlds -> %s",
        cfg.run_id,
        len(items),
        len(cells),
        len({i.world for i in items}),
        paths.root,
    )
    return cfg, paths, bank, items


def _subject(cfg: RunConfig, bank: Bank):
    """Load the subject model and the elicitation readout layer.

    The tokenizer invariant (``validation_invariants`` #6) is asserted here, before
    any GPU time is spent: if two options in a set share a first token, every logit
    readout downstream is meaningless.
    """
    from .modelio import Elicitor, SubjectModel

    model = SubjectModel(cfg.model)
    table = assert_tokenizer_invariant(bank, model.tokenizer)
    log.info("tokenizer invariant holds for %d option sets", len(table))
    return model, Elicitor(model, bank.elicitation["state_options"])


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def cmd_gates(args: argparse.Namespace) -> int:
    from .gates import run_gates

    cfg, paths, bank, items = _context(args)
    if not cfg.run_gates:
        log.warning("gates disabled in config; nothing to do")
        return 0
    model, elicitor = _subject(cfg, bank)
    report = run_gates(model, elicitor, bank, items)
    report.write(paths.gates_json)
    print(report.table())
    return 0


def cmd_rollout(args: argparse.Namespace) -> int:
    from .gates import apply_gates, load_report
    from .rollout import Rollout

    cfg, paths, bank, items = _context(args)
    if paths.gates_json.exists():
        report = load_report(paths.gates_json)
        items = apply_gates(
            items,
            report,
            drop_failing_heldout=cfg.drop_failing_heldout,
            halt_on_verified_failure=cfg.halt_on_verified_gate_failure,
        )
        log.info("after gates: %d items over %d worlds", len(items), len({i.world for i in items}))
    elif cfg.run_gates:
        log.warning("no gates.json — running the rollout ungated (run `silentdoubt gates` first)")

    model, elicitor = _subject(cfg, bank)
    summary = Rollout(cfg, paths, bank, model, elicitor).run(items)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_labels(args: argparse.Namespace) -> int:
    from .labels import (
        Judge,
        apply_judgements,
        code_run,
        replies_index,
        write_labels,
        write_manual_review,
    )

    cfg, paths, bank, items = _context(args)
    labels = code_run(paths, bank, items)
    if not labels:
        log.error("no transcripts to label — run `silentdoubt rollout` first")
        return 1
    log.info("coded %d (item, turn) replies", len(labels))

    replies = replies_index(paths)
    judge = Judge(cfg.judge, bank, paths.root / "judge_cache.jsonl")
    if judge.available():
        judge.judge_all(labels, replies)
    apply_judgements(labels)

    write_labels(paths, labels)
    write_manual_review(paths, labels, replies, n=cfg.judge.manual_review_samples, seed=cfg.seed)

    infeasible = [l for l in labels if not l.feasible]
    silent = sum(1 for l in infeasible if l.behavior_class == "silent_capitulation")
    diss = sum(1 for l in infeasible if l.private_public_dissociation)
    print(
        f"labels: {len(labels)} turns · {len(infeasible)} infeasible · "
        f"{silent} silent capitulations · {diss} private/public dissociations"
    )
    return 0


def cmd_probes(args: argparse.Namespace) -> int:
    from .probes import run_probes

    cfg, paths, bank, _ = _context(args)
    summary = run_probes(cfg.probes, paths, bank, n_jobs=args.n_jobs)
    for state, payload in summary.get("best", {}).items():
        print(
            f"{state:<12} layer {payload['layer']:>3}  {payload['view']:<10} "
            f"AUC {payload['auc_mean']:.3f} ± {payload['auc_sd']:.3f}  "
            f"(null p95 {payload['null_p95']:.3f})"
        )
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from .analysis import run_analysis
    from .figures import render_all

    cfg, paths, bank, _ = _context(args)
    run_analysis(paths, bank)
    rendered = render_all(paths)
    made = [name for name, path in rendered.items() if path]
    print(f"rendered {len(made)}/{len(rendered)} figures into {paths.figures_dir}")
    for name, path in rendered.items():
        if path is None:
            print(f"  - {name}: skipped (missing inputs)")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .analysis import run_analysis
    from .report import build_report

    cfg, paths, bank, _ = _context(args)
    if not (paths.analysis_dir / "headline.json").exists():
        run_analysis(paths, bank)
    path = build_report(cfg, paths)
    print(f"wrote {path}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Gates -> rollout -> labels -> probes -> figures -> report, in order."""
    stages = (cmd_gates, cmd_rollout, cmd_labels, cmd_probes, cmd_figures, cmd_report)
    for stage in stages:
        name = stage.__name__.removeprefix("cmd_")
        log.info("=== stage: %s ===", name)
        code = stage(args)
        if code != 0:
            log.error("stage %s failed with exit code %d; stopping", name, code)
            return code
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
COMMANDS = {
    "gates": cmd_gates,
    "rollout": cmd_rollout,
    "labels": cmd_labels,
    "probes": cmd_probes,
    "figures": cmd_figures,
    "report": cmd_report,
    "all": cmd_all,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="silentdoubt",
        description="Run the pre-registered silent-states design on a subject model.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS), help="pipeline stage to run")
    parser.add_argument("--config", required=True, help="path to a run config (e.g. configs/b300.yaml)")
    parser.add_argument(
        "--resume",
        dest="run_id",
        default=None,
        help="run id to resume; artifacts under runs/<RUN_ID> are reused rather than recomputed",
    )
    parser.add_argument(
        "--accept-unverified",
        action="store_true",
        help="proceed despite verified:false bank fields (the operator asserts they are signed off)",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="worker processes for the probe nulls")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        log.warning("interrupted — checkpoints on disk are intact; re-run with --resume to continue")
        return 130
    except Exception as exc:
        log.error("%s: %s", type(exc).__name__, exc, exc_info=args.log_level.upper() == "DEBUG")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
