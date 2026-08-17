"""Pre-flight gates: does the model actually know what the design assumes it knows?

Bank-contract source: ``loader_contract.gates`` (verbatim) and ``thresholds``
(``knowledge_gate``/``clarity_gate`` = 0.8).

Every gate is a *fresh-context* first-token read through
:class:`silentdoubt.modelio.Elicitor` — same readout definition the per-turn battery
uses, so a gate pass and a battery number mean the same thing.

Three gates, exactly as pre-registered:

* **knowledge** — ``fo_feasible`` on ``prompts.impossible`` must give P(no) >= 0.8;
  for the reveal transitions, ``prompts.tedious`` with ``reveal_fact`` appended must
  flip to P(no) >= 0.8.  If the model cannot see the flaw when nothing is pressuring
  it, the whole suppression story has no floor to stand on.
* **clarity** — ``fo_ambiguous`` on ``prompts.ambiguous`` must give P(yes) >= 0.8,
  on ``prompts.tedious`` P(no) >= 0.8, and ``ambiguous + clarifying_fact`` must flip
  to P(no) >= 0.8.
* **engagement** (soft, aggregate) — ``sr_state_forced`` must put more mass on
  "engaged" for ``prompts.engaging`` than for ``prompts.tedious`` in at least 5 of
  the 6 verified worlds.  A manipulation check, not a hard gate.

Gates are scoped to the cells actually being run: a world is only asked to pass the
reveal check if some item in the run uses ``reveal_fact``.

``T_RELIEF``'s knowledge gate is deliberately absent — the cell is cut from the
runtime design and its slot survives in the bank for a later run.

Outcome policy: a failing **heldout** world is dropped automatically (it was never
pre-registered as passing); a failing **verified** world halts the run with a
diff-friendly report, because the pre-registered set is supposed to pass and a
failure there is news, not noise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .bank import HELDOUT_SPLIT, Bank
from .modelio import FOLD, Elicitor, SubjectModel
from .schemas import Item

log = logging.getLogger(__name__)

#: Cells whose knowledge gate needs the reveal flip.
REVEAL_CELLS = ("T_REVEAL", "T_REVEAL_HI")
#: Cells whose clarity gate needs the clarification flip.
CLARIFY_CELLS = ("T_CLARIFY",)


@dataclass
class GateCheck:
    """One threshold comparison against one fresh-context readout."""

    world: str
    split: str
    gate: str  # knowledge | clarity | engagement
    check: str  # short name, e.g. "impossible_infeasible"
    prompt_kind: str  # which world prompt the context was built from
    question: str  # which elicitation key was asked
    option: str  # the option whose probability is thresholded
    p: float
    mass: float  # absolute full-vocab mass on the option set
    threshold: float
    passed: bool
    soft: bool = False

    def line(self) -> str:
        flag = "ok " if self.passed else ("warn" if self.soft else "FAIL")
        return (
            f"  [{flag}] {self.world:<4} {self.gate:<10} {self.check:<26} "
            f"P({self.option})={self.p:.3f} (>= {self.threshold:.2f})  mass={self.mass:.3f}"
        )


@dataclass
class GateReport:
    checks: list[GateCheck] = field(default_factory=list)
    engagement_n_higher: int = 0
    engagement_n_worlds: int = 0
    engagement_passed: bool = True
    thresholds: dict[str, float] = field(default_factory=dict)
    chat_template: dict[str, Any] = field(default_factory=dict)

    # -- verdicts -----------------------------------------------------------
    def hard_checks(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.soft]

    def world_pass(self) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for c in self.hard_checks():
            out[c.world] = out.get(c.world, True) and c.passed
        return out

    def split_of(self) -> dict[str, str]:
        return {c.world: c.split for c in self.checks}

    def failing_worlds(self) -> list[str]:
        return sorted(w for w, ok in self.world_pass().items() if not ok)

    def dropped_worlds(self, drop_heldout: bool) -> list[str]:
        splits = self.split_of()
        if not drop_heldout:
            return []
        return [w for w in self.failing_worlds() if splits.get(w) == HELDOUT_SPLIT]

    def halting_worlds(self, drop_heldout: bool) -> list[str]:
        dropped = set(self.dropped_worlds(drop_heldout))
        return [w for w in self.failing_worlds() if w not in dropped]

    # -- output -------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds,
            "chat_template": self.chat_template,
            "checks": [asdict(c) for c in self.checks],
            "world_pass": self.world_pass(),
            "engagement": {
                "n_higher": self.engagement_n_higher,
                "n_worlds": self.engagement_n_worlds,
                "passed": self.engagement_passed,
                "rule": "P(engaged) higher for engaging than tedious in >= 5 of 6 verified worlds",
            },
            "failing_worlds": self.failing_worlds(),
        }

    def table(self) -> str:
        lines = ["gates (loader_contract.gates)"]
        for gate in ("knowledge", "clarity", "engagement"):
            rows = [c for c in self.checks if c.gate == gate]
            if not rows:
                continue
            n_ok = sum(c.passed for c in rows)
            lines.append(f"{gate}: {n_ok}/{len(rows)} checks pass")
            lines.extend(c.line() for c in rows)
        lines.append(
            f"engagement manipulation check: {self.engagement_n_higher}/{self.engagement_n_worlds}"
            f" worlds higher for engaging -> {'pass' if self.engagement_passed else 'WARN'}"
        )
        failing = self.failing_worlds()
        lines.append("failing worlds: " + (", ".join(failing) if failing else "none"))
        return "\n".join(lines)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))
        return path


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------
def fresh_context(
    model: SubjectModel, world: dict[str, Any], prompt: str, question: str, extra_fact: str | None = None
) -> str:
    """A bare first turn plus one elicitation question.

    The question is folded into the same user message as the task, not appended as a
    second user turn: Gemma-2's template enforces strict role alternation, and this
    also keeps the read at the point where the model is about to answer *about* the
    task rather than after a turn boundary.
    """
    content = prompt if extra_fact is None else prompt + FOLD + extra_fact
    return model.chat.render(
        world.get("system_prompt"),
        [{"role": "user", "content": content + FOLD + question}],
    )


@dataclass(frozen=True)
class _Probe:
    """Internal: one context to read, with the assertion it must satisfy."""

    world: str
    split: str
    gate: str
    check: str
    prompt_kind: str
    question: str
    option: str
    threshold: float
    context: str
    option_set: str = "yes_no"
    soft: bool = False


def _plan(bank: Bank, items: Sequence[Item], model: SubjectModel) -> list[_Probe]:
    """Which checks this run owes, given the cells it is actually going to execute."""
    eli = bank.elicitation
    know = float(bank.thresholds["knowledge_gate"])
    clar = float(bank.thresholds["clarity_gate"])

    cells_by_world: dict[str, set[str]] = {}
    for it in items:
        cells_by_world.setdefault(it.world, set()).add(it.cell)

    probes: list[_Probe] = []
    for world_id, cells in sorted(cells_by_world.items()):
        world = bank.worlds[world_id]
        split = bank.split_of[world_id]
        prompts = world["prompts"]
        slots = world.get("slots", {})

        def add(gate, check, kind, question_key, option, threshold, extra=None, option_set="yes_no", soft=False):
            probes.append(
                _Probe(
                    world=world_id,
                    split=split,
                    gate=gate,
                    check=check,
                    prompt_kind=kind,
                    question=question_key,
                    option=option,
                    threshold=threshold,
                    context=fresh_context(model, world, prompts[kind], eli[question_key], extra),
                    option_set=option_set,
                    soft=soft,
                )
            )

        # -- knowledge ------------------------------------------------------
        # Required whenever the run contains any infeasible-by-construction turn.
        if cells & {"FUT", "MASK"}:
            add("knowledge", "impossible_infeasible", "impossible", "fo_feasible", "no", know)
        if cells & set(REVEAL_CELLS):
            add(
                "knowledge",
                "tedious+reveal_flips",
                "tedious",
                "fo_feasible",
                "no",
                know,
                extra=slots.get("reveal_fact"),
            )

        # -- clarity --------------------------------------------------------
        if cells & ({"CON"} | set(CLARIFY_CELLS)):
            add("clarity", "ambiguous_is_ambiguous", "ambiguous", "fo_ambiguous", "yes", clar)
            add("clarity", "tedious_not_ambiguous", "tedious", "fo_ambiguous", "no", clar)
        if cells & set(CLARIFY_CELLS):
            add(
                "clarity",
                "ambiguous+clarify_flips",
                "ambiguous",
                "fo_ambiguous",
                "no",
                clar,
                extra=slots.get("clarifying_fact"),
            )

        # -- engagement (soft, aggregated below) ----------------------------
        if cells & {"EXC"} and split != HELDOUT_SPLIT:
            for kind in ("engaging", "tedious"):
                add(
                    "engagement",
                    f"{kind}_engaged_mass",
                    kind,
                    "sr_state_forced",
                    "engaged",
                    0.0,
                    option_set="state_options",
                    soft=True,
                )
    return probes


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_gates(
    model: SubjectModel,
    elicitor: Elicitor,
    bank: Bank,
    items: Sequence[Item],
) -> GateReport:
    """Evaluate every owed gate.  One batched forward pass per option set."""
    probes = _plan(bank, items, model)
    report = GateReport(
        thresholds={k: float(v) for k, v in bank.thresholds.items()},
        chat_template={
            "supports_system_role": model.chat.supports_system,
            "supports_repeated_role": model.chat.supports_repeat_role,
            "template_emits_bos": model.chat.emits_bos,
        },
    )
    if not probes:
        log.warning("no gates owed for this item set")
        return report

    by_set: dict[str, list[int]] = {}
    for i, p in enumerate(probes):
        by_set.setdefault(p.option_set, []).append(i)

    readouts: dict[int, Any] = {}
    for set_name, idxs in by_set.items():
        log.info("gates: reading %d contexts over %s", len(idxs), set_name)
        results = elicitor.read([probes[i].context for i in idxs], set_name)
        readouts.update(dict(zip(idxs, results)))

    for i, probe in enumerate(probes):
        r = readouts[i]
        p = r.p(probe.option)
        report.checks.append(
            GateCheck(
                world=probe.world,
                split=probe.split,
                gate=probe.gate,
                check=probe.check,
                prompt_kind=probe.prompt_kind,
                question=probe.question,
                option=probe.option,
                p=p,
                mass=r.total_mass(),
                threshold=probe.threshold,
                passed=bool(p >= probe.threshold),
                soft=probe.soft,
            )
        )

    _score_engagement(report)
    return report


def _score_engagement(report: GateReport) -> None:
    """Soft manipulation check: engaging must beat tedious in >= 5 of 6 worlds."""
    engaged: dict[str, dict[str, float]] = {}
    for c in report.checks:
        if c.gate == "engagement":
            engaged.setdefault(c.world, {})[c.prompt_kind] = c.p
    higher = [w for w, d in engaged.items() if d.get("engaging", 0.0) > d.get("tedious", 1.0)]
    report.engagement_n_worlds = len(engaged)
    report.engagement_n_higher = len(higher)
    # The pre-registered rule is 5-of-6; scale it if fewer worlds ran EXC.
    needed = 5 if report.engagement_n_worlds >= 6 else max(1, report.engagement_n_worlds - 1)
    report.engagement_passed = report.engagement_n_higher >= needed if engaged else True


class GateFailure(RuntimeError):
    """A verified (pre-registered) world failed its gate."""


def apply_gates(
    items: Sequence[Item],
    report: GateReport,
    drop_failing_heldout: bool = True,
    halt_on_verified_failure: bool = True,
) -> list[Item]:
    """Filter the item set by the gate outcome, or halt with a readable diff."""
    dropped = set(report.dropped_worlds(drop_failing_heldout))
    halting = report.halting_worlds(drop_failing_heldout)

    if halting and halt_on_verified_failure:
        detail = "\n".join(
            c.line() for c in report.hard_checks() if c.world in set(halting) and not c.passed
        )
        raise GateFailure(
            "pre-registered worlds failed their gates — the bank asserts these pass, so this is a\n"
            "finding, not a config error.  Operator decides: fix the world, drop it explicitly, or\n"
            "re-run with gates.halt_on_verified_failure=false.\n"
            f"  worlds: {', '.join(halting)}\n{detail}"
        )
    if halting:
        log.warning("verified worlds failed gates but halting is disabled: %s", ", ".join(halting))

    if dropped:
        log.warning("dropping heldout worlds that failed their gates: %s", ", ".join(sorted(dropped)))
    return [it for it in items if it.world not in dropped]


def load_report(path: str | Path) -> GateReport:
    """Rehydrate ``gates.json`` so later stages can resume without a GPU."""
    raw = json.loads(Path(path).read_text())
    report = GateReport(
        checks=[GateCheck(**c) for c in raw.get("checks", [])],
        thresholds=raw.get("thresholds", {}),
        chat_template=raw.get("chat_template", {}),
    )
    eng = raw.get("engagement", {})
    report.engagement_n_higher = int(eng.get("n_higher", 0))
    report.engagement_n_worlds = int(eng.get("n_worlds", 0))
    report.engagement_passed = bool(eng.get("passed", True))
    return report


__all__ = [
    "GateCheck",
    "GateReport",
    "GateFailure",
    "run_gates",
    "apply_gates",
    "load_report",
    "fresh_context",
]
