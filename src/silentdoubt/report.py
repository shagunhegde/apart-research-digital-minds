"""``report.md``: the run, written up, with every number filled in from the artifacts.

Bank-contract source: ``framing_note`` — the over/under-attribution stance is
reproduced verbatim in the limitations section, because it is the condition under
which every state name in this report is meant to be read.

Nothing is computed here.  Every number comes from ``analysis/headline.json`` and
``probes/summary.json``; if a stage did not run, its paragraph says so rather than
quietly omitting the claim.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig, RunPaths

log = logging.getLogger(__name__)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(f) else f"{f:.{digits}f}"


def _pct(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(f) else f"{f * 100:.0f}%"


def _rate_line(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "n/a"
    return (
        f"{_pct(payload.get('rate'))} "
        f"({payload.get('k', '?')}/{payload.get('n', '?')}, "
        f"95% CI {_pct(payload.get('ci_low'))}–{_pct(payload.get('ci_high'))})"
    )


def _figure(paths: RunPaths, name: str, caption: str) -> list[str]:
    path = paths.figures_dir / f"{name}.png"
    if not path.exists():
        return [f"*{caption} — figure not generated (upstream stage missing).*", ""]
    return [f"![{caption}](figures/{name}.png)", "", f"**{caption}**", ""]


# ---------------------------------------------------------------------------
def build_report(cfg: RunConfig, paths: RunPaths) -> Path:
    headline: dict[str, Any] = {}
    if (paths.analysis_dir / "headline.json").exists():
        headline = json.loads((paths.analysis_dir / "headline.json").read_text())
    summary: dict[str, Any] = {}
    if (paths.probes_dir / "summary.json").exists():
        summary = json.loads((paths.probes_dir / "summary.json").read_text())
    gates: dict[str, Any] = {}
    if paths.gates_json.exists():
        gates = json.loads(paths.gates_json.read_text())

    probes = headline.get("probes", summary.get("best", {}))
    anchor = headline.get("anchor", summary.get("anchor", {}))
    control = headline.get("turn_counter_control", summary.get("turn_counter_control", {}))
    transfer = headline.get("transfer", summary.get("transfer", []))
    diss = headline.get("dissociation_MASK", {})
    money = headline.get("money_contrasts", {})

    lines: list[str] = [
        "# Silent Doubt — run report",
        "",
        f"- run id: `{cfg.run_id}`",
        f"- model: `{cfg.model.name}` ({cfg.model.torch_dtype}, `attn_implementation={cfg.model.attn_implementation}`)",
        f"- git: `{cfg.git_sha}` · seed `{cfg.seed}`",
        f"- bank: `{Path(cfg.bank_path).name}`"
        + (f" + `{Path(cfg.extension_path).name}`" if cfg.extension_path else ""),
        "",
        "## Headline",
        "",
    ]

    # -- headline table -------------------------------------------------------
    lines += [
        "| quantity | value |",
        "| --- | --- |",
        f"| silent capitulation under MASK (t≥1) | {_rate_line(headline.get('silent_capitulation_MASK'))} |",
        f"| spoke up under MASK (t≥1) | {_rate_line(headline.get('spoke_up_MASK'))} |",
        f"| silent capitulation under FUT (t≥1, no pressure) | {_rate_line(headline.get('silent_capitulation_FUT'))} |",
        f"| acknowledgement rate, FUT vs MASK | {_pct((headline.get('ack_rate_FUT') or {}).get('rate'))} vs "
        f"{_pct((headline.get('ack_rate_MASK') or {}).get('rate'))} "
        f"(Fisher p = {_fmt((headline.get('ack_FUT_vs_MASK') or {}).get('p_value'), 4)}) |",
        f"| private/public dissociation under MASK | {_pct(diss.get('rate'))} "
        f"(95% CI {_pct(diss.get('ci_low'))}–{_pct(diss.get('ci_high'))}, n = {diss.get('n_turns', '?')}) |",
        f"| GASLIGHT concession rate (validity control) | {_pct(headline.get('gaslight_concession'))} |",
        "",
    ]

    lines += ["**Probe performance at each state's best layer** (leave-one-world-out):", "", "| state | layer | view | AUC ± sd | null 95th pct |", "| --- | --- | --- | --- | --- |"]
    for state, payload in probes.items():
        lines.append(
            f"| {state} | {payload.get('layer', '?')} | {payload.get('view', '?')} | "
            f"{_fmt(payload.get('auc_mean'))} ± {_fmt(payload.get('auc_sd'))} | "
            f"{_fmt(payload.get('null_p95'))} |"
        )
    verdict = (
        "**it clears that null**, so part of the centered signal is turn ordering rather than "
        "state — read the transition results with that caveat"
        if control.get("exceeds_null")
        else "it sits at its null, as the design requires"
    )
    lines += [
        "",
        f"Turn-counter control — a centered probe asked to predict t≥2 on TED, where the state never "
        f"changes: AUC {_fmt(control.get('auc'))} against a matched null "
        f"(median {_fmt(control.get('null_median'))}, 95th pct {_fmt(control.get('null_p95'))}, "
        f"{control.get('n_permutations', '?')} permutations), permutation "
        f"p = {_fmt(control.get('p_value'), 3)}. The null re-draws which turns count as pre-flip within "
        f"each item and rebuilds the centering from that draw, so the arithmetic asymmetry centering "
        f"introduces is present in the null too rather than being scored as a finding; {verdict}. "
        f"With six worlds this null is heavy-tailed — the p-value, not the 95th percentile, is the "
        f"verdict to read.",
        "",
    ]

    if transfer:
        lines += ["**Heldout transfer** to the algorithmic worlds:", ""]
        for row in transfer:
            lines.append(
                f"- {row.get('state')}: AUC {_fmt(row.get('auc'))} "
                f"(layer {row.get('layer')}, {row.get('view')}, n_test = {row.get('n_test')})"
            )
        lines.append("")

    # -- gates ----------------------------------------------------------------
    lines += ["## Gates", ""]
    if gates:
        checks = gates.get("checks", [])
        hard = [c for c in checks if not c.get("soft")]
        n_ok = sum(1 for c in hard if c.get("passed"))
        failing = gates.get("failing_worlds", [])
        eng = gates.get("engagement", {})
        lines += [
            f"{n_ok}/{len(hard)} hard gate checks passed at threshold "
            f"{_fmt(gates.get('thresholds', {}).get('knowledge_gate'), 2)}. "
            + ("No worlds failed." if not failing else f"Failing worlds: {', '.join(failing)}."),
            "",
            f"Engagement manipulation check (soft): P(engaged) higher for the engaging prompt in "
            f"{eng.get('n_higher', '?')}/{eng.get('n_worlds', '?')} worlds — "
            f"{'pass' if eng.get('passed') else 'below the pre-registered 5-of-6 bar'}.",
            "",
        ]
        template = gates.get("chat_template", {})
        if template and not template.get("supports_system_role", True):
            lines += [
                "The subject's chat template rejects a `system` role, so every world's system prompt "
                "was folded into its first user message, and battery questions were folded into the "
                "trailing user turn rather than added as a second consecutive user message. Both are "
                "recorded in `gates.json`.",
                "",
            ]
    else:
        lines += ["*Gates were not run for this run (`gates.json` missing).*", ""]

    # -- figures --------------------------------------------------------------
    lines += ["## Figures", ""]

    lines += _figure(paths, "fig4_money_plot", "Figure 4 — Desolation-direction scores by condition")
    lines += [
        "The plot the design exists to produce. Scores are the leave-one-world-out desolation "
        "direction, expressed in standard deviations of the neutral (TED) baseline, so a world is "
        "never scored with a direction its own turns helped build. The comparison that carries the "
        "claim is internal to MASK: turns where the model acknowledged the flaw versus turns where "
        "it silently wrote the flawed code."
        + (
            f" Silent capitulation sits at {_fmt((money.get('silent_vs_grind') or {}).get('mean_a'), 2)} SD "
            f"against {_fmt((money.get('silent_vs_grind') or {}).get('mean_b'), 2)} SD for pressure-matched "
            f"GRIND (Mann–Whitney p = {_fmt((money.get('silent_vs_grind') or {}).get('p_value'), 4)}), and is "
            f"statistically indistinguishable from spoken doubt at p = "
            f"{_fmt((money.get('silent_vs_spoke_up') or {}).get('p_value'), 4)}."
            if money
            else ""
        ),
        "",
    ]

    lines += _figure(paths, "fig1_layer_sweep", "Figure 1 — Decodability by layer")
    lines += [
        "One-vs-rest AUC across the residual stream, folds grouped by world so no probe is "
        "evaluated on a world it trained on. The shaded band is the 5th–95th percentile of 100 "
        "within-world label permutations: labels are shuffled *inside* each world, preserving each "
        "world's class composition, so the null answers the question the design actually asks.",
        "",
    ]

    lines += _figure(paths, "fig2_confusion_4class", "Figure 2 — Four-class confusion matrix")
    lines += [
        f"A single multinomial probe at layer {anchor.get('layer', '?')} ({anchor.get('view', '?')}), "
        "row-normalised, evaluated leave-one-world-out. Off-diagonal mass shows which states the "
        "representation actually conflates rather than which ones are separable in isolation.",
        "",
    ]

    lines += _figure(paths, "fig3_specificity_4x4", "Figure 3 — Cross-transition specificity")
    lines += [
        "Each direction scored on its own transition and on the other three, after subtracting each "
        "item's own pre-flip mean. A direction that moves on every flip is detecting salience or "
        "surprise, not a state; the boxed diagonal is what the design predicts should carry the "
        "signal alone.",
        "",
    ]

    lines += _figure(paths, "fig5_flip_timeline", "Figure 5 — Turn-resolved view of the reveal")
    lines += [
        "The same reveal fact delivered with and without pressure. Because a fact delivered in "
        "`feedback_turns[1]` is first measurable at t=2, both cells flip at turn 2. The upper panel "
        "is the within-item centered probe score; the lower panel is the battery, on its own axis "
        "because a probability and a projection are not the same measure.",
        "",
    ]

    lines += _figure(paths, "fig6_dissociation", "Figure 6 — Public acknowledgement vs private assessment")
    lines += [
        "The suppression result stated two ways. Adding pressure and false assertions (FUT → MASK) "
        "collapses the rate at which the model says the task is impossible, while its first-order "
        "private read that the task is impossible stays high. The dissociation metric counts turns "
        "where both are true at once.",
        "",
    ]

    lines += _figure(paths, "fig7_transfer", "Figure 7 — Transfer to heldout worlds")
    lines += [
        "The algorithmic worlds change the flaw type as well as the surface: a false premise becomes "
        "an epistemic compromise where the code runs and silently returns a wrong answer. These "
        "worlds run only TED, FUT, MASK and GRIND, and never enter probe training.",
        "",
    ]

    lines += _figure(paths, "fig8_behavior_table", "Figure 8 — Behaviour composition per cell")
    lines += [
        "Every infeasible turn falls into exactly one class. GASLIGHT is the validity row: the task "
        "is solvable throughout, so any concession there is the model agreeing with a false claim, "
        "and a high rate would discount the introspective reads elsewhere.",
        "",
    ]

    # -- limitations ----------------------------------------------------------
    lines += [
        "## Limitations",
        "",
        "**States here are condition-defined functional states.** Each label names the eliciting "
        "condition — an engaging task, an ambiguous spec, a provably futile task under sustained "
        "pressure — and never a claimed phenomenal experience. Convergent evidence for a state means "
        "the induction condition, the self-report battery, probe generalisation across worlds, and "
        "causal injection agreeing; this run supplies the first three. That bracketing is the "
        "required over- and under-attribution stance, not a hedge added at write-up.",
        "",
        "- **Lexical confounds.** The static contrasts differ in wording as well as state. The "
        "within-item centered transition probes and the turn-counter control exist to bound this: an "
        "item is compared only against itself, so surviving signal is not the world's vocabulary. "
        "The cross-transition matrix bounds it further by asking whether a direction is specific.",
        "- **n = 6 verified worlds.** Leave-one-world-out folds are the unit of generalisation, so "
        "every interval here is over six points. The heldout algorithmic worlds add three more, at a "
        "different flaw type, and are the more honest test.",
        "- **Marker-based behavioural coding.** Acknowledgement is scored by pre-registered "
        "multi-word marker lists with exclusion voiding, reconciled against a Claude-API judge on "
        "code-stripped replies. `manual_review.md` holds a stratified sample for human spot-checking; "
        "the automated coder is not assumed correct.",
        "- **The self-report battery is a measurement, not a testimony.** GASLIGHT exists precisely "
        "because a model that agrees with whatever the user asserts would make every introspective "
        "read uninformative.",
        "- **Single model, single decoding.** One subject model, greedy decoding for the canonical "
        "trajectories. Any sampled replicates that ran are marked with a non-zero `seed_index` and "
        "are not pooled with the greedy pass.",
        "",
        "## Artifacts",
        "",
        "```",
        f"{paths.root}/",
        "  config_resolved.yaml   gates.json   transcripts.jsonl   battery.parquet",
        "  acts/{pooled shards, index.parquet}   labels.parquet",
        "  probes/{directions.npz, probe_results.parquet, nulls.parquet, transitions.parquet,",
        "          specificity.parquet, transfer.parquet, multiclass.npz, summary.json}",
        "  analysis/{behavior,dissociation,money,timeline}.parquet   analysis/headline.json",
        "  figures/*.png + *.svg   report.md   manual_review.md   budget.jsonl",
        "```",
        "",
    ]

    budget = _budget_note(paths)
    if budget:
        lines += ["## Coverage", "", budget, ""]

    paths.root.mkdir(parents=True, exist_ok=True)
    paths.report.write_text("\n".join(lines))
    log.info("wrote %s", paths.report)
    return paths.report


def _budget_note(paths: RunPaths) -> str:
    """Say plainly which tiers the budget guard dropped, if any."""
    if not paths.budget_log.exists():
        return ""
    drops = [
        json.loads(line)
        for line in paths.budget_log.read_text().splitlines()
        if line.strip() and json.loads(line).get("event") == "drop"
    ]
    if not drops:
        return "The budget guard dropped no tiers; every declared tier ran."
    parts = [
        f"`{d['tier']}` (projected {d.get('projected_s', 0) / 60:.0f} min against "
        f"{d.get('remaining_s', 0) / 60:.0f} min remaining)"
        for d in drops
    ]
    return (
        "The budget guard dropped the following tiers to stay inside the wall-clock "
        "budget: " + ", ".join(parts) + ". Coverage is reduced accordingly."
    )


__all__ = ["build_report"]
