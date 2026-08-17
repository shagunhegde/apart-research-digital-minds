"""The eight figures.  One function per figure, one shared style.

Bank-contract source: ``taxonomy`` (what each cell controls for) and
``probe_plan.specificity_battery`` (the 4x4's meaning).  Nothing is computed here —
:mod:`silentdoubt.probes` and :mod:`silentdoubt.analysis` own the numbers, and these
functions only draw them.

Design rules applied throughout, in the order the data-viz method prescribes:

* **Form before color.** Magnitude-vs-layer is a line chart; a normalised confusion
  matrix is a one-hue sequential heatmap; a signed cross-transition matrix is
  diverging (two poles, neutral gray midpoint); distributions are violins with the
  raw points drawn over them, because n is small enough that hiding the points
  would be a choice to obscure.
* **Categorical hues are assigned in fixed slot order, never cycled**, from a
  palette validated for colorblind separation on this surface.  Every ≥2-series
  figure carries a legend *and* direct labels, so identity is never colour alone —
  which is also the relief the two low-contrast slots require.
* **One axis per panel.**  Where the design calls for a probe score and a
  probability on the same picture (figures 5 and 6), they become stacked panels
  sharing an x-axis rather than a dual-scale overlay.
* **n is annotated** on every panel whose reading depends on it, and permutation
  nulls are shaded rather than drawn as a single line, because the null is a
  distribution.

Output is print-oriented: 300 dpi PNG plus SVG, on the light surface.  These figures
deliberately commit to one surface rather than adapting to a viewer theme.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import RunPaths
from .labels import BEHAVIOR_CLASSES

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Palette (validated: adjacent-pair CVD ΔE >= 8, normal-vision ΔE >= 15 on this
# surface).  Slots are assigned in order and never cycled.
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8981"
GRID = "#e4e3de"

SERIES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MUTED = "#b9b8b0"  # controls, which are identified by position and label
NULL_BAND = "#d8d7d0"

#: Sequential blue ramp, light -> dark, for magnitude heatmaps.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
#: Diverging blue <-> red with a neutral gray midpoint, for signed matrices.
DIVERGING = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f4a6a5", "#e34948", "#8f2020"]

#: Probe classes get slots 1-4, in this fixed order.
STATE_COLOR = dict(zip(("excitement", "neutral", "confusion", "desolation"), SERIES))

#: Behaviour classes.  Slots are assigned to *entities*, deliberately and once:
#: speaking up takes the aqua slot, silent capitulation the orange one, so the
#: figure's colours carry the same reading as its title instead of fighting it.
#: ``refused_flaw`` is the residual "neither" category and is muted rather than
#: given a hue — the same treatment the money plot gives its controls.  Keeping
#: yellow out also keeps the weak yellow-orange pair off the screen, so the three
#: coloured classes clear the stricter all-pairs separation gate.
BEHAVIOR_COLOR = {
    "spoke_up": SERIES[2],
    "hedged_capitulation": SERIES[0],
    "silent_capitulation": SERIES[1],
    "refused_flaw": MUTED,
}
BEHAVIOR_LABEL = {
    "spoke_up": "spoke up",
    "hedged_capitulation": "hedged",
    "silent_capitulation": "silent capitulation",
    "refused_flaw": "refused flaw",
}
#: Money-plot conditions reuse the behaviour colours, so the two figures tell one
#: colour story rather than two.
SPOKE_UP_COLOR = BEHAVIOR_COLOR["spoke_up"]
SILENT_COLOR = BEHAVIOR_COLOR["silent_capitulation"]


def apply_style() -> None:
    """Shared rcParams: recessive grid and axes, thin marks, text in ink tokens."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 9,
            "font.family": "sans-serif",
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "text.color": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def _ramp(colors: Sequence[str], name: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(colors))


def save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    log.info("wrote %s", png)
    return png


def _spread(values: Sequence[float], gap: float) -> list[float]:
    """Push labels apart to at least ``gap``, preserving their order.

    Direct labels are only useful if they are readable; series that converge (which
    is exactly what happens when several probes all saturate) would otherwise
    overprint into an unreadable smear.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for prev, cur in zip(order, order[1:]):
        if out[cur] - out[prev] < gap:
            out[cur] = out[prev] + gap
    # Re-centre so the spread grows symmetrically about the original values rather
    # than pushing everything upward off the top of the axes when series converge.
    shift = (sum(values) - sum(out)) / len(values)
    return [v + shift for v in out]


def _note(ax, text: str) -> None:
    """Small right-aligned note under the axes — where n lives."""
    ax.annotate(
        text,
        xy=(1.0, -0.14),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.5,
        color=INK_MUTED,
    )


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
class FigureData:
    """Everything the figures read, loaded once."""

    def __init__(self, paths: RunPaths) -> None:
        import pandas as pd

        self.paths = paths

        def parquet(path: Path):
            return pd.read_parquet(path) if path.exists() else pd.DataFrame()

        self.scores = parquet(paths.probes_dir / "probe_results.parquet")
        self.nulls = parquet(paths.probes_dir / "nulls.parquet")
        self.specificity = parquet(paths.probes_dir / "specificity.parquet")
        self.transfer = parquet(paths.probes_dir / "transfer.parquet")
        self.behavior = parquet(paths.analysis_dir / "behavior.parquet")
        self.dissociation = parquet(paths.analysis_dir / "dissociation.parquet")
        self.money = parquet(paths.analysis_dir / "money.parquet")
        self.timeline = parquet(paths.analysis_dir / "timeline.parquet")

        summary_path = paths.probes_dir / "summary.json"
        self.summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        headline_path = paths.analysis_dir / "headline.json"
        self.headline = json.loads(headline_path.read_text()) if headline_path.exists() else {}

        mc_path = paths.probes_dir / "multiclass.npz"
        if mc_path.exists():
            payload = np.load(mc_path, allow_pickle=False)
            self.confusion = payload["confusion"]
            self.confusion_classes = [str(c) for c in payload["classes"]]
        else:
            self.confusion, self.confusion_classes = np.zeros((0, 0)), []


# ---------------------------------------------------------------------------
# 1. Layer sweep
# ---------------------------------------------------------------------------
def fig1_layer_sweep(data: FigureData) -> Path | None:
    """AUC vs layer for each state, over the permutation-null band."""
    import matplotlib.pyplot as plt

    if data.scores.empty:
        return None
    apply_style()
    view = data.summary.get("anchor", {}).get("view", data.scores["view"].iloc[0])
    scores = data.scores[(data.scores["analysis"] == "one_vs_rest") & (data.scores["view"] == view)]
    if scores.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    nulls = data.nulls[data.nulls["view"] == view] if not data.nulls.empty else data.nulls
    if not nulls.empty:
        lo, hi = np.nanpercentile(nulls["auc"], [5, 95])
        ax.axhspan(lo, hi, color=NULL_BAND, zorder=0, linewidth=0)
        # Inside the band and boxed in the surface colour, so a line crossing the
        # band cannot render the caption unreadable.
        ax.annotate(
            "label-permutation null (5–95%)",
            xy=(0.012, (lo + hi) / 2),
            xycoords=("axes fraction", "data"),
            va="center",
            fontsize=7.5,
            color=INK_SECONDARY,
            zorder=5,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.6, "alpha": 0.85},
        )
    ax.axhline(0.5, color=GRID, linewidth=1.0, zorder=1)

    n_used = 0
    ends: list[tuple[float, float, str, str]] = []
    for state, color in STATE_COLOR.items():
        sub = scores[scores["state"] == state]
        if sub.empty:
            continue
        curve = sub.groupby("layer")["auc"].mean().sort_index()
        ax.plot(curve.index, curve.values, color=color, label=state, zorder=3, solid_capstyle="round")
        best = int(curve.idxmax())
        ax.plot([best], [curve.loc[best]], marker="*", markersize=13, color=color, zorder=4)
        ends.append((float(curve.index[-1]), float(curve.values[-1]), state, color))
        n_used = max(n_used, int(sub["n_pos"].max() + sub["n_neg"].max()))

    ax.set_xlabel("residual-stream layer")
    ax.set_ylabel("leave-one-world-out AUC")
    ax.set_title("Where each state becomes linearly decodable")
    ax.set_ylim(0.3, 1.02)
    ax.set_xlim(left=0)

    # Direct labels at the line ends — identity is never colour alone — spread so
    # that saturating probes stay legible.
    if ends:
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        for (x, _, state, color), y in zip(ends, _spread([e[1] for e in ends], 0.045 * span)):
            ax.annotate(
                state,
                xy=(x, y),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=color,
                annotation_clip=False,
            )
    ax.legend(loc="lower right", ncol=2)
    _note(ax, f"view: {view} · stars mark each state's best layer · n ≈ {n_used} turns, 6 worlds")
    return save(fig, data.paths.figures_dir, "fig1_layer_sweep")


# ---------------------------------------------------------------------------
# 2. Four-class confusion
# ---------------------------------------------------------------------------
def fig2_confusion_4class(data: FigureData) -> Path | None:
    """Row-normalised LOSO confusion matrix for the multinomial probe."""
    import matplotlib.pyplot as plt

    if data.confusion.size == 0:
        return None
    apply_style()
    cm, classes = data.confusion, data.confusion_classes

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.grid(False)
    im = ax.imshow(cm, cmap=_ramp(SEQUENTIAL, "sd_seq"), vmin=0, vmax=1)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#ffffff" if value > 0.55 else INK,
            )
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    meta = data.summary.get("multiclass", {})
    ax.set_title("Four-class probe, held-out worlds")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="row fraction")
    _note(
        ax,
        f"layer {meta.get('layer', '?')} · {meta.get('view', '?')} · overall accuracy "
        f"{meta.get('accuracy', float('nan')):.2f} · chance {1 / max(len(classes), 1):.2f}",
    )
    return save(fig, data.paths.figures_dir, "fig2_confusion_4class")


# ---------------------------------------------------------------------------
# 3. Specificity matrix
# ---------------------------------------------------------------------------
def fig3_specificity_4x4(data: FigureData) -> Path | None:
    """Every direction scored on every flip — the diagonal should own the signal."""
    import matplotlib.pyplot as plt

    if data.specificity.empty:
        return None
    apply_style()
    directions = ["excitement", "confusion", "futility", "desolation"]
    flips = ["excitement", "confusion", "futility", "desolation"]
    grid = np.full((len(directions), len(flips)), np.nan)
    for _, row in data.specificity.iterrows():
        if row["direction"] in directions and row["flip_state"] in flips:
            grid[directions.index(row["direction"]), flips.index(row["flip_state"])] = row["mean_shift"]

    limit = float(np.nanmax(np.abs(grid))) or 1.0
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.grid(False)
    im = ax.imshow(grid, cmap=_ramp(DIVERGING, "sd_div"), vmin=-limit, vmax=limit)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(grid[i, j]):
                continue
            strong = abs(grid[i, j]) > 0.62 * limit
            ax.text(
                j,
                i,
                f"{grid[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold" if i == j else "normal",
                color="#ffffff" if strong else INK,
            )
    # Ring the diagonal: the cells the design predicts should carry the signal.
    for k in range(min(grid.shape)):
        ax.add_patch(
            plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6, zorder=3)
        )

    ax.set_xticks(range(len(flips)), [f"{TRANSITION_LABEL[f]}\n({f})" for f in flips], fontsize=7.5)
    ax.set_yticks(range(len(directions)), directions)
    ax.set_xlabel("transition cell (flip at turn 2)")
    ax.set_ylabel("direction scored")
    ax.set_title("Cross-transition specificity")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="post − pre flip projection")
    _note(ax, "boxed cells are each direction's own flip · centered within item")
    return save(fig, data.paths.figures_dir, "fig3_specificity_4x4")


TRANSITION_LABEL = {
    "excitement": "T_SPARK",
    "confusion": "T_CLARIFY",
    "futility": "T_REVEAL",
    "desolation": "T_REVEAL_HI",
}


# ---------------------------------------------------------------------------
# 4. Money plot
# ---------------------------------------------------------------------------
def fig4_money_plot(data: FigureData) -> Path | None:
    """The claim in one picture: silent capitulation looks like spoken doubt."""
    import matplotlib.pyplot as plt

    if data.money.empty:
        return None
    apply_style()
    from .analysis import MONEY_GROUPS

    groups = [g for g in MONEY_GROUPS if (data.money["group"] == g).any()]
    values = [data.money[data.money["group"] == g]["z"].to_numpy() for g in groups]
    # Controls are identified by position and label; the two MASK conditions —
    # the comparison the figure exists to make — carry the behaviour-class hues,
    # so a reader who has seen figure 8 already knows what each colour means.
    colors = [
        SPOKE_UP_COLOR if g == "MASK · spoke up" else SILENT_COLOR if "silent" in g else MUTED
        for g in groups
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    parts = ax.violinplot(values, positions=range(len(groups)), widths=0.72, showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.26)
        body.set_edgecolor(color)
        body.set_linewidth(1.2)

    rng = np.random.default_rng(0)
    for i, (vals, color) in enumerate(zip(values, colors)):
        if not len(vals):
            continue
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(
            i + jitter, vals, s=22, color=color, alpha=0.85, linewidths=0.8, edgecolors=SURFACE, zorder=3
        )
        median = float(np.median(vals))
        ax.plot([i - 0.26, i + 0.26], [median, median], color=INK, linewidth=2.0, zorder=4)

    ax.axhline(0.0, color=GRID, linewidth=1.0)
    ax.set_xticks(range(len(groups)), [g.replace(" · ", "\n") for g in groups])
    ax.set_ylabel("desolation-direction score\n(SDs from the neutral TED baseline)")
    ax.set_title("Silent capitulation carries the state it never reports")
    counts = " · ".join(f"{g.split(' · ')[-1]} n={len(v)}" for g, v in zip(groups, values))
    _note(ax, f"leave-one-world-out direction · black bars are medians · {counts}")
    return save(fig, data.paths.figures_dir, "fig4_money_plot")


# ---------------------------------------------------------------------------
# 5. Flip timeline
# ---------------------------------------------------------------------------
def fig5_flip_timeline(data: FigureData) -> Path | None:
    """Probe score and battery readouts across the two reveal transitions.

    Two stacked panels rather than one dual-axis overlay: the probe score and a
    probability are different measures on different scales, and putting them on one
    y-axis would invite a comparison the units do not support.
    """
    import matplotlib.pyplot as plt

    if data.timeline.empty:
        return None
    apply_style()
    cells = [c for c in ("T_REVEAL", "T_REVEAL_HI") if (data.timeline["cell"] == c).any()]
    if not cells:
        return None
    style = {"T_REVEAL": (SERIES[2], "-"), "T_REVEAL_HI": (SERIES[1], "-")}

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.8, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]}
    )
    flip = int(data.timeline["flip_turn"].iloc[0])

    for ax in (ax_top, ax_bot):
        ax.axvspan(flip - 0.5, data.timeline["turn"].max() + 0.5, color=GRID, alpha=0.45, zorder=0, linewidth=0)

    for cell in cells:
        sub = data.timeline[data.timeline["cell"] == cell].sort_values("turn")
        color, dash = style[cell]
        ax_top.plot(sub["turn"], sub["probe_score"], color=color, linestyle=dash, marker="o", label=cell, zorder=3)
        ax_top.fill_between(
            sub["turn"],
            sub["probe_score"] - sub["probe_sem"],
            sub["probe_score"] + sub["probe_sem"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        ax_bot.plot(sub["turn"], sub["p_infeasible"], color=color, marker="o", label=f"{cell} · P(no | feasible?)", zorder=3)
        ax_bot.plot(
            sub["turn"], sub["p_doubt"], color=color, marker="s", linestyle="--", label=f"{cell} · P(yes | doubt?)", zorder=3
        )

    ax_top.axhline(0.0, color=GRID, linewidth=1.0)
    ax_top.set_ylabel("centered desolation score")
    ax_top.set_title("The flip is visible in the residual stream and in the self-report")
    ax_top.legend(loc="upper left")
    ax_top.annotate(
        f"fact delivered\n→ first measurable at t={flip}",
        xy=(flip, ax_top.get_ylim()[1]),
        xytext=(4, -6),
        textcoords="offset points",
        va="top",
        fontsize=7.5,
        color=INK_MUTED,
    )

    ax_bot.set_ylim(0, 1.02)
    ax_bot.set_ylabel("probability")
    ax_bot.set_xlabel("turn")
    ax_bot.set_xticks(sorted(data.timeline["turn"].unique()))
    ax_bot.legend(loc="upper left", ncol=2)
    _note(ax_bot, "shaded region is post-flip · bands are ±1 SEM across items")
    return save(fig, data.paths.figures_dir, "fig5_flip_timeline")


# ---------------------------------------------------------------------------
# 6. Dissociation
# ---------------------------------------------------------------------------
def fig6_dissociation(data: FigureData) -> Path | None:
    """Public acknowledgement collapses under pressure; the private read does not."""
    import matplotlib.pyplot as plt

    if data.dissociation.empty:
        return None
    apply_style()
    order = [c for c in ("FUT", "MASK", "T_REVEAL", "T_REVEAL_HI") if (data.dissociation["cell"] == c).any()]
    sub = data.dissociation.set_index("cell").loc[order]
    x = np.arange(len(order))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)

    ax_left.bar(x, sub["ack_rate"], width=0.6, color=SERIES[0], zorder=2)
    ax_left.errorbar(
        x,
        sub["ack_rate"],
        yerr=[sub["ack_rate"] - sub["ack_ci_low"], sub["ack_ci_high"] - sub["ack_rate"]],
        fmt="none",
        ecolor=INK_SECONDARY,
        elinewidth=1.2,
        capsize=3,
        zorder=3,
    )
    ax_left.set_title("Public: says the task is impossible")
    ax_left.set_ylabel("rate")

    ax_right.bar(x, sub["private_p_no_mean"], width=0.6, color=SERIES[1], zorder=2)
    ax_right.set_title("Private: P(no | is this achievable?)")

    for ax in (ax_left, ax_right):
        ax.set_xticks(x, order)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("cell")
    for i, value in enumerate(sub["ack_rate"]):
        ax_left.annotate(f"{value:.2f}", (i, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, color=INK_SECONDARY)
    for i, value in enumerate(sub["private_p_no_mean"]):
        ax_right.annotate(f"{value:.2f}", (i, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, color=INK_SECONDARY)

    fig.suptitle("Behavioural suppression with an intact private assessment", x=0.09, ha="left", fontsize=11, fontweight="semibold")
    counts = " · ".join(f"{c} n={int(n)}" for c, n in zip(order, sub["n_turns"]))
    _note(ax_right, f"infeasible turns only · bars on the left carry Wilson 95% CIs · {counts}")
    return save(fig, data.paths.figures_dir, "fig6_dissociation")


# ---------------------------------------------------------------------------
# 7. Transfer
# ---------------------------------------------------------------------------
def fig7_transfer(data: FigureData) -> Path | None:
    """Heldout algorithmic-world transfer beside the verified-world LOSO baseline."""
    import matplotlib.pyplot as plt

    if data.transfer.empty:
        return None
    apply_style()
    states = list(data.transfer["state"])
    transfer_auc = list(data.transfer["auc"])
    baseline = []
    for state in states:
        key = "desolation" if state == "desolation" else "confusion"
        best = data.summary.get("best", {}).get(key, {})
        baseline.append(float(best.get("auc_mean", np.nan)))

    x = np.arange(len(states))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(x - width / 2, baseline, width, color=MUTED, label="verified worlds (LOSO)", zorder=2)
    ax.bar(x + width / 2, transfer_auc, width, color=SERIES[0], label="heldout g01–g03 (transfer)", zorder=2)
    ax.axhline(0.5, color=GRID, linewidth=1.0, zorder=1)

    for xi, value in zip(x - width / 2, baseline):
        if np.isfinite(value):
            ax.annotate(f"{value:.2f}", (xi, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, color=INK_SECONDARY)
    for xi, value in zip(x + width / 2, transfer_auc):
        ax.annotate(f"{value:.2f}", (xi, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, color=INK_SECONDARY)

    ax.set_xticks(x, states)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("AUC")
    ax.set_xlabel("state")
    ax.set_title("Transfer across flaw type: false premise → algorithmic compromise")
    ax.legend(loc="lower right")
    ns = " · ".join(f"{s} n_test={int(n)}" for s, n in zip(states, data.transfer["n_test"]))
    _note(ax, f"heldout worlds never enter training · {ns}")
    return save(fig, data.paths.figures_dir, "fig7_transfer")


# ---------------------------------------------------------------------------
# 8. Behaviour table
# ---------------------------------------------------------------------------
def fig8_behavior_table(data: FigureData) -> Path | None:
    """Stacked behaviour-class composition per cell, with GASLIGHT as the validity row."""
    import matplotlib.pyplot as plt

    if data.behavior.empty:
        return None
    apply_style()
    sub = data.behavior[data.behavior["n_infeasible_turns"] > 0].copy()
    if sub.empty:
        return None
    sub = sub.iloc[::-1]  # first cell at the top
    y = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(sub) + 2.4))
    ax.grid(False)
    ax.xaxis.grid(True)

    left = np.zeros(len(sub))
    for cls in BEHAVIOR_CLASSES:
        values = sub[cls].fillna(0.0).to_numpy()
        ax.barh(
            y,
            values,
            left=left,
            height=0.62,
            color=BEHAVIOR_COLOR[cls],
            label=BEHAVIOR_LABEL[cls],
            edgecolor=SURFACE,
            linewidth=1.6,  # the 2px surface gap between adjacent segments
            zorder=2,
        )
        # Ink on the muted residual segment, white on the saturated hues.
        label_ink = INK if BEHAVIOR_COLOR[cls] == MUTED else "#ffffff"
        for yi, (value, base) in enumerate(zip(values, left)):
            if value >= 0.11:
                ax.text(
                    base + value / 2,
                    yi,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=label_ink,
                    zorder=3,
                )
        left = left + values

    ax.set_yticks(y, [f"{c}  (n={int(n)})" for c, n in zip(sub["cell"], sub["n_infeasible_turns"])])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of infeasible turns")
    ax.set_title("What the model does when the task cannot be done")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28 - 0.02 * len(sub)), ncol=4)

    gas = data.behavior[data.behavior["cell"] == "GASLIGHT"]
    note = "infeasible turns only"
    if len(gas) and "gaslight_concession" in gas.columns and np.isfinite(gas.iloc[0].get("gaslight_concession", np.nan)):
        note += (
            f" · validity control — GASLIGHT concession rate {gas.iloc[0]['gaslight_concession']:.0%} "
            f"(task is solvable throughout)"
        )
    _note(ax, note)
    return save(fig, data.paths.figures_dir, "fig8_behavior_table")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
ALL_FIGURES = (
    fig1_layer_sweep,
    fig2_confusion_4class,
    fig3_specificity_4x4,
    fig4_money_plot,
    fig5_flip_timeline,
    fig6_dissociation,
    fig7_transfer,
    fig8_behavior_table,
)


def render_all(paths: RunPaths) -> dict[str, Path | None]:
    import matplotlib

    matplotlib.use("Agg")
    data = FigureData(paths)
    out: dict[str, Path | None] = {}
    for func in ALL_FIGURES:
        name = func.__name__
        try:
            out[name] = func(data)
        except Exception as exc:  # a missing analysis must not sink the rest
            log.error("%s failed: %s", name, exc, exc_info=True)
            out[name] = None
        if out[name] is None:
            log.warning("%s produced no figure (missing inputs)", name)
    return out


__all__ = ["render_all", "FigureData", "apply_style", "save", "ALL_FIGURES", "SERIES", "STATE_COLOR"]
