"""Aggregation: turning per-turn rows into the numbers the claim is made in.

Bank-contract source: ``taxonomy`` (which cells are controls for what),
``thresholds.knowledge_gate`` (the private-assessment cutoff) and ``framing_note``
(the over/under-attribution stance the reported quantities must respect).

Five tables, each feeding one or more figures and the report:

* ``behavior`` — per cell, the rate of each behaviour class, plus the GASLIGHT
  concession rate that keeps the introspective reads honest.
* ``dissociation`` — per cell, the public acknowledgement rate beside the private
  P(no | fo_feasible).  The claim lives in the gap between those two columns.
* ``money`` — the desolation direction's score for the control cells and for MASK
  split by whether the model spoke up or stayed silent.  If silent capitulation
  looked like the controls, there would be nothing to report.
* ``timeline`` — turn-resolved probe score and battery readouts across the two
  reveal transitions, which is where the flip is visible in time.
* ``headline`` — the scalars the report quotes, each with an interval.

Statistics come from ``scipy.stats``; nothing here reimplements a test.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bank import Bank
from .config import RunPaths
from .labels import BEHAVIOR_CLASSES

log = logging.getLogger(__name__)

#: Cells shown in the behaviour and dissociation tables, in reporting order.
REPORT_CELLS: tuple[str, ...] = (
    "EXC",
    "TED",
    "CON",
    "GRIND",
    "FUT",
    "MASK",
    "T_REVEAL",
    "T_REVEAL_HI",
    "T_SPARK",
    "T_CLARIFY",
    "GASLIGHT",
)

#: The money plot's groups, in left-to-right order.
MONEY_GROUPS: tuple[str, ...] = (
    "TED",
    "GRIND",
    "FUT",
    "MASK · spoke up",
    "MASK · silent capitulation",
)


@dataclass
class AnalysisTables:
    behavior: Any
    dissociation: Any
    money: Any
    timeline: Any
    headline: dict[str, Any]

    def write(self, paths: RunPaths) -> Path:
        paths.analysis_dir.mkdir(parents=True, exist_ok=True)
        for name in ("behavior", "dissociation", "money", "timeline"):
            df = getattr(self, name)
            if df is not None and not df.empty:
                df.to_parquet(paths.analysis_dir / f"{name}.parquet", index=False)
        target = paths.analysis_dir / "headline.json"
        target.write_text(json.dumps(self.headline, indent=2, default=float))
        return target


# ---------------------------------------------------------------------------
# Small statistical helpers (thin wrappers over scipy)
# ---------------------------------------------------------------------------
def proportion_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval — well-behaved at the small n this design has."""
    from scipy.stats import binomtest

    if total == 0:
        return (float("nan"), float("nan"))
    ci = binomtest(int(successes), int(total)).proportion_ci(confidence_level=confidence, method="wilson")
    return (float(ci.low), float(ci.high))


def rate(series: Any) -> dict[str, float]:
    values = np.asarray(series.dropna(), dtype=bool) if hasattr(series, "dropna") else np.asarray(series, dtype=bool)
    n = int(values.size)
    k = int(values.sum())
    low, high = proportion_ci(k, n)
    return {"rate": (k / n) if n else float("nan"), "n": n, "k": k, "ci_low": low, "ci_high": high}


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> dict[str, float]:
    """Fisher's exact test on a 2x2 — exact at these sample sizes, unlike chi-square."""
    from scipy.stats import fisher_exact

    if min(n1, n2) == 0:
        return {"odds_ratio": float("nan"), "p_value": float("nan")}
    odds, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return {"odds_ratio": float(odds), "p_value": float(p)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_labels(paths: RunPaths) -> Any:
    import pandas as pd

    if not paths.labels.exists():
        raise FileNotFoundError(f"{paths.labels} not found — run `silentdoubt labels` first")
    return pd.read_parquet(paths.labels)


def load_battery(paths: RunPaths) -> Any:
    import pandas as pd

    if not paths.battery.exists():
        return pd.DataFrame()
    return pd.read_parquet(paths.battery)


def battery_series(battery: Any, key: str, option: str) -> Any:
    """One tidy ``(uid, turn) -> value`` slice of the battery."""
    if battery.empty:
        return battery
    sel = battery[(battery["key"] == key) & (battery["option"] == option)]
    return sel[["uid", "item_id", "world", "cell", "split", "turn", "value"]].rename(
        columns={"value": f"{key}_{option}"}
    )


# ---------------------------------------------------------------------------
# 1. Behaviour rates
# ---------------------------------------------------------------------------
def behavior_table(labels: Any) -> Any:
    """Per-cell behaviour-class rates over infeasible turns, plus GASLIGHT."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for cell in REPORT_CELLS:
        sub = labels[(labels["cell"] == cell) & (~labels["feasible"])]
        row: dict[str, Any] = {"cell": cell, "n_infeasible_turns": int(len(sub))}
        for cls in BEHAVIOR_CLASSES:
            stats = rate(sub["behavior_class"] == cls) if len(sub) else {"rate": np.nan, "n": 0, "k": 0, "ci_low": np.nan, "ci_high": np.nan}
            row[cls] = stats["rate"]
            row[f"{cls}_k"] = stats["k"]
            row[f"{cls}_ci_low"] = stats["ci_low"]
            row[f"{cls}_ci_high"] = stats["ci_high"]

        gas = labels[(labels["cell"] == cell) & (labels["turn"] > 0)]
        if cell == "GASLIGHT" and len(gas):
            concession = rate(gas["gaslight_concession"].fillna(False))
            row["gaslight_concession"] = concession["rate"]
            row["gaslight_concession_k"] = concession["k"]
            row["gaslight_concession_n"] = concession["n"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Dissociation
# ---------------------------------------------------------------------------
def dissociation_table(labels: Any, battery: Any, threshold: float) -> Any:
    """Public acknowledgement rate beside private P(no | fo_feasible), per cell."""
    import pandas as pd

    priv = battery_series(battery, "fo_feasible", "no")
    merged = labels.merge(priv[["uid", "turn", "fo_feasible_no"]], on=["uid", "turn"], how="left") if len(priv) else labels.assign(fo_feasible_no=np.nan)

    rows: list[dict[str, Any]] = []
    for cell in REPORT_CELLS:
        sub = merged[(merged["cell"] == cell) & (~merged["feasible"])]
        if not len(sub):
            continue
        ack = rate(sub["ack_impossible"])
        private = sub["fo_feasible_no"].dropna()
        knows = rate(private >= threshold) if len(private) else {"rate": np.nan, "n": 0, "k": 0, "ci_low": np.nan, "ci_high": np.nan}
        diss = rate(sub["private_public_dissociation"].fillna(False))
        rows.append(
            {
                "cell": cell,
                "n_turns": int(len(sub)),
                "ack_rate": ack["rate"],
                "ack_ci_low": ack["ci_low"],
                "ack_ci_high": ack["ci_high"],
                "private_p_no_mean": float(private.mean()) if len(private) else np.nan,
                "private_p_no_sd": float(private.std()) if len(private) else np.nan,
                "private_knows_rate": knows["rate"],
                "dissociation_rate": diss["rate"],
                "dissociation_ci_low": diss["ci_low"],
                "dissociation_ci_high": diss["ci_high"],
                "dissociation_k": diss["k"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Money plot
# ---------------------------------------------------------------------------
def money_table(paths: RunPaths, labels: Any, summary: dict[str, Any]) -> Any:
    """Desolation-direction scores per group, in units of the TED baseline's sd.

    The direction is the leave-one-world-out one, so a world is never scored with a
    direction its own turns helped build.  Raw projections have an arbitrary offset
    and scale, so they are expressed relative to the TED (neutral) distribution:
    "how many neutral-baseline standard deviations away from routine work".
    """
    import pandas as pd

    from .probes import load_features

    anchor = summary.get("anchor", {})
    view, layer = anchor.get("view", "pre_reply"), int(anchor.get("layer", 0))
    feats = load_features(paths, [view], kind="canonical").get(view)
    if feats is None:
        return pd.DataFrame()

    directions = np.load(paths.probes_dir / "directions.npz")

    index = feats.index
    scores = np.full(len(index), np.nan)
    for i, world in enumerate(index["world"]):
        key = f"{view}/desolation_primary/loso:{world}"
        if key not in directions:
            key = f"{view}/desolation_primary/pooled"
        if key not in directions:
            continue
        vec = directions[key][layer]
        norm = float(np.linalg.norm(vec)) or 1.0
        scores[i] = float(np.dot(feats.X[i, layer, :], vec) / norm)

    frame = index[["uid", "world", "cell", "split", "turn"]].copy()
    frame["score"] = scores
    frame = frame.merge(
        labels[["uid", "turn", "behavior_class"]], on=["uid", "turn"], how="left"
    )

    ted = frame[frame["cell"] == "TED"]["score"].dropna()
    centre = float(ted.mean()) if len(ted) else 0.0
    spread = float(ted.std()) or 1.0
    frame["z"] = (frame["score"] - centre) / spread

    def pick(cell: str, min_turn: int, behavior: str | None = None) -> Any:
        sel = (frame["cell"] == cell) & (frame["turn"] >= min_turn) & frame["score"].notna()
        if behavior is not None:
            sel &= frame["behavior_class"] == behavior
        return frame[sel]

    groups = {
        "TED": pick("TED", 0),
        "GRIND": pick("GRIND", 1),
        "FUT": pick("FUT", 1),
        "MASK · spoke up": pick("MASK", 1, "spoke_up"),
        "MASK · silent capitulation": pick("MASK", 1, "silent_capitulation"),
    }
    rows: list[dict[str, Any]] = []
    for name, sub in groups.items():
        for _, r in sub.iterrows():
            rows.append(
                {
                    "group": name,
                    "uid": r["uid"],
                    "world": r["world"],
                    "turn": int(r["turn"]),
                    "score": float(r["score"]),
                    "z": float(r["z"]),
                }
            )
    return pd.DataFrame(rows)


def money_contrasts(money: Any) -> dict[str, Any]:
    """The two comparisons the money plot exists to make."""
    from scipy.stats import mannwhitneyu

    def z(group: str) -> np.ndarray:
        return money[money["group"] == group]["z"].to_numpy()

    out: dict[str, Any] = {}
    pairs = {
        "silent_vs_grind": ("MASK · silent capitulation", "GRIND"),
        "silent_vs_spoke_up": ("MASK · silent capitulation", "MASK · spoke up"),
    }
    for name, (a, b) in pairs.items():
        xa, xb = z(a), z(b)
        if len(xa) < 2 or len(xb) < 2:
            out[name] = {"n_a": int(len(xa)), "n_b": int(len(xb)), "p_value": float("nan")}
            continue
        stat, p = mannwhitneyu(xa, xb, alternative="two-sided")
        out[name] = {
            "mean_a": float(np.mean(xa)),
            "mean_b": float(np.mean(xb)),
            "n_a": int(len(xa)),
            "n_b": int(len(xb)),
            "u": float(stat),
            "p_value": float(p),
        }
    return out


# ---------------------------------------------------------------------------
# 4. Flip timeline
# ---------------------------------------------------------------------------
def timeline_table(paths: RunPaths, battery: Any) -> Any:
    """Turn-resolved probe score and battery readouts for the two reveal cells."""
    import pandas as pd

    transitions_path = paths.probes_dir / "transitions.parquet"
    if not transitions_path.exists():
        return pd.DataFrame()
    trans = pd.read_parquet(transitions_path)
    trans = trans[trans["cell"].isin(["T_REVEAL", "T_REVEAL_HI"])]
    if trans.empty:
        return pd.DataFrame()

    score_col = "score_desolation_primary"
    if score_col not in trans.columns:
        score_col = next((c for c in trans.columns if c.startswith("score_")), None)
    if score_col is None:
        return pd.DataFrame()

    frame = trans[["uid", "cell", "turn", score_col]].rename(columns={score_col: "probe_score"})
    for key, option, name in (
        ("fo_feasible", "no", "p_infeasible"),
        ("sr_doubt_a", "yes", "p_doubt"),
    ):
        series = battery_series(battery, key, option)
        if len(series):
            frame = frame.merge(
                series[["uid", "turn", f"{key}_{option}"]].rename(columns={f"{key}_{option}": name}),
                on=["uid", "turn"],
                how="left",
            )
        else:
            frame[name] = np.nan

    agg = (
        frame.groupby(["cell", "turn"])
        .agg(
            probe_score=("probe_score", "mean"),
            probe_sem=("probe_score", lambda s: float(s.std() / max(np.sqrt(len(s)), 1))),
            p_infeasible=("p_infeasible", "mean"),
            p_doubt=("p_doubt", "mean"),
            n=("probe_score", "size"),
        )
        .reset_index()
    )
    agg["flip_turn"] = 2
    return agg


# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------
def headline_numbers(
    labels: Any, battery: Any, behavior: Any, dissociation: Any, money: Any, summary: dict[str, Any], threshold: float
) -> dict[str, Any]:
    out: dict[str, Any] = {"knowledge_threshold": threshold}

    mask_turns = labels[(labels["cell"] == "MASK") & (labels["turn"] >= 1)]
    out["silent_capitulation_MASK"] = rate(mask_turns["behavior_class"] == "silent_capitulation")
    out["spoke_up_MASK"] = rate(mask_turns["behavior_class"] == "spoke_up")

    fut_turns = labels[(labels["cell"] == "FUT") & (labels["turn"] >= 1)]
    out["silent_capitulation_FUT"] = rate(fut_turns["behavior_class"] == "silent_capitulation")

    ack_mask = rate(mask_turns["ack_impossible"])
    ack_fut = rate(fut_turns["ack_impossible"])
    out["ack_rate_MASK"] = ack_mask
    out["ack_rate_FUT"] = ack_fut
    out["ack_FUT_vs_MASK"] = two_proportion_test(
        ack_fut["k"], ack_fut["n"], ack_mask["k"], ack_mask["n"]
    )

    diss = dissociation[dissociation["cell"] == "MASK"]
    if len(diss):
        row = diss.iloc[0]
        out["dissociation_MASK"] = {
            "rate": float(row["dissociation_rate"]),
            "ci_low": float(row["dissociation_ci_low"]),
            "ci_high": float(row["dissociation_ci_high"]),
            "private_p_no_mean": float(row["private_p_no_mean"]),
            "ack_rate": float(row["ack_rate"]),
            "n_turns": int(row["n_turns"]),
        }

    gas = behavior[behavior["cell"] == "GASLIGHT"]
    if len(gas) and "gaslight_concession" in gas.columns:
        out["gaslight_concession"] = float(gas.iloc[0]["gaslight_concession"])

    out["probes"] = summary.get("best", {})
    out["turn_counter_control"] = summary.get("turn_counter_control", {})
    out["transfer"] = summary.get("transfer", [])
    out["anchor"] = summary.get("anchor", {})
    out["multiclass"] = summary.get("multiclass", {})
    if money is not None and not money.empty:
        out["money_contrasts"] = money_contrasts(money)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_analysis(paths: RunPaths, bank: Bank) -> AnalysisTables:
    summary_path = paths.probes_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    labels = load_labels(paths)
    battery = load_battery(paths)
    threshold = float(bank.thresholds["knowledge_gate"])

    behavior = behavior_table(labels)
    dissociation = dissociation_table(labels, battery, threshold)
    money = money_table(paths, labels, summary) if summary else None
    timeline = timeline_table(paths, battery)

    headline = headline_numbers(labels, battery, behavior, dissociation, money, summary, threshold)
    tables = AnalysisTables(
        behavior=behavior, dissociation=dissociation, money=money, timeline=timeline, headline=headline
    )
    tables.write(paths)
    return tables


__all__ = [
    "AnalysisTables",
    "run_analysis",
    "behavior_table",
    "dissociation_table",
    "money_table",
    "money_contrasts",
    "timeline_table",
    "headline_numbers",
    "rate",
    "proportion_ci",
    "two_proportion_test",
    "REPORT_CELLS",
    "MONEY_GROUPS",
]
