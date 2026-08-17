"""The probe suite, implementing ``bank.probe_plan`` verbatim.

Bank-contract source: ``probe_plan`` (features, ``class_map``,
``contrast_directions``, ``probes``, ``specificity_battery``) and ``taxonomy``.

Seven analyses, each answering a specific objection to the headline claim:

1. **Contrast directions** — mass-mean differences, per world then averaged
   leave-one-world-out.  Desolation gets both the pressure-matched (MASK − GRIND)
   and impossibility-matched (MASK − FUT) contrast, reported together, because
   either one alone is confounded with the thing the other controls for.
2. **One-vs-rest probes** — ``StandardScaler`` → ``LogisticRegression``, folds
   grouped by world, with a 100-run within-world label-permutation null.  The null
   is what turns "AUC 0.8" into a claim.
3. **Four-class multinomial** — the same features, one confusion matrix.
4. **Within-item centered transition probes** — subtract each item's own pre-flip
   mean, then score per turn.  This is the topic-leakage killer: an item is
   compared only against itself, so nothing a probe picks up can be the world's
   vocabulary.
5. **Turn-counter control** — the same centered machinery asked to predict *turn
   index* on TED.  If that succeeds, the transition probes were reading the clock.
6. **Specificity battery** — every direction scored on its own flip and the other
   three, as a 4x4 matrix.  A direction that moves on all four flips is a
   generic-salience detector, not a state probe.
7. **Heldout transfer** — frozen probes scored on the algorithmic worlds, which
   change the *flaw type* (false-premise → epistemic compromise) as well as the
   surface. These worlds never enter training.

Estimators, scalers and cross-validation splitters all come from scikit-learn;
nothing here reimplements them.  Training uses the six verified worlds only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .bank import HELDOUT_SPLIT, TRAIN_SPLIT, Bank
from .config import ProbeConfig, RunPaths
from .rollout import load_view
from .schemas import PROBE_CLASSES

log = logging.getLogger(__name__)

#: The four transition cells, paired with the direction each one is supposed to move.
TRANSITION_OF: dict[str, str] = {
    "excitement": "T_SPARK",
    "confusion": "T_CLARIFY",
    "futility": "T_REVEAL",
    "desolation": "T_REVEAL_HI",
}

#: Contrast directions, as ``name -> (positive cells, negative cells, min_turn)``.
#: ``min_turn`` implements the ``t >= 1`` restriction the bank puts on the
#: desolation contrasts (MASK t0 carries no pressure yet and is labelled
#: futility_free, so including it would blunt the very thing being measured).
CONTRASTS: dict[str, tuple[tuple[str, ...], tuple[str, ...], int]] = {
    "excitement": (("EXC",), ("TED",), 0),
    "confusion": (("CON",), ("TED",), 0),
    "futility": (("FUT",), ("TED",), 0),
    "desolation_primary": (("MASK",), ("GRIND",), 1),
    "desolation_secondary": (("MASK",), ("FUT",), 1),
}

#: Fresh-instance ("no dialogue history") variants of the same contrasts.
FRESH_CONTRASTS: dict[str, tuple[str, str]] = {
    "excitement_fresh": ("FRESH_ENGAGING", "FRESH_TEDIOUS"),
    "confusion_fresh": ("FRESH_AMBIGUOUS", "FRESH_TEDIOUS"),
    "futility_fresh": ("FRESH_IMPOSSIBLE", "FRESH_TEDIOUS"),
}


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------
@dataclass
class Features:
    """One view's pooled residuals plus its aligned index."""

    view: str
    X: np.ndarray  # (N, n_layers, hidden)
    index: Any  # pandas DataFrame, one row per X row

    @property
    def n_layers(self) -> int:
        return self.X.shape[1]

    def layer(self, layer: int) -> np.ndarray:
        return self.X[:, layer, :]

    def mask(self, **eq: Any) -> np.ndarray:
        m = np.ones(len(self.index), dtype=bool)
        for key, value in eq.items():
            col = self.index[key]
            m &= col.isin(value).to_numpy() if isinstance(value, (list, tuple, set)) else (col == value).to_numpy()
        return m


def load_features(paths: RunPaths, views: Sequence[str], kind: str = "canonical") -> dict[str, Features]:
    out: dict[str, Features] = {}
    for view in views:
        X, index = load_view(paths, view, kind=kind)
        if X.size == 0:
            log.warning("no %s activations for view %s", kind, view)
            continue
        finite = np.isfinite(X).all(axis=(1, 2))
        if not finite.all():
            log.info("view %s: dropping %d rows with non-finite pooling", view, int((~finite).sum()))
            X, index = X[finite], index[finite].reset_index(drop=True)
        out[view] = Features(view=view, X=X, index=index)
    return out


# ---------------------------------------------------------------------------
# Class assignment (probe_plan.class_map)
# ---------------------------------------------------------------------------
def probe_class(bank: Bank, state_label: str) -> str | None:
    """Map a per-turn state label to its probe class, or ``None`` if excluded."""
    mapped = bank.probe_plan["class_map"].get(state_label)
    return mapped if mapped in PROBE_CLASSES else None


def training_mask(bank: Bank, feats: Features) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mask, y, groups)`` for the verified-world training palette."""
    mapped = [probe_class(bank, s) for s in feats.index["state_label"]]
    labels = np.array(["" if m is None else m for m in mapped], dtype="<U32")
    mask = (labels != "") & (feats.index["split"] == TRAIN_SPLIT).to_numpy()
    return mask, labels[mask], feats.index.loc[mask, "world"].to_numpy()


# ---------------------------------------------------------------------------
# 1. Contrast directions
# ---------------------------------------------------------------------------
def _cell_mean(feats: Features, world: str, cells: Sequence[str], min_turn: int) -> np.ndarray | None:
    m = feats.mask(world=world, cell=list(cells)) & (feats.index["turn"] >= min_turn).to_numpy()
    m &= (feats.index["split"] == TRAIN_SPLIT).to_numpy()
    return feats.X[m].mean(axis=0) if m.any() else None


def contrast_directions(feats: Features, worlds: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    """Per-world mass-mean differences, plus the LOSO average for each held-out world.

    Returns ``{name: {"per_world": (n_worlds, L, H), "loso": {world: (L, H)}}}``
    where ``loso[w]`` is the direction to use *when scoring world w* — i.e. the mean
    over every other world.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, (pos, neg, min_turn) in CONTRASTS.items():
        per_world: dict[str, np.ndarray] = {}
        for world in worlds:
            a = _cell_mean(feats, world, pos, min_turn)
            b = _cell_mean(feats, world, neg, min_turn)
            if a is not None and b is not None:
                per_world[world] = a - b
        if not per_world:
            continue
        loso = {}
        for world in per_world:
            others = [v for w, v in per_world.items() if w != world]
            if others:
                loso[world] = np.mean(others, axis=0)
        out[name] = {
            "per_world": per_world,
            "loso": loso,
            "pooled": np.mean(list(per_world.values()), axis=0),
        }
    return out


def fresh_directions(fresh: Features, worlds: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The ``probe_plan`` "fresh version" of each direction: bare turn-1 contexts."""
    out: dict[str, dict[str, Any]] = {}
    for name, (pos, neg) in FRESH_CONTRASTS.items():
        per_world: dict[str, np.ndarray] = {}
        for world in worlds:
            pm = fresh.mask(world=world, cell=pos)
            nm = fresh.mask(world=world, cell=neg)
            if pm.any() and nm.any():
                per_world[world] = fresh.X[pm].mean(axis=0) - fresh.X[nm].mean(axis=0)
        if per_world:
            out[name] = {
                "per_world": per_world,
                "pooled": np.mean(list(per_world.values()), axis=0),
            }
    return out


# ---------------------------------------------------------------------------
# 2-3. Supervised probes
# ---------------------------------------------------------------------------
def _pipeline(cfg: ProbeConfig):
    """``StandardScaler`` -> ``LogisticRegression``, per ``probe_plan.probes``.

    The same pipeline serves the binary one-vs-rest probes and the four-class
    multinomial: scikit-learn's ``LogisticRegression`` fits a softmax model when
    handed more than two classes, so nothing needs to differ between them.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, random_state=cfg.random_state),
    )


def _loso_auc(cfg: ProbeConfig, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[dict[str, Any]]:
    """Leave-one-world-out AUC for a binary target."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut

    rows: list[dict[str, Any]] = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        held = str(groups[test_idx][0])
        y_test = y[test_idx]
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y_test)) < 2:
            rows.append({"fold": held, "auc": np.nan, "n_pos": int(y_test.sum()), "n_neg": int((~y_test.astype(bool)).sum())})
            continue
        model = _pipeline(cfg).fit(X[train_idx], y[train_idx])
        score = model.predict_proba(X[test_idx])[:, 1]
        rows.append(
            {
                "fold": held,
                "auc": float(roc_auc_score(y_test, score)),
                "n_pos": int(y_test.sum()),
                "n_neg": int((~y_test.astype(bool)).sum()),
            }
        )
    return rows


def _permute_within_groups(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels *inside* each world, preserving per-world class composition.

    A global shuffle would break the world/class coupling and give an optimistic
    null; permuting within world keeps every world's marginal intact so the null
    answers the right question — can a probe separate these classes *given* that
    the world composition is what it is?
    """
    out = y.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = rng.permutation(y[m])
    return out


def one_vs_rest(
    cfg: ProbeConfig,
    bank: Bank,
    views: dict[str, Features],
    n_jobs: int = -1,
) -> tuple[Any, Any]:
    """Per-state layer sweep with a permutation null.

    Returns ``(scores_df, null_df)``.  The null is estimated on a stride-sampled
    subset of layers and pooled per ``(state, view)``: under label permutation the
    null distribution depends on sample size and fold structure, not on which layer
    the features came from, so pooling buys a tighter estimate of the same quantity
    for a fraction of the compute.  The stride is recorded in the output.
    """
    import pandas as pd
    from joblib import Parallel, delayed

    score_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []

    for view_name, feats in views.items():
        mask, y_class, groups = training_mask(bank, feats)
        X_all = feats.X[mask]
        layers = range(0, feats.n_layers, cfg.layer_stride)
        null_layers = list(range(0, feats.n_layers, max(cfg.layer_stride, 6)))
        log.info(
            "one-vs-rest [%s]: n=%d over %d worlds, %d layers",
            view_name,
            len(y_class),
            len(np.unique(groups)),
            len(list(layers)),
        )

        for state in PROBE_CLASSES:
            y = (y_class == state).astype(int)
            if y.sum() == 0:
                continue
            for layer in layers:
                for row in _loso_auc(cfg, X_all[:, layer, :], y, groups):
                    score_rows.append(
                        {"analysis": "one_vs_rest", "state": state, "view": view_name, "layer": layer, **row}
                    )

            def _null_once(layer: int, seed: int) -> float:
                rng = np.random.default_rng(seed)
                y_perm = _permute_within_groups(y, groups, rng)
                aucs = [r["auc"] for r in _loso_auc(cfg, X_all[:, layer, :], y_perm, groups)]
                return float(np.nanmean(aucs))

            jobs = [
                (layer, cfg.random_state + 1000 * layer + p)
                for layer in null_layers
                for p in range(cfg.n_permutations)
            ]
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_null_once)(layer, seed) for layer, seed in jobs
            )
            for (layer, _), auc in zip(jobs, results):
                null_rows.append(
                    {"analysis": "one_vs_rest", "state": state, "view": view_name, "layer": layer, "auc": auc}
                )

    return pd.DataFrame(score_rows), pd.DataFrame(null_rows)


def multiclass_confusion(
    cfg: ProbeConfig, bank: Bank, feats: Features, layer: int
) -> tuple[np.ndarray, list[str], float]:
    """LOSO four-class multinomial confusion matrix at one layer."""
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import LeaveOneGroupOut

    mask, y, groups = training_mask(bank, feats)
    X = feats.X[mask][:, layer, :]
    classes = [c for c in PROBE_CLASSES if c in set(y)]

    y_true: list[str] = []
    y_pred: list[str] = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = _pipeline(cfg).fit(X[train_idx], y[train_idx])
        y_true.extend(y[test_idx])
        y_pred.extend(model.predict(X[test_idx]))

    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else float("nan")
    return cm, classes, accuracy


# ---------------------------------------------------------------------------
# 4-5. Within-item centered analyses
# ---------------------------------------------------------------------------
def centered_scores(
    feats: Features,
    directions: dict[str, dict[str, Any]],
    cells: Sequence[str],
    flip_turn: int,
    layer: int,
) -> Any:
    """Per-turn projection onto each direction, after subtracting the item's own
    pre-flip mean.

    Centering per item is what makes this a test of *time-indexed state* rather than
    prompt topic: the world's vocabulary, the task, the system prompt and the
    speaker are all held fixed inside an item, so whatever survives centering is
    something that changed at the flip.

    Note the one asymmetry the baseline introduces: pre-flip rows help compute the
    mean they are then measured against, so their residuals are shrunk relative to
    post-flip ones.  That biases the *magnitude* of pre-flip scores toward zero, not
    their sign or their mean, and a projection onto a fixed direction stays centred
    at zero under it.  :func:`turn_counter_control` is the check on whether it
    matters, and its null is built to carry the same asymmetry.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    sel = feats.mask(cell=list(cells))
    index = feats.index[sel].reset_index(drop=True)
    X = feats.X[sel][:, layer, :]

    for uid, group in index.groupby("uid"):
        rows_idx = group.index.to_numpy()
        turns = group["turn"].to_numpy()
        pre = rows_idx[turns < flip_turn]
        if len(pre) == 0:
            continue
        baseline = X[pre].mean(axis=0)
        world = str(group["world"].iloc[0])
        for r, turn in zip(rows_idx, turns):
            delta = X[r] - baseline
            row = {
                "uid": uid,
                "world": world,
                "cell": str(group["cell"].iloc[0]),
                "split": str(group["split"].iloc[0]),
                "turn": int(turn),
                "post_flip": bool(turn >= flip_turn),
                "layer": layer,
                "view": feats.view,
            }
            for name, payload in directions.items():
                # LOSO: score a world with the direction built from the others.
                vec = payload.get("loso", {}).get(world, payload.get("pooled"))
                if vec is None:
                    continue
                v = vec[layer]
                norm = float(np.linalg.norm(v)) or 1.0
                row[f"score_{name}"] = float(np.dot(delta, v) / norm)
            rows.append(row)
    return pd.DataFrame(rows)


def _center_by_assignment(
    X: np.ndarray, index: Any, is_pre: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre each item on the mean of the rows its ``is_pre`` flags select."""
    centered, y, groups = [], [], []
    for _, group in index.groupby("uid"):
        idx = group.index.to_numpy()
        pre = idx[is_pre[idx]]
        if len(pre) == 0 or len(pre) == len(idx):
            continue
        baseline = X[pre].mean(axis=0)
        for r in idx:
            centered.append(X[r] - baseline)
            y.append(int(not is_pre[r]))
            groups.append(str(group["world"].iloc[0]))
    if not centered:
        return np.zeros((0, X.shape[1])), np.zeros(0, dtype=int), np.zeros(0, dtype=object)
    return np.asarray(centered), np.asarray(y), np.asarray(groups)


def turn_counter_control(
    cfg: ProbeConfig, feats: Features, layer: int, flip_turn: int = 2
) -> dict[str, Any]:
    """Centered probe predicting ``t >= flip_turn`` on TED — must sit at its null.

    TED never changes state, so a centered probe that can still tell late turns from
    early ones means the transition probes' apparent signal is partly a clock.

    **The null has to be built the same way the observed statistic is.**  Centering
    on the pre-flip mean makes pre-flip residuals structurally different from
    post-flip ones — a row that helped compute its own baseline has a shrunken
    residual — so a plain label shuffle would sit at 0.5 and score that arithmetic
    artifact as a finding.  Instead each permutation re-draws *which turns count as
    pre-flip* within each item and rebuilds the centering from that draw, so the
    artifact appears in the null too and only genuine turn-ordering information
    clears it.
    """
    sel = feats.mask(cell="TED") & (feats.index["split"] == TRAIN_SPLIT).to_numpy()
    index = feats.index[sel].reset_index(drop=True)
    X = feats.X[sel][:, layer, :]
    if not len(index):
        return {"auc": float("nan"), "folds": [], "null": [], "null_p95": float("nan")}

    turns = index["turn"].to_numpy()
    observed_pre = turns < flip_turn
    Xc, y, groups = _center_by_assignment(X, index, observed_pre)
    if not len(y):
        return {"auc": float("nan"), "folds": [], "null": [], "null_p95": float("nan")}
    folds = _loso_auc(cfg, Xc, y, groups)
    observed = float(np.nanmean([f["auc"] for f in folds]))

    def _null_once(seed: int) -> float:
        rng = np.random.default_rng(seed)
        shuffled = observed_pre.copy()
        for _, group in index.groupby("uid"):
            idx = group.index.to_numpy()
            shuffled[idx] = rng.permutation(observed_pre[idx])
        Xn, yn, gn = _center_by_assignment(X, index, shuffled)
        if not len(yn):
            return float("nan")
        return float(np.nanmean([f["auc"] for f in _loso_auc(cfg, Xn, yn, gn)]))

    from joblib import Parallel, delayed

    if cfg.n_permutations < 20:
        log.warning(
            "turn-counter null uses only %d permutations; the tail is under-resolved and "
            "the p-value is coarse (>= 100 recommended)",
            cfg.n_permutations,
        )
    null = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_null_once)(cfg.random_state + 7919 * p) for p in range(cfg.n_permutations)
    )
    null = [v for v in null if np.isfinite(v)]
    # A permutation p-value rather than a p95 threshold: with six worlds the null is
    # heavy-tailed and its 95th percentile is essentially its maximum, so a
    # threshold verdict flips on a single draw.  The +1 correction keeps the
    # p-value valid at finite permutation counts.
    p_value = (1 + sum(1 for v in null if v >= observed)) / (1 + len(null)) if null else float("nan")
    return {
        "auc": observed,
        "folds": folds,
        "null": null,
        "n_permutations": len(null),
        "null_p95": float(np.percentile(null, 95)) if null else float("nan"),
        "null_median": float(np.median(null)) if null else float("nan"),
        "p_value": float(p_value),
        "exceeds_null": bool(null and p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# 6. Specificity battery
# ---------------------------------------------------------------------------
def specificity_matrix(
    feats: Features, directions: dict[str, dict[str, Any]], layer: int, flip_turn: int = 2
) -> Any:
    """The 4x4 cross-transition matrix: every direction on every flip.

    Cell ``(d, f)`` is the mean post-flip-minus-pre-flip projection of direction
    ``d`` on transition ``f``.  A state probe should own its diagonal and stay flat
    off it; a generic salience or surprise detector lights up the whole row.
    """
    import pandas as pd

    direction_names = {
        "excitement": "excitement",
        "confusion": "confusion",
        "futility": "futility",
        "desolation": "desolation_primary",
    }
    rows: list[dict[str, Any]] = []
    for flip_state, cell in TRANSITION_OF.items():
        scores = centered_scores(feats, directions, [cell], flip_turn, layer)
        if scores.empty:
            continue
        post = scores[scores["post_flip"]]
        for dir_state, key in direction_names.items():
            col = f"score_{key}"
            if col not in post.columns:
                continue
            values = post[col].to_numpy()
            rows.append(
                {
                    "direction": dir_state,
                    "flip_cell": cell,
                    "flip_state": flip_state,
                    "layer": layer,
                    "view": feats.view,
                    "mean_shift": float(np.nanmean(values)),
                    "sem": float(np.nanstd(values) / max(np.sqrt(len(values)), 1.0)),
                    "n": int(len(values)),
                    "own_flip": bool(dir_state == flip_state),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Heldout transfer
# ---------------------------------------------------------------------------
#: Transfer targets: the two states the heldout cells can express.
TRANSFER_TASKS: dict[str, tuple[tuple[str, ...], tuple[str, ...], int]] = {
    "futility": (("FUT",), ("TED",), 0),
    "desolation": (("MASK",), ("GRIND",), 1),
}


def transfer_eval(
    cfg: ProbeConfig, feats: Features, layer: int
) -> list[dict[str, Any]]:
    """Train on the verified worlds, score the algorithmic ones.

    The heldout worlds change the *flaw type* — a false premise becomes an
    epistemic compromise, where the code runs and silently returns the wrong answer.
    A probe that transfers across that gap is tracking something more general than
    "the prompt mentions a missing field".
    """
    from sklearn.metrics import roc_auc_score

    rows: list[dict[str, Any]] = []
    split = feats.index["split"].to_numpy()
    turn = feats.index["turn"].to_numpy()
    cell = feats.index["cell"].to_numpy()

    for name, (pos, neg, min_turn) in TRANSFER_TASKS.items():
        in_task = np.isin(cell, list(pos) + list(neg)) & (turn >= min_turn)
        y = np.isin(cell, list(pos)).astype(int)

        train = in_task & (split == TRAIN_SPLIT)
        test = in_task & (split == HELDOUT_SPLIT)
        if train.sum() == 0 or test.sum() == 0 or len(np.unique(y[test])) < 2:
            log.warning("transfer %s: insufficient data (train=%d test=%d)", name, train.sum(), test.sum())
            continue

        model = _pipeline(cfg).fit(feats.X[train][:, layer, :], y[train])
        score = model.predict_proba(feats.X[test][:, layer, :])[:, 1]
        rows.append(
            {
                "analysis": "transfer",
                "state": name,
                "view": feats.view,
                "layer": layer,
                "auc": float(roc_auc_score(y[test], score)),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def best_layer_view(scores: Any, state: str) -> tuple[int, str, float, float]:
    """``(layer, view, mean_auc, sd)`` maximising mean LOSO AUC for one state."""
    sub = scores[(scores["analysis"] == "one_vs_rest") & (scores["state"] == state)]
    if sub.empty:
        return 0, "pre_reply", float("nan"), float("nan")
    agg = sub.groupby(["layer", "view"])["auc"].agg(["mean", "std"]).reset_index()
    row = agg.loc[agg["mean"].idxmax()]
    return int(row["layer"]), str(row["view"]), float(row["mean"]), float(row["std"])


def run_probes(cfg: ProbeConfig, paths: RunPaths, bank: Bank, n_jobs: int = -1) -> dict[str, Any]:
    """Run every analysis and persist the results.  Returns a summary dict."""
    import pandas as pd

    paths.probes_dir.mkdir(parents=True, exist_ok=True)
    views = load_features(paths, cfg.views, kind="canonical")
    if not views:
        raise RuntimeError("no canonical activations found — run `silentdoubt rollout` first")
    fresh = load_features(paths, cfg.views, kind="fresh")

    worlds = sorted(
        set(views[next(iter(views))].index.loc[lambda d: d["split"] == TRAIN_SPLIT, "world"])
    )
    log.info("probe training worlds: %s", ", ".join(worlds))

    # -- 1. directions --------------------------------------------------------
    directions = {v: contrast_directions(feats, worlds) for v, feats in views.items()}
    fresh_dirs = {v: fresh_directions(f, worlds) for v, f in fresh.items()}
    _save_directions(paths, directions, fresh_dirs)

    # -- 2. one-vs-rest + nulls ----------------------------------------------
    scores, nulls = one_vs_rest(cfg, bank, views, n_jobs=n_jobs)
    scores.to_parquet(paths.probes_dir / "probe_results.parquet", index=False)
    nulls.to_parquet(paths.probes_dir / "nulls.parquet", index=False)

    summary: dict[str, Any] = {"worlds": worlds, "views": list(views), "best": {}}
    for state in PROBE_CLASSES:
        layer, view, mean, sd = best_layer_view(scores, state)
        null_sub = nulls[(nulls["state"] == state) & (nulls["view"] == view)]["auc"]
        summary["best"][state] = {
            "layer": layer,
            "view": view,
            "auc_mean": mean,
            "auc_sd": sd,
            "null_p95": float(np.nanpercentile(null_sub, 95)) if len(null_sub) else float("nan"),
            "null_median": float(np.nanmedian(null_sub)) if len(null_sub) else float("nan"),
        }

    # A single (layer, view) is used for every analysis that needs one, so the
    # transition, specificity and transfer results are all directly comparable.
    anchor_layer = summary["best"]["desolation"]["layer"]
    anchor_view = summary["best"]["desolation"]["view"]
    summary["anchor"] = {"layer": anchor_layer, "view": anchor_view}
    anchor = views[anchor_view]

    # -- 3. multinomial -------------------------------------------------------
    cm, classes, accuracy = multiclass_confusion(cfg, bank, anchor, anchor_layer)
    np.savez(paths.probes_dir / "multiclass.npz", confusion=cm, classes=np.array(classes))
    summary["multiclass"] = {"accuracy": accuracy, "classes": classes, "layer": anchor_layer, "view": anchor_view}

    # -- 4. transitions -------------------------------------------------------
    transition_frames = []
    for state, cell in TRANSITION_OF.items():
        df = centered_scores(anchor, directions[anchor_view], [cell], flip_turn=2, layer=anchor_layer)
        if not df.empty:
            df["flip_state"] = state
            transition_frames.append(df)
    transitions = pd.concat(transition_frames, ignore_index=True) if transition_frames else pd.DataFrame()
    transitions.to_parquet(paths.probes_dir / "transitions.parquet", index=False)

    # -- 5. turn-counter control ---------------------------------------------
    summary["turn_counter_control"] = turn_counter_control(cfg, anchor, anchor_layer)

    # -- 6. specificity -------------------------------------------------------
    spec = specificity_matrix(anchor, directions[anchor_view], anchor_layer)
    spec.to_parquet(paths.probes_dir / "specificity.parquet", index=False)

    # -- 7. transfer ----------------------------------------------------------
    # Both transfer tasks run at the anchor (layer, view) — the best desolation
    # probe — so the two AUCs and the LOSO baseline they are compared against all
    # come from the same frozen representation, with no per-task layer shopping.
    transfer_rows = transfer_eval(cfg, anchor, anchor_layer)
    transfer = pd.DataFrame(transfer_rows)
    transfer.to_parquet(paths.probes_dir / "transfer.parquet", index=False)
    summary["transfer"] = transfer_rows

    (paths.probes_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def _save_directions(
    paths: RunPaths,
    directions: dict[str, dict[str, dict[str, Any]]],
    fresh_dirs: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Persist every direction, keyed ``view/name/{pooled,loso:world,world:world}``."""
    arrays: dict[str, np.ndarray] = {}
    for view, by_name in directions.items():
        for name, payload in by_name.items():
            arrays[f"{view}/{name}/pooled"] = payload["pooled"]
            for world, vec in payload.get("loso", {}).items():
                arrays[f"{view}/{name}/loso:{world}"] = vec
            for world, vec in payload.get("per_world", {}).items():
                arrays[f"{view}/{name}/world:{world}"] = vec
    for view, by_name in fresh_dirs.items():
        for name, payload in by_name.items():
            arrays[f"{view}/{name}/pooled"] = payload["pooled"]
            for world, vec in payload.get("per_world", {}).items():
                arrays[f"{view}/{name}/world:{world}"] = vec
    np.savez_compressed(paths.probes_dir / "directions.npz", **arrays)


__all__ = [
    "Features",
    "load_features",
    "probe_class",
    "training_mask",
    "contrast_directions",
    "fresh_directions",
    "one_vs_rest",
    "multiclass_confusion",
    "centered_scores",
    "turn_counter_control",
    "specificity_matrix",
    "transfer_eval",
    "best_layer_view",
    "run_probes",
    "CONTRASTS",
    "TRANSITION_OF",
]
