"""The turn loop: trajectories, the per-turn battery, and activation capture.

Bank-contract source: ``turn_semantics`` (verbatim) and ``elicitation`` (every
battery prompt is read from the bank, never restated here).

``turn_semantics`` reads: *turns 0-indexed, t = 0..n_turns-1; at turn t measure on
canonical msgs, generate reply t, then append feedback_turns[t]*.  So the battery and
the ``pre_reply`` capture both see a context **ending in user content, before the
model has replied** — which is the whole point: it is the model's private assessment
at the moment before it decides what to say.  A fact delivered in
``feedback_turns[1]`` is first measurable at t=2, which is why every transition cell
flips at 2.

Execution shape (spec §4, §13):

* Work units are ``(item, seed)`` pairs, grouped into priority tiers.  Tiers run
  **in order**, each as its own lockstep pass, so the budget guard can drop a whole
  tier at a clean boundary instead of abandoning half-built trajectories.
* Within a tier every unit advances together, turn by turn, so batches are full and
  all forks of one measurement are batched into one forward pass.
* One JSONL line per ``(unit, turn)`` is the checkpoint.  Resume replays recorded
  replies to rebuild message state without touching the GPU.
"""

from __future__ import annotations

import json
import logging
import time
import zlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .bank import Bank
from .config import BudgetConfig, RunConfig, RunPaths, Tier
from .modelio import FOLD, Elicitor, SubjectModel
from .schemas import Item, TurnRecord

log = logging.getLogger(__name__)

Message = dict[str, str]

# ---------------------------------------------------------------------------
# Battery layout (spec §5).  Every prompt is looked up in ``bank.elicitation``;
# this table only says *how to read the answer*, never what to ask.
# ---------------------------------------------------------------------------
#: key -> (elicitation key, option set)
YES_NO_ITEMS: tuple[tuple[str, str], ...] = (
    ("fo_feasible", "fo_feasible"),
    ("fo_ambiguous", "fo_ambiguous"),
    ("sr_doubt_a", "sr_doubt_a"),
    ("sr_confused", "sr_confused"),
    ("sr_belief_change", "sr_belief_change"),
    ("self_predict", "self_predict_prompt"),
)
#: sr_doubt_b is a paraphrase-robustness twin of sr_doubt_a and runs on a
#: pre-registered subset only (``elicitation._doc``: 25%).
PARAPHRASE_ITEM = ("sr_doubt_b", "sr_doubt_b")


def in_paraphrase_subset(item_id: str, fraction: float) -> bool:
    """Deterministic, content-addressed subset membership.

    A stable hash of the item id, so the same items are selected on every run and
    on every machine — no RNG state, no ordering dependence.
    """
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    bucket = zlib.crc32(item_id.encode()) % 1000
    return bucket < fraction * 1000


def stable_seed(base: int, uid: str) -> int:
    return (base + zlib.crc32(uid.encode())) % (2**31 - 1)


# ---------------------------------------------------------------------------
# Work units
# ---------------------------------------------------------------------------
@dataclass
class WorkUnit:
    """One trajectory in flight: an item, its seed, and its canonical messages."""

    item: Item
    tier: str
    messages: list[Message] = field(default_factory=list)
    turn: int = 0

    @property
    def uid(self) -> str:
        return self.item.uid

    def start(self) -> None:
        self.messages = [{"role": "user", "content": self.item.turn_1_prompt}]
        self.turn = 0

    def advance(self, reply: str) -> None:
        """Append the reply, then the feedback for the turn just completed."""
        self.messages.append({"role": "assistant", "content": reply})
        if self.turn < len(self.item.feedback_turns):
            self.messages.append({"role": "user", "content": self.item.feedback_turns[self.turn]})
        self.turn += 1

    def active(self) -> bool:
        return self.turn < self.item.n_turns


def build_units(items: Sequence[Item], tiers: Sequence[Tier], bank: Bank) -> list[WorkUnit]:
    """Expand items into tier-ordered work units, deduplicated by uid.

    A tier with ``temperature > 0`` contributes ``seeds`` sampled replicates with
    ``seed_index`` starting at 1, leaving index 0 for the canonical greedy pass.
    """
    by_key = {(it.world, it.cell): it for it in items}
    units: list[WorkUnit] = []
    seen: set[str] = set()

    for tier in tiers:
        for (world, cell), item in sorted(by_key.items()):
            if cell not in tier.cells:
                continue
            if tier.split is not None and item.split != tier.split:
                continue
            if tier.temperature > 0:
                replicates = [
                    replace(item, seed_index=k, temperature=tier.temperature)
                    for k in range(1, tier.seeds + 1)
                ]
            else:
                replicates = [item]
            for rep in replicates:
                if rep.uid in seen:
                    continue
                seen.add(rep.uid)
                units.append(WorkUnit(item=rep, tier=tier.name))
    return units


# ---------------------------------------------------------------------------
# Activation store
# ---------------------------------------------------------------------------
class ActStore:
    """Sharded pooled residuals plus an append-only index.

    Pooling happens on-GPU inside :meth:`SubjectModel.capture`; this class only ever
    sees ``(n, n_layers, hidden)`` fp32 blocks, so full token x layer tensors are
    never written to disk.  Shards keep the store append-only, which is what makes
    a mid-run crash recoverable without recomputation.
    """

    def __init__(self, paths: RunPaths, views: Sequence[str]) -> None:
        self.paths = paths
        self.views = tuple(views)
        self.dir = paths.acts_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = paths.acts_index_jsonl
        self._shard = self._next_shard()

    def _next_shard(self) -> int:
        existing = [p.stem.rsplit("_", 1)[-1] for p in self.dir.glob("*_*.npy")]
        nums = [int(s) for s in existing if s.isdigit()]
        return max(nums) + 1 if nums else 0

    def existing_keys(self) -> set[tuple[str, int, str]]:
        """``(uid, turn, kind)`` already captured — the resume filter."""
        keys: set[tuple[str, int, str]] = set()
        if self.index_path.exists():
            for line in self.index_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                keys.add((row["uid"], int(row["turn"]), row["kind"]))
        return keys

    def append(self, kind: str, rows: Sequence[dict[str, Any]], arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        """Write one shard per view and return the index rows that were logged."""
        if not rows:
            return []
        shard = self._shard
        self._shard += 1
        for view in self.views:
            block = arrays[view]
            if block.shape[0] != len(rows):
                raise ValueError(f"view {view}: {block.shape[0]} rows for {len(rows)} records")
            np.save(self.dir / f"{kind}_{view}_{shard:04d}.npy", block.astype(np.float32))

        logged = []
        with self.index_path.open("a") as fh:
            for i, row in enumerate(rows):
                entry = {**row, "kind": kind, "shard": shard, "row": i}
                fh.write(json.dumps(entry) + "\n")
                logged.append(entry)
        return logged

    def finalize(self) -> Path:
        """Materialise ``acts/index.parquet`` from the append-only JSONL."""
        import pandas as pd

        rows = [
            json.loads(line)
            for line in (self.index_path.read_text().splitlines() if self.index_path.exists() else [])
            if line.strip()
        ]
        df = pd.DataFrame(rows)
        df.to_parquet(self.paths.acts_index_parquet, index=False)
        return self.paths.acts_index_parquet


def load_view(paths: RunPaths, view: str, kind: str = "canonical") -> tuple[np.ndarray, "Any"]:
    """Load one view back as ``(X, index_df)`` with rows aligned.

    Used by :mod:`silentdoubt.probes`; keeps shard bookkeeping in one place.
    """
    import pandas as pd

    index_file = paths.acts_index_parquet if paths.acts_index_parquet.exists() else None
    if index_file is not None:
        idx = pd.read_parquet(index_file)
    else:
        rows = [json.loads(l) for l in paths.acts_index_jsonl.read_text().splitlines() if l.strip()]
        idx = pd.DataFrame(rows)
    idx = idx[idx["kind"] == kind].reset_index(drop=True)

    blocks: list[np.ndarray] = []
    order: list[int] = []
    for shard, group in idx.groupby("shard", sort=True):
        arr = np.load(paths.acts_dir / f"{kind}_{view}_{int(shard):04d}.npy")
        blocks.append(arr[group["row"].to_numpy()])
        order.extend(group.index.tolist())
    if not blocks:
        return np.zeros((0, 0, 0), dtype=np.float32), idx
    X = np.concatenate(blocks, axis=0)
    inv = np.argsort(np.asarray(order))
    return X[inv], idx


# ---------------------------------------------------------------------------
# Budget guard (spec §13)
# ---------------------------------------------------------------------------
class BudgetGuard:
    """Projects remaining wall-clock and drops the lowest tiers first.

    The projection is deliberately crude — a moving average of seconds per
    ``(unit, turn)`` — because the only decision it drives is binary and coarse:
    does the next tier fit before the reserve window opens?  Every drop is logged
    with the numbers that caused it, so the run's coverage is auditable after the
    fact rather than mysterious.
    """

    def __init__(self, budget: BudgetConfig, log_path: Path) -> None:
        self.budget = budget
        self.log_path = log_path
        self.t0 = time.time()
        self.cost_per_unit_turn: float | None = None
        self.events: list[dict[str, Any]] = []

    def elapsed(self) -> float:
        return time.time() - self.t0

    def remaining(self) -> float:
        return self.budget.deadline_seconds - self.elapsed()

    def observe(self, seconds: float, unit_turns: int) -> None:
        if unit_turns <= 0:
            return
        sample = seconds / unit_turns
        self.cost_per_unit_turn = (
            sample if self.cost_per_unit_turn is None else 0.7 * self.cost_per_unit_turn + 0.3 * sample
        )

    def project(self, unit_turns: int) -> float | None:
        if self.cost_per_unit_turn is None:
            return None
        return unit_turns * self.cost_per_unit_turn

    def admits(self, tier: str, unit_turns: int) -> bool:
        """Decide whether ``tier`` fits.  Unknown cost always admits (P0 runs)."""
        if not self.budget.enabled:
            return True
        projected = self.project(unit_turns)
        if projected is None:
            self._log("admit", tier=tier, unit_turns=unit_turns, reason="no cost estimate yet")
            return True
        fits = projected <= self.remaining()
        self._log(
            "admit" if fits else "drop",
            tier=tier,
            unit_turns=unit_turns,
            projected_s=round(projected, 1),
            remaining_s=round(self.remaining(), 1),
            cost_per_unit_turn_s=round(self.cost_per_unit_turn, 2),
        )
        if not fits:
            log.warning(
                "budget guard: dropping tier %s (needs ~%.0f min, %.0f min left)",
                tier,
                projected / 60,
                self.remaining() / 60,
            )
        return fits

    def _log(self, event: str, **fields: Any) -> None:
        row = {"event": event, "elapsed_s": round(self.elapsed(), 1), **fields}
        self.events.append(row)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------
def fold_question(messages: Sequence[Message], question: str) -> list[Message]:
    """Append a battery question to the trailing user turn.

    Folding rather than adding a turn is deliberate and model-independent: it keeps
    the read at the pre-reply measurement point, and it does not depend on whether a
    given chat template tolerates two user messages in a row.
    """
    msgs = [dict(m) for m in messages]
    if not msgs or msgs[-1]["role"] != "user":
        raise ValueError("battery expects a context ending in a user turn (turn_semantics)")
    msgs[-1]["content"] = msgs[-1]["content"] + FOLD + question
    return msgs


class Battery:
    """Runs the per-turn battery for a whole lockstep group in a few passes."""

    def __init__(self, model: SubjectModel, elicitor: Elicitor, bank: Bank, cfg: RunConfig) -> None:
        self.model = model
        self.eli = elicitor
        self.prompts = bank.elicitation
        self.cfg = cfg

    def _ctx(self, unit: WorkUnit, question: str) -> str:
        return self.model.chat.render(unit.item.system_prompt, fold_question(unit.messages, question))

    def run(self, units: Sequence[WorkUnit]) -> list[dict[str, Any]]:
        """One dict of readouts per unit, in unit order."""
        out: list[dict[str, Any]] = [{} for _ in units]
        if not units:
            return out

        # -- round 1: everything that only needs the canonical context ------
        yes_no_plan = list(YES_NO_ITEMS)
        contexts: list[str] = []
        addr: list[tuple[int, str]] = []
        for i, unit in enumerate(units):
            for key, prompt_key in yes_no_plan:
                contexts.append(self._ctx(unit, self.prompts[prompt_key]))
                addr.append((i, key))
            if in_paraphrase_subset(unit.item.item_id, self.cfg.sr_doubt_b_fraction):
                key, prompt_key = PARAPHRASE_ITEM
                contexts.append(self._ctx(unit, self.prompts[prompt_key]))
                addr.append((i, key))
        for (i, key), r in zip(addr, self.eli.yes_no(contexts)):
            out[i][key] = _categorical(r)

        for key, prompt_key, reader in (
            ("sr_state_forced", "sr_state_forced", self.eli.state),
            ("abandon", "abandon_prompt", self.eli.choice),
        ):
            reads = reader([self._ctx(u, self.prompts[prompt_key]) for u in units])
            for i, r in enumerate(reads):
                out[i][key] = _categorical(r)

        valence = self.eli.digit([self._ctx(u, self.prompts["sr_valence"]) for u in units])
        for i, r in enumerate(valence):
            out[i]["sr_valence"] = _numeric(r)

        # -- round 2: the type-2 confidence fork ----------------------------
        # ctx + fo_feasible + the model's own argmax answer + confidence prompt.
        conf_ctx: list[str] = []
        for i, unit in enumerate(units):
            answer = out[i]["fo_feasible"]["argmax"]
            msgs = fold_question(unit.messages, self.prompts["fo_feasible"])
            msgs = msgs + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": self.prompts["confidence_prompt"]},
            ]
            conf_ctx.append(self.model.chat.render(unit.item.system_prompt, msgs))
        for i, r in enumerate(self.eli.digit(conf_ctx)):
            out[i]["confidence"] = _numeric(r)

        # -- round 3: the prefilled assistant turn --------------------------
        prefill = self.prompts["prefill"]
        prefill_ctx = [
            self.model.chat.render(u.item.system_prompt, u.messages, suffix=prefill) for u in units
        ]
        gens = self.model.generate(prefill_ctx, max_new_tokens=self.cfg.model.prefill_max_new_tokens)
        for i, g in enumerate(gens):
            out[i]["prefill"] = {"kind": "text", "seed_text": prefill, "value": g.text}
        return out


def _categorical(readout: Any) -> dict[str, Any]:
    return {
        "kind": "categorical",
        "options": list(readout.options),
        "prob": [float(x) for x in readout.prob],
        "logit": [float(x) for x in readout.logit],
        "mass": [float(x) for x in readout.mass],
        "total_mass": readout.total_mass(),
        "argmax": readout.argmax(),
    }


def _numeric(readout: Any) -> dict[str, Any]:
    return {
        "kind": "numeric",
        "options": list(readout.options),
        "prob": [float(x) for x in readout.prob],
        "value": readout.expectation(range(10)),
        "total_mass": readout.total_mass(),
        "argmax": readout.argmax(),
    }


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
class Rollout:
    """Drives tiers, turns, battery, generation and capture; checkpoints as it goes."""

    def __init__(
        self,
        cfg: RunConfig,
        paths: RunPaths,
        bank: Bank,
        model: SubjectModel,
        elicitor: Elicitor,
    ) -> None:
        self.cfg = cfg
        self.paths = paths.ensure()
        self.bank = bank
        self.model = model
        self.battery = Battery(model, elicitor, bank, cfg)
        self.store = ActStore(paths, cfg.capture_views)
        self.guard = BudgetGuard(cfg.budget, paths.budget_log)
        self.done: dict[tuple[str, int], dict[str, Any]] = self._load_checkpoint()
        self.captured = self.store.existing_keys()
        if self.done:
            log.info("resuming: %d (unit, turn) records already on disk", len(self.done))

    # -- checkpointing ------------------------------------------------------
    def _load_checkpoint(self) -> dict[tuple[str, int], dict[str, Any]]:
        out: dict[tuple[str, int], dict[str, Any]] = {}
        if not self.paths.transcripts.exists():
            return out
        for line in self.paths.transcripts.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out[(row["uid"], int(row["turn"]))] = row
        return out

    def _write(self, record: TurnRecord) -> None:
        with self.paths.transcripts.open("a") as fh:
            fh.write(record.to_line() + "\n")
        self.done[(record.uid, record.turn)] = asdict(record)

    # -- main entry ---------------------------------------------------------
    def run(self, items: Sequence[Item]) -> dict[str, Any]:
        units = build_units(items, self.cfg.tiers, self.bank)
        by_tier: dict[str, list[WorkUnit]] = {}
        for u in units:
            by_tier.setdefault(u.tier, []).append(u)

        summary: dict[str, Any] = {"tiers_run": [], "tiers_dropped": [], "units": len(units)}
        for tier in self.cfg.tiers:
            group = by_tier.get(tier.name, [])
            if not group:
                continue
            pending = sum(
                1
                for u in group
                for t in range(u.item.n_turns)
                if (u.uid, t) not in self.done
            )
            if not self.guard.admits(tier.name, pending):
                summary["tiers_dropped"].append(tier.name)
                continue
            started = time.time()
            self._run_tier(tier.name, group)
            self.guard.observe(time.time() - started, max(pending, 1))
            summary["tiers_run"].append(tier.name)

        if self.cfg.capture_fresh_instances:
            summary["fresh_instances"] = self._capture_fresh(items)

        self.store.finalize()
        summary["battery_rows"] = self.write_battery_parquet()
        summary["records"] = len(self.done)
        return summary

    # -- one tier -----------------------------------------------------------
    def _run_tier(self, name: str, units: list[WorkUnit]) -> None:
        for u in units:
            u.start()
        max_turns = max(u.item.n_turns for u in units)
        log.info("tier %s: %d units, up to %d turns", name, len(units), max_turns)

        for t in range(max_turns):
            live = [u for u in units if u.active() and u.turn == t]
            if not live:
                continue
            fresh = [u for u in live if (u.uid, t) not in self.done]
            replayed = [u for u in live if (u.uid, t) in self.done]

            for u in replayed:  # rebuild state without touching the GPU
                u.advance(self.done[(u.uid, t)]["reply"])

            if fresh:
                started = time.time()
                self._run_turn(fresh, t)
                self.guard.observe(time.time() - started, len(fresh))
                log.info(
                    "tier %s turn %d: %d units in %.1fs (%d replayed)",
                    name,
                    t,
                    len(fresh),
                    time.time() - started,
                    len(replayed),
                )

    # -- one turn across a lockstep group ----------------------------------
    def _run_turn(self, units: list[WorkUnit], t: int) -> None:
        contexts = [self.model.chat.render(u.item.system_prompt, u.messages) for u in units]

        # measure (forked) -> generate -> capture, exactly in that order
        battery = self.battery.run(units)

        gens = self._generate_grouped(units, contexts)

        ctx_ids = [self.model.chat.encode(c) for c in contexts]
        reply_ids = [g.token_ids for g in gens]
        acts = self.model.capture(ctx_ids, reply_ids)

        rows = []
        for i, u in enumerate(units):
            it = u.item
            rows.append(
                {
                    "uid": u.uid,
                    "item_id": it.item_id,
                    "world": it.world,
                    "cell": it.cell,
                    "split": it.split,
                    "turn": t,
                    "seed_index": it.seed_index,
                    "tier": u.tier,
                    "state_label": it.state_schedule[t],
                    "feasible": bool(it.feasible_schedule[t]),
                    "ambiguous": bool(it.ambiguous_schedule[t]),
                    "n_reply_tokens": len(reply_ids[i]),
                }
            )
        logged = self.store.append("canonical", rows, acts)

        for i, u in enumerate(units):
            it = u.item
            record = TurnRecord(
                uid=u.uid,
                item_id=it.item_id,
                world=it.world,
                cell=it.cell,
                split=it.split,
                turn=t,
                seed_index=it.seed_index,
                state_label=it.state_schedule[t],
                feasible=bool(it.feasible_schedule[t]),
                ambiguous=bool(it.ambiguous_schedule[t]),
                context=contexts[i],
                reply=gens[i].text,
                battery={**battery[i], "_truncated": gens[i].truncated, "_tier": u.tier},
                act_ref={"kind": "canonical", "shard": logged[i]["shard"], "row": logged[i]["row"]},
            )
            self._write(record)
            u.advance(gens[i].text)

    def _generate_grouped(self, units: Sequence[WorkUnit], contexts: Sequence[str]) -> list[Any]:
        """Greedy units batch together; sampled units carry per-unit seeds."""
        gens: list[Any] = [None] * len(units)
        greedy = [i for i, u in enumerate(units) if u.item.temperature <= 0]
        if greedy:
            got = self.model.generate(
                [contexts[i] for i in greedy], max_new_tokens=self.cfg.model.reply_max_new_tokens
            )
            for slot, g in zip(greedy, got):
                gens[slot] = g
        for i, u in enumerate(units):
            if u.item.temperature > 0:
                gens[i] = self.model.generate(
                    [contexts[i]],
                    max_new_tokens=self.cfg.model.reply_max_new_tokens,
                    temperature=u.item.temperature,
                    seed=stable_seed(self.cfg.seed, f"{u.uid}:{u.turn}"),
                )[0]
        return gens

    # -- fresh-instance captures (probe_plan "fresh version" directions) ----
    def _capture_fresh(self, items: Sequence[Item]) -> int:
        """Every world x every turn-1 variant, as a bare first turn.

        These are the contexts behind ``probe_plan.contrast_directions``' "fresh
        version" of each direction: no dialogue history, no pressure, just the task.
        """
        variants = ("tedious", "engaging", "ambiguous", "impossible")
        worlds = sorted({it.world for it in items})
        rows: list[dict[str, Any]] = []
        contexts: list[str] = []

        for world_id in worlds:
            world = self.bank.worlds[world_id]
            for variant in variants:
                prompt = world["prompts"].get(variant)
                if prompt is None:
                    continue
                uid = f"fresh::{world_id}::{variant}"
                if (uid, 0, "fresh") in self.captured:
                    continue
                contexts.append(
                    self.model.chat.render(
                        world["system_prompt"], [{"role": "user", "content": prompt}]
                    )
                )
                rows.append(
                    {
                        "uid": uid,
                        "item_id": uid,
                        "world": world_id,
                        "cell": f"FRESH_{variant.upper()}",
                        "split": self.bank.split_of[world_id],
                        "turn": 0,
                        "seed_index": 0,
                        "tier": "fresh",
                        "state_label": variant,
                        "feasible": variant != "impossible",
                        "ambiguous": variant == "ambiguous",
                        "n_reply_tokens": 0,
                    }
                )
        if not rows:
            return 0
        log.info("fresh-instance capture: %d contexts", len(rows))
        acts = self.model.capture([self.model.chat.encode(c) for c in contexts], None)
        self.store.append("fresh", rows, acts)
        return len(rows)

    # -- derived artifact ---------------------------------------------------
    def write_battery_parquet(self) -> int:
        return write_battery_parquet(self.paths)


def write_battery_parquet(paths: RunPaths) -> int:
    """Explode ``transcripts.jsonl`` battery blobs into tidy ``battery.parquet``.

    One row per ``(uid, turn, key, option)`` for categorical reads and one per
    ``(uid, turn, key)`` for numeric/text ones, carrying the raw logits and absolute
    mass alongside the probability (spec §5).
    """
    import pandas as pd

    if not paths.transcripts.exists():
        return 0

    rows: list[dict[str, Any]] = []
    for line in paths.transcripts.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        base = {k: rec[k] for k in ("uid", "item_id", "world", "cell", "split", "turn", "seed_index")}
        base["state_label"] = rec["state_label"]
        for key, payload in rec.get("battery", {}).items():
            if key.startswith("_") or not isinstance(payload, dict):
                continue
            kind = payload.get("kind")
            if kind == "categorical":
                for j, option in enumerate(payload["options"]):
                    rows.append(
                        {
                            **base,
                            "key": key,
                            "kind": kind,
                            "option": option,
                            "value": payload["prob"][j],
                            "raw_logit": payload["logit"][j],
                            "raw_mass": payload["mass"][j],
                            "total_mass": payload["total_mass"],
                            "argmax": payload["argmax"],
                            "text": None,
                        }
                    )
            elif kind == "numeric":
                rows.append(
                    {
                        **base,
                        "key": key,
                        "kind": kind,
                        "option": "E[0-9]",
                        "value": payload["value"],
                        "raw_logit": None,
                        "raw_mass": None,
                        "total_mass": payload["total_mass"],
                        "argmax": payload["argmax"],
                        "text": None,
                    }
                )
            else:
                rows.append(
                    {
                        **base,
                        "key": key,
                        "kind": "text",
                        "option": None,
                        "value": np.nan,
                        "raw_logit": None,
                        "raw_mass": None,
                        "total_mass": None,
                        "argmax": None,
                        "text": payload.get("value"),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_parquet(paths.battery, index=False)
    return len(df)


__all__ = [
    "Rollout",
    "WorkUnit",
    "Battery",
    "ActStore",
    "BudgetGuard",
    "build_units",
    "fold_question",
    "load_view",
    "write_battery_parquet",
    "in_paraphrase_subset",
]
