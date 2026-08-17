"""Typed records for the whole pipeline.

Bank-contract source: ``loader_contract.expansion``, ``turn_semantics`` and
``taxonomy`` in ``silent_states_bank.json``.  Every field here is either carried
verbatim from the bank or derived by :mod:`silentdoubt.bank`; nothing is invented.

Turn indexing follows ``turn_semantics`` exactly: turns are 0-indexed,
``t = 0 .. n_turns-1``; at turn ``t`` we measure on the canonical messages, then
generate reply ``t``, then append ``feedback_turns[t]``.  Evidence visible at the
measurement of turn ``t`` is therefore ``feedback_turns[0..t-1]``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Views captured per (item, turn, layer).  See probe_plan.features.
# ---------------------------------------------------------------------------
VIEWS: tuple[str, ...] = ("pre_reply", "reply_mean", "reply_last")

#: The four probe classes plus the auxiliary labels, from ``taxonomy``.
PROBE_CLASSES: tuple[str, ...] = ("excitement", "neutral", "confusion", "desolation")


@dataclass(frozen=True)
class Item:
    """One expanded ``world x cell`` trajectory specification.

    Produced by :func:`silentdoubt.bank.expand`, implementing
    ``loader_contract.expansion``.
    """

    item_id: str
    world: str
    cell: str
    split: str  # "train" (six verified worlds) | "heldout_eval" (g01-g03)

    system_prompt: str
    turn_1_prompt: str
    feedback_turns: list[str]

    n_turns: int
    state_schedule: list[str]
    feasible_schedule: list[bool]
    ambiguous_schedule: list[bool]
    feedback_profile: list[dict[str, str]]

    pressure: str
    turn1_variant: str
    flip_turn: int | None = None

    # Carried from the world so labelling never needs the raw bank again.
    markers: dict[str, list[str]] = field(default_factory=dict)
    capitulation_signature: dict[str, str] | None = None

    # Optional replicate index for tier-P2 extra seeds (temp 0.7).
    seed_index: int = 0
    temperature: float = 0.0

    @property
    def uid(self) -> str:
        """Unique per (item, seed) — the row key everywhere downstream."""
        return self.item_id if self.seed_index == 0 else f"{self.item_id}#s{self.seed_index}"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnRecord:
    """One completed (item, turn): the reply plus pointers into the stores.

    Serialised as a single line of ``transcripts.jsonl`` — this file is the
    resume checkpoint for :mod:`silentdoubt.rollout`.
    """

    uid: str
    item_id: str
    world: str
    cell: str
    split: str
    turn: int
    seed_index: int

    # Ground truth for this turn, straight off the cell schedules.
    state_label: str
    feasible: bool
    ambiguous: bool

    context: str  # fully templated measurement context (ends with generation prompt)
    reply: str

    battery: dict[str, Any] = field(default_factory=dict)
    act_ref: dict[str, Any] | None = None  # {"shard": str, "row": int}

    def to_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class Measurement:
    """One battery readout, one row of ``battery.parquet``."""

    uid: str
    item_id: str
    world: str
    cell: str
    turn: int
    key: str  # e.g. "fo_feasible"
    value: float | str  # P(yes), E[0-9], or prefill text
    option: str | None = None  # which option `value` refers to, when categorical
    raw_logit: float | None = None
    raw_mass: float | None = None


@dataclass
class Labels:
    """Behavioural coding of one reply.  Bank-contract source: ``markers``,
    ``capitulation_signature``, ``feasible_schedule``, ``taxonomy``."""

    uid: str
    item_id: str
    world: str
    cell: str
    turn: int
    split: str
    state_label: str
    feasible: bool
    ambiguous: bool

    ack_impossible: bool = False
    ack_ambiguous: bool = False
    capitulated: bool = False
    behavior_class: str | None = None  # spoke_up|hedged_capitulation|silent_capitulation|refused_flaw
    private_public_dissociation: bool | None = None
    gaslight_concession: bool | None = None

    # offline judge (Claude API), reconciled against the marker hits
    judge_impossible: bool | None = None
    judge_ambiguity: bool | None = None
    judge_marker_agree: bool | None = None

    marker_hits: list[str] = field(default_factory=list)
    exclusion_voids: list[str] = field(default_factory=list)
    n_code_blocks: int = 0
    reply_chars: int = 0

    #: Diagnostic only.  The pre-registered ``capitulation_signature.target`` is
    #: ``"code"``, so no metric reads this — it records whether the signature also
    #: fires in prose (inline backticked accessors), which tells the operator
    #: whether the code-only scope is under-counting capitulation.
    signature_in_text: bool = False
