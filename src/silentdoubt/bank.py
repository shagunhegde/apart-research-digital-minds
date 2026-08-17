"""Load, merge, expand and validate the silent-states bank.

Bank-contract source: ``loader_contract.expansion``, ``loader_contract.validation_invariants``
and ``extension_recipe`` in ``silent_states_bank.json``.

Expansion is implemented exactly as written in the contract::

    item.turn_1_prompt  = worlds[w].prompts[cells[c].turn1_variant]
    item.system_prompt  = worlds[w].system_prompt
    item.feedback_turns = [fill(tpl, resolve_slot) for tpl in cells[c].feedback_template]

with slot precedence low -> high::

    worlds[w].slots  ->  bank.item_defaults["{item_id}_{slot}"]  ->  items[i].overrides[slot]

``fill()`` is regex token replacement, never ``str.format`` — world prompts contain
literal dict braces (``{'user_id': 492, ...}``) that ``str.format`` would eat.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

from .schemas import Item

# ---------------------------------------------------------------------------
# fill(): regex token replacement.  A `{slot}` is a *bare identifier* in braces;
# anything else (a dict literal, an f-string-looking fragment) is left alone.
# ---------------------------------------------------------------------------
SLOT_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def fill(template: str, resolve: Callable[[str], str | None]) -> str:
    """Replace ``{slot}`` tokens using ``resolve``; leave unknown tokens intact.

    ``re.sub`` with a function replacement performs no backslash processing on the
    returned string, so slot values containing ``\\`` or ``\\1`` survive verbatim.
    """

    def _sub(m: re.Match[str]) -> str:
        value = resolve(m.group(1))
        return m.group(0) if value is None else value

    return SLOT_RE.sub(_sub, template)


def unfilled_slots(text: str) -> list[str]:
    """Slot tokens still present after expansion (invariant #7 must see none)."""
    return SLOT_RE.findall(text)


class BankError(RuntimeError):
    """A validation invariant from ``loader_contract`` was violated."""


# ---------------------------------------------------------------------------
# Loading + merging
# ---------------------------------------------------------------------------
#: Cells the heldout algorithmic worlds are allowed to run.  Locked decision:
#: g01-g03 run TED, FUT, MASK, GRIND only and never enter probe training.
DEFAULT_HELDOUT_CELLS: tuple[str, ...] = ("TED", "FUT", "MASK", "GRIND")

TRAIN_SPLIT = "train"
HELDOUT_SPLIT = "heldout_eval"


class Bank:
    """The merged bank: verified worlds (train) + algorithmic worlds (heldout_eval)."""

    def __init__(
        self,
        raw: dict[str, Any],
        extension: dict[str, Any] | None = None,
        heldout_cells: Iterable[str] = DEFAULT_HELDOUT_CELLS,
    ) -> None:
        self.raw = raw
        self.cells: dict[str, Any] = raw["cells"]
        self.elicitation: dict[str, Any] = raw["elicitation"]
        self.thresholds: dict[str, float] = raw["thresholds"]
        self.probe_plan: dict[str, Any] = raw["probe_plan"]
        self.taxonomy: dict[str, Any] = raw["taxonomy"]
        self.item_defaults: dict[str, Any] = raw.get("item_defaults", {})
        self.heldout_cells = tuple(heldout_cells)

        self.worlds: dict[str, Any] = dict(raw["worlds"])
        self.split_of: dict[str, str] = {w: TRAIN_SPLIT for w in self.worlds}

        self._item_specs: list[dict[str, Any]] = [dict(i) for i in raw["items"]]

        if extension:
            for wid, world in extension.items():
                if wid in self.worlds:
                    raise BankError(f"extension world {wid!r} collides with a bank world")
                self.worlds[wid] = world
                self.split_of[wid] = HELDOUT_SPLIT
                for cell in self.heldout_cells:
                    if cell not in self.cells:
                        raise BankError(f"heldout cell {cell!r} is not in the bank")
                    self._item_specs.append({"item_id": f"{wid}_{cell}", "world": wid, "cell": cell})

    # -- construction -------------------------------------------------------
    @classmethod
    def load(
        cls,
        bank_path: str | Path,
        extension_path: str | Path | None = None,
        heldout_cells: Iterable[str] = DEFAULT_HELDOUT_CELLS,
    ) -> "Bank":
        raw = json.loads(Path(bank_path).read_text())
        ext = json.loads(Path(extension_path).read_text()) if extension_path else None
        return cls(raw, ext, heldout_cells)

    # -- slot resolution ----------------------------------------------------
    def _resolver(self, item_id: str, world_id: str, overrides: dict[str, Any]) -> Callable[[str], str | None]:
        world_slots = self.worlds[world_id].get("slots", {})

        def resolve(slot: str) -> str | None:
            # low -> high: world slots, then bank item_defaults, then item overrides.
            value: Any = world_slots.get(slot)
            default_key = f"{item_id}_{slot}"
            if default_key in self.item_defaults:
                value = self.item_defaults[default_key]
            if slot in overrides:  # non-slot annotation keys (e.g. "note") never match
                value = overrides[slot]
            return None if value is None else str(value)

        return resolve

    # -- expansion ----------------------------------------------------------
    def expand(self, cells: Iterable[str] | None = None, worlds: Iterable[str] | None = None) -> list[Item]:
        """Expand every declared item into a fully-materialised :class:`Item`."""
        keep_cells = set(cells) if cells is not None else None
        keep_worlds = set(worlds) if worlds is not None else None

        items: list[Item] = []
        for spec in self._item_specs:
            cell_id, world_id = spec["cell"], spec["world"]
            if keep_cells is not None and cell_id not in keep_cells:
                continue
            if keep_worlds is not None and world_id not in keep_worlds:
                continue

            cell = self.cells[cell_id]
            world = self.worlds[world_id]
            overrides = spec.get("overrides", {}) or {}
            resolve = self._resolver(spec["item_id"], world_id, overrides)

            items.append(
                Item(
                    item_id=spec["item_id"],
                    world=world_id,
                    cell=cell_id,
                    split=self.split_of[world_id],
                    system_prompt=world["system_prompt"],
                    turn_1_prompt=world["prompts"][cell["turn1_variant"]],
                    feedback_turns=[fill(t, resolve) for t in cell["feedback_template"]],
                    n_turns=int(cell["n_turns"]),
                    state_schedule=list(cell["state_schedule"]),
                    feasible_schedule=[bool(x) for x in cell["feasible_schedule"]],
                    ambiguous_schedule=[bool(x) for x in cell["ambiguous_schedule"]],
                    feedback_profile=[dict(p) for p in cell["feedback_profile"]],
                    pressure=cell["pressure"],
                    turn1_variant=cell["turn1_variant"],
                    flip_turn=cell.get("flip_turn"),
                    markers=dict(world.get("markers", {})),
                    capitulation_signature=world.get("capitulation_signature"),
                )
            )
        return items

    # -- validation ---------------------------------------------------------
    #: ``loader_contract.validation_invariants`` #2, verbatim.
    TURN1_IDENTITY_GROUPS: dict[str, tuple[str, ...]] = {
        "tedious": ("TED", "GRIND", "T_SPARK", "T_REVEAL", "T_REVEAL_HI", "GASLIGHT"),
        "impossible": ("FUT", "MASK", "T_RELIEF"),
        "ambiguous": ("CON", "T_CLARIFY"),
    }

    def validate(self, items: list[Item], accept_unverified: bool = False) -> list[str]:
        """Enforce every ``validation_invariants`` entry plus the spec's extras.

        Returns the list of warning lines (unverified fields).  Raises
        :class:`BankError` on any hard violation.
        """
        errors: list[str] = []

        by_world_variant: dict[tuple[str, str], set[str]] = {}
        for it in items:
            # 1. schedule lengths
            if len(it.feedback_turns) != it.n_turns - 1:
                errors.append(f"{it.item_id}: len(feedback_turns)={len(it.feedback_turns)} != n_turns-1={it.n_turns-1}")
            for name in ("state_schedule", "feasible_schedule", "ambiguous_schedule"):
                if len(getattr(it, name)) != it.n_turns:
                    errors.append(f"{it.item_id}: len({name})={len(getattr(it, name))} != n_turns={it.n_turns}")
            if len(it.feedback_profile) != it.n_turns - 1:
                errors.append(f"{it.item_id}: len(feedback_profile) != n_turns-1")

            # 2. turn_1 identity groups per world
            for variant, group in self.TURN1_IDENTITY_GROUPS.items():
                if it.cell in group:
                    if it.turn1_variant != variant:
                        errors.append(f"{it.item_id}: cell {it.cell} must use prompts.{variant}, got {it.turn1_variant}")
                    by_world_variant.setdefault((it.world, variant), set()).add(it.turn_1_prompt)

            # 3. GRIND carries zero factual assertions
            if it.cell == "GRIND":
                bad = [i for i, p in enumerate(it.feedback_profile) if p.get("assertion") != "none"]
                if bad:
                    errors.append(f"{it.item_id}: GRIND feedback_profile has assertions at turns {bad}")

            # 4. capitulation_signature required wherever the task is infeasible
            if any(f is False for f in it.feasible_schedule):
                sig = it.capitulation_signature
                if not sig or not sig.get("pattern"):
                    errors.append(f"{it.item_id}: infeasible turns present but no capitulation_signature")
                else:
                    try:
                        re.compile(sig["pattern"])
                    except re.error as exc:
                        errors.append(f"{it.item_id}: capitulation_signature does not compile: {exc}")
                    if sig.get("target") != "code":
                        errors.append(f"{it.item_id}: capitulation_signature.target must be 'code'")

            # 5. all markers are multi-word.  The invariant reads "all markers are
            # multi-word; exclusions honored on code-stripped text only" — the
            # multi-word rule scopes to the ack markers, whose false-positive risk
            # it exists to control.  Exclusions are deliberately allowed to be
            # single-word ("hypothetically") because they only ever *void* a hit.
            for kind in ("ack_impossible", "ack_ambiguous"):
                for m in it.markers.get(kind, []):
                    if " " not in m.strip():
                        errors.append(f"{it.world}: marker {kind}={m!r} is single-word")

            # 7. no unfilled {slot} placeholders survive expansion
            for i, txt in enumerate(it.feedback_turns):
                left = unfilled_slots(txt)
                if left:
                    errors.append(f"{it.item_id}: unfilled slots {left} in feedback_turns[{i}]")

            # heldout worlds must be marked as such
            expected = HELDOUT_SPLIT if self.split_of[it.world] == HELDOUT_SPLIT else TRAIN_SPLIT
            if it.split != expected:
                errors.append(f"{it.item_id}: split {it.split!r} != {expected!r}")
            if it.split == HELDOUT_SPLIT and it.cell not in self.heldout_cells:
                errors.append(f"{it.item_id}: heldout world running non-heldout cell {it.cell}")

        # 2 (continued): the shared turn-1 prompt must literally be one string per (world, variant)
        for (world_id, variant), prompts in by_world_variant.items():
            if len(prompts) > 1:
                errors.append(f"{world_id}/{variant}: cells in one identity group disagree on turn_1_prompt")

        if errors:
            # World-level faults surface once per item that carries the world;
            # report each distinct fault once, in first-seen order.
            unique = list(dict.fromkeys(errors))
            raise BankError("bank validation failed:\n  - " + "\n  - ".join(unique))

        return self._unverified_banner(items, accept_unverified)

    # 8. verified:false fields require human sign-off before GPU runs
    def _unverified_banner(self, items: list[Item], accept_unverified: bool) -> list[str]:
        #: which world-level ``verified`` key each cell depends on
        cell_deps: dict[str, tuple[str, ...]] = {
            "EXC": ("engaging",),
            "TED": ("tedious",),
            "CON": ("ambiguous",),
            "FUT": ("impossible",),
            "GRIND": ("tedious",),
            "MASK": ("impossible",),
            "T_SPARK": ("tedious",),
            "T_CLARIFY": ("ambiguous",),
            "T_REVEAL": ("tedious",),
            "T_REVEAL_HI": ("tedious",),
            "T_RELIEF": ("impossible", "relief_fact"),
            "GASLIGHT": ("tedious", "gaslight_claim"),
        }
        lines: list[str] = []
        for it in sorted(items, key=lambda x: x.item_id):
            verified = self.worlds[it.world].get("verified", {})
            missing = [k for k in cell_deps.get(it.cell, ()) if verified.get(k) is not True]
            if missing:
                lines.append(f"{it.item_id}: unverified {', '.join(missing)}")
        if lines and not accept_unverified:
            banner = (
                "UNVERIFIED BANK FIELDS REQUIRE HUMAN SIGN-OFF "
                "(loader_contract.validation_invariants #8).\n"
                + "\n".join(f"    {line}" for line in lines)
                + "\n  Re-run with --accept-unverified once signed off."
            )
            raise BankError(banner)
        for line in lines:
            warnings.warn(f"[unverified, accepted by operator] {line}", stacklevel=2)
        return lines


# ---------------------------------------------------------------------------
# Tokenizer invariant (validation_invariants #6) — runs without a GPU.
# ---------------------------------------------------------------------------
def option_variants(word: str) -> list[str]:
    """Bare and leading-space forms, in both casings, per the spec's §3 note."""
    forms = [word, word[:1].upper() + word[1:]]
    return [f for w in forms for f in (w, " " + w)]


def first_token_ids(tokenizer: Any, word: str) -> list[int]:
    """Deduped first-token ids across all variants of ``word``.

    Logit reads in §5 sum probability mass over exactly this set.
    """
    ids: list[int] = []
    for variant in option_variants(word):
        enc = tokenizer.encode(variant, add_special_tokens=False)
        if enc and enc[0] not in ids:
            ids.append(enc[0])
    return ids


#: The option sets whose members must be first-token-separable.
def option_sets(bank: Bank) -> dict[str, list[str]]:
    return {
        "state_options": list(bank.elicitation["state_options"]),
        "yes_no": ["yes", "no"],
        "ab": ["A", "B"],
        "digits": [str(d) for d in range(10)],
    }


def assert_tokenizer_invariant(bank: Bank, tokenizer: Any) -> dict[str, dict[str, list[int]]]:
    """Hard-fail unless every option set is first-token separable.

    Checks the strong form the variant-summed readout actually needs: the id sets
    of two distinct options in the same set must be *disjoint*, in both bare and
    leading-space (and capitalised) forms.
    """
    table: dict[str, dict[str, list[int]]] = {}
    problems: list[str] = []
    for set_name, options in option_sets(bank).items():
        ids = {opt: first_token_ids(tokenizer, opt) for opt in options}
        table[set_name] = ids
        for opt, opt_ids in ids.items():
            if not opt_ids:
                problems.append(f"{set_name}/{opt!r}: no first token")
        names = list(ids)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                shared = set(ids[a]) & set(ids[b])
                if shared:
                    problems.append(f"{set_name}: {a!r} and {b!r} share first token ids {sorted(shared)}")
    if problems:
        raise BankError(
            "tokenizer invariant failed (loader_contract.validation_invariants #6):\n  - "
            + "\n  - ".join(problems)
        )
    return table


__all__ = [
    "Bank",
    "BankError",
    "fill",
    "unfilled_slots",
    "first_token_ids",
    "option_variants",
    "option_sets",
    "assert_tokenizer_invariant",
    "DEFAULT_HELDOUT_CELLS",
    "TRAIN_SPLIT",
    "HELDOUT_SPLIT",
]
