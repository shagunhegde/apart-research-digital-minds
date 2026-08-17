"""silentdoubt — a callable, resumable harness for the silent-states v3.0 bank.

The package is deliberately import-light: nothing here pulls in torch, nnsight or
matplotlib, so ``import silentdoubt`` stays fast on a CPU box that only needs the
bank loader or the analysis layer.  Heavy modules import their own dependencies at
call time.

Module map, and which part of the bank each one implements:

===================  ====================================================
:mod:`~.bank`        ``loader_contract`` — load, merge, expand, validate
:mod:`~.schemas`     typed records for every stage
:mod:`~.config`      operational settings (tiers, budget, batch sizes)
:mod:`~.modelio`     the nnsight subject-model wrapper and readout layer
:mod:`~.gates`       ``loader_contract.gates``
:mod:`~.rollout`     ``turn_semantics`` + ``elicitation`` (the turn loop)
:mod:`~.labels`      ``markers`` / ``capitulation_signature`` / ``taxonomy``
:mod:`~.probes`      ``probe_plan``
:mod:`~.analysis`    aggregation and the reported statistics
:mod:`~.figures`     the eight figures
:mod:`~.report`      ``report.md``, including ``framing_note``
:mod:`~.cli`         the ``silentdoubt`` command
===================  ====================================================
"""

from __future__ import annotations

__version__ = "0.1.0"

from .bank import Bank, BankError
from .config import RunConfig, RunPaths
from .schemas import PROBE_CLASSES, VIEWS, Item, Labels, Measurement, TurnRecord

__all__ = [
    "__version__",
    "Bank",
    "BankError",
    "RunConfig",
    "RunPaths",
    "Item",
    "TurnRecord",
    "Measurement",
    "Labels",
    "PROBE_CLASSES",
    "VIEWS",
]
