"""The subject model, wrapped for the three things this design needs from it.

Bank-contract source: ``elicitation`` (what gets read), ``turn_semantics`` (where
the read happens), ``probe_plan.features`` (what gets captured).

**Nothing about the model is reimplemented here.**  nnsight owns tracing and
interleaving, transformers owns the weights, the chat template, tokenisation and
generation, and torch owns the arithmetic.  This module only:

1. renders bank content into the subject's chat format (:class:`ChatFormatter`),
2. issues the three forward passes the design needs — an option-logit read, a
   greedy generation, and a pooled residual capture — through nnsight,
3. hands the results back as numpy.

Three model-specific facts drive the implementation and are worth stating plainly,
because getting any of them wrong silently corrupts the measurements:

* **Gemma-2's chat template raises on a ``system`` role** and enforces strict
  user/assistant alternation.  Every world in the bank carries a system prompt, and
  the battery appends a question to a context that already ends in a user turn.
  :class:`ChatFormatter` detects both constraints once at load and folds the system
  prompt into the first user message / merges same-role neighbours accordingly.
* **Gemma-2 soft-caps its final logits** inside ``Gemma2ForCausalLM.forward``, not
  inside ``lm_head``.  Reading ``lm_head.output`` would give uncapped logits and
  therefore wrong probabilities, so every readout here goes through the model's own
  output object.
* **Left padding plus a plain forward needs explicit ``position_ids``.**  ``generate``
  builds them from the attention mask; a bare ``forward`` does not, and would hand
  pad tokens real positions.  :meth:`SubjectModel._pad` builds them the same way
  transformers does.

Residual-stream capture reads ``model.layers[i]`` output — the post-block residual
stream — at every layer, and pools on-GPU so full token x layer tensors are never
materialised, let alone persisted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .config import ModelConfig

log = logging.getLogger(__name__)

Message = dict[str, str]

#: Separator used when two messages must be folded into one (system -> first user,
#: battery question -> trailing user turn).  A blank line, i.e. a paragraph break.
FOLD = "\n\n"


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------
class ChatFormatter:
    """Renders ``(system, messages)`` into the subject's chat format.

    Capability detection happens once, against the tokenizer's own template, so the
    same code path serves a template that accepts system turns and one that does not.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tok = tokenizer
        self.supports_system = self._probe(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        )
        self.supports_repeat_role = self._probe(
            [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
        )
        self.emits_bos = self._emits_bos()
        log.info(
            "chat template: system=%s repeat_role=%s emits_bos=%s",
            self.supports_system,
            self.supports_repeat_role,
            self.emits_bos,
        )

    # -- capability probes --------------------------------------------------
    def _probe(self, messages: list[Message]) -> bool:
        try:
            self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return True
        except Exception:
            return False

    def _emits_bos(self) -> bool:
        bos = getattr(self.tok, "bos_token", None)
        if not bos:
            return False
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": "u"}], tokenize=False, add_generation_prompt=True
        )
        return text.startswith(bos)

    # -- rendering ----------------------------------------------------------
    def compose(self, system: str | None, messages: Sequence[Message]) -> list[Message]:
        """Fold ``system`` in and collapse same-role neighbours if the template needs it."""
        msgs: list[Message] = [dict(m) for m in messages]
        if system:
            if self.supports_system:
                msgs.insert(0, {"role": "system", "content": system})
            else:
                idx = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
                if idx is None:
                    raise ValueError("no user message to carry the system prompt")
                msgs[idx]["content"] = system + FOLD + msgs[idx]["content"]
        if not self.supports_repeat_role:
            msgs = _merge_same_role(msgs)
        return msgs

    def render(
        self,
        system: str | None,
        messages: Sequence[Message],
        add_generation_prompt: bool = True,
        suffix: str = "",
    ) -> str:
        """Full context string.  ``suffix`` is appended raw, after the generation
        prompt — that is how the ``prefill`` probe seeds the assistant turn."""
        text = self.tok.apply_chat_template(
            self.compose(system, messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return text + suffix

    def encode(self, text: str) -> list[int]:
        """Token ids for a rendered context.

        ``apply_chat_template`` already emitted BOS for templates that want one, so
        the tokenizer must not add a second; templates that emit none get one here.
        """
        ids = self.tok.encode(text, add_special_tokens=False)
        if not self.emits_bos and getattr(self.tok, "bos_token_id", None) is not None:
            ids = [self.tok.bos_token_id] + ids
        return ids


def _merge_same_role(messages: list[Message]) -> list[Message]:
    out: list[Message] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + FOLD + m["content"]}
        else:
            out.append(dict(m))
    return out


# ---------------------------------------------------------------------------
# Readout containers
# ---------------------------------------------------------------------------
@dataclass
class OptionReadout:
    """One first-token categorical read over a fixed option set (spec §5).

    ``prob`` is the *restricted* distribution — softmax over the variant-summed
    option logits, which is what the gates and all reported P(yes)/P(no) use.
    ``mass`` is the absolute full-vocabulary probability the model puts on the
    option set's tokens, kept because a low total mass means the model wanted to say
    something else entirely and the restricted probability is then a thin reed.
    """

    options: list[str]
    logit: np.ndarray  # (n_options,) log-sum-exp over each option's variant ids
    mass: np.ndarray  # (n_options,) absolute probability mass, full-vocab softmax
    prob: np.ndarray  # (n_options,) restricted, sums to 1

    def p(self, option: str) -> float:
        return float(self.prob[self.options.index(option)])

    def total_mass(self) -> float:
        return float(self.mass.sum())

    def argmax(self) -> str:
        return self.options[int(np.argmax(self.prob))]

    def expectation(self, values: Sequence[float]) -> float:
        return float(np.dot(self.prob, np.asarray(values, dtype=np.float64)))


@dataclass
class Generation:
    """One greedy (or sampled) continuation."""

    text: str
    token_ids: list[int]
    truncated: bool  # hit max_new_tokens without emitting an end-of-turn token


# ---------------------------------------------------------------------------
# Pooling helpers.  These run inside the intervention graph via ``nnsight.apply``,
# on real tensors, which keeps them robust to transformers' decoder-layer return
# type drifting between a bare tensor and a tuple across versions.
# ---------------------------------------------------------------------------
def _resid(layer_output: Any):
    """Post-block residual stream from a decoder layer's output."""
    while isinstance(layer_output, (tuple, list)):
        layer_output = layer_output[0]
    return layer_output


def _pool_views(layer_output: Any, pre_idx, reply_mask, last_idx):
    """The three per-(item, turn, layer) views, pooled on device, returned fp32/CPU.

    Returns ``(B, 3, H)`` ordered ``(pre_reply, reply_mean, reply_last)``.
    """
    import torch

    h = _resid(layer_output).float()
    b = h.shape[0]
    rows = torch.arange(b, device=h.device)

    pre = h[rows, pre_idx]
    last = h[rows, last_idx]

    mask = reply_mask.to(h.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    mean = (h * mask).sum(dim=1) / denom

    return torch.stack([pre, mean, last], dim=1).cpu()


def _read_options(model_output: Any, id_matrix, id_mask):
    """Variant-summed option logits + absolute mass from the model's own logits.

    ``id_matrix`` is ``(n_options, max_variants)`` of token ids, ``id_mask`` the
    matching validity mask (options have different variant counts).  Reading
    ``model_output.logits`` rather than ``lm_head.output`` is what keeps Gemma-2's
    final-logit soft-capping in the numbers.
    """
    import torch

    logits = model_output.logits[:, -1, :].float()  # (B, V), post soft-cap
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # (B, 1)

    picked = logits[:, id_matrix.reshape(-1)].reshape(logits.shape[0], *id_matrix.shape)
    picked = picked.masked_fill(~id_mask.unsqueeze(0), float("-inf"))

    option_logit = torch.logsumexp(picked, dim=-1)  # (B, n_options)
    option_mass = torch.exp(option_logit - lse)  # (B, n_options)
    return option_logit.cpu(), option_mass.cpu()


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class SubjectModel:
    """nnsight-wrapped subject model: read options, generate, capture residuals."""

    def __init__(self, cfg: ModelConfig) -> None:
        import torch
        from nnsight import LanguageModel

        self.cfg = cfg
        self.torch = torch

        load_kwargs: dict[str, Any] = {
            "device_map": cfg.device_map,
            "torch_dtype": getattr(torch, cfg.torch_dtype),
            "attn_implementation": cfg.attn_implementation,
            "dispatch": True,  # nnsight 0.4 keeps the model on meta without this
        }
        if cfg.max_memory:
            load_kwargs["max_memory"] = cfg.max_memory
        if cfg.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        log.info("loading %s (%s, %s)", cfg.name, cfg.torch_dtype, cfg.attn_implementation)
        self.lm = LanguageModel(cfg.name, **load_kwargs)
        self.tokenizer = self.lm.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.chat = ChatFormatter(self.tokenizer)

        self.n_layers = int(self.lm.config.num_hidden_layers)
        self.hidden_size = int(self.lm.config.hidden_size)
        self._device = self._pick_device()
        self._eos_ids = self._end_of_turn_ids()
        log.info(
            "%s ready: %d layers x %d dims, device=%s, stop ids=%s",
            cfg.name,
            self.n_layers,
            self.hidden_size,
            self._device,
            sorted(self._eos_ids),
        )

    # -- device / stopping --------------------------------------------------
    def _pick_device(self):
        try:
            dev = next(self.lm._model.parameters()).device
            return dev
        except Exception:  # pragma: no cover - depends on dispatch layout
            return self.torch.device("cpu")

    def _end_of_turn_ids(self) -> set[int]:
        """Every id that terminates an assistant turn, per the model's own config."""
        ids: set[int] = set()
        eos = getattr(self.lm.config, "eos_token_id", None)
        for v in eos if isinstance(eos, (list, tuple)) else [eos]:
            if isinstance(v, int):
                ids.add(v)
        gen_cfg = getattr(self.lm._model, "generation_config", None)
        gen_eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg else None
        for v in gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos]:
            if isinstance(v, int):
                ids.add(v)
        if self.tokenizer.pad_token_id is not None:
            ids.add(int(self.tokenizer.pad_token_id))
        return ids

    # -- token ids for an option set ---------------------------------------
    def option_id_table(self, options: Sequence[str], variants: Callable[[str], list[int]]) -> tuple[Any, Any]:
        """``(id_matrix, id_mask)`` device tensors for :func:`_read_options`."""
        per_option = [variants(o) for o in options]
        width = max(len(v) for v in per_option)
        matrix = np.zeros((len(options), width), dtype=np.int64)
        mask = np.zeros((len(options), width), dtype=bool)
        for i, ids in enumerate(per_option):
            matrix[i, : len(ids)] = ids
            mask[i, : len(ids)] = True
        t = self.torch
        return (
            t.tensor(matrix, device=self._device),
            t.tensor(mask, device=self._device),
        )

    # -- batching -----------------------------------------------------------
    def _pad(self, sequences: Sequence[Sequence[int]]) -> dict[str, Any]:
        """Left-padded batch with explicit ``position_ids``.

        Position ids are built exactly the way ``prepare_inputs_for_generation``
        does: cumulative count of attended tokens, pads pinned to 1.  Without this a
        plain forward would give pad tokens real positions and shift every real
        token's rotary phase.
        """
        t = self.torch
        width = max(len(s) for s in sequences)
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids = t.full((len(sequences), width), pad_id, dtype=t.long)
        attention_mask = t.zeros((len(sequences), width), dtype=t.long)
        for i, s in enumerate(sequences):
            if not s:
                continue
            input_ids[i, width - len(s) :] = t.tensor(s, dtype=t.long)
            attention_mask[i, width - len(s) :] = 1

        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        return {
            "input_ids": input_ids.to(self._device),
            "attention_mask": attention_mask.to(self._device),
            "position_ids": position_ids.to(self._device),
        }

    # -- (1) option-logit reads --------------------------------------------
    def read_options(
        self,
        contexts: Sequence[str],
        options: Sequence[str],
        id_matrix: Any,
        id_mask: Any,
        batch_size: int | None = None,
    ) -> list[OptionReadout]:
        """First-token categorical read over ``options`` for each context.

        One forward pass per batch; the readout is taken at the final (left-padded)
        position, which is the token the model is about to continue from.
        """
        import nnsight

        bs = batch_size or self.cfg.measure_batch
        out: list[OptionReadout] = []
        opts = list(options)

        for start in range(0, len(contexts), bs):
            chunk = contexts[start : start + bs]
            batch = self._pad([self.chat.encode(c) for c in chunk])
            with self.lm.trace(batch):
                packed = nnsight.apply(_read_options, self.lm.output, id_matrix, id_mask).save()
            logit, mass = packed
            logit = np.asarray(logit, dtype=np.float64)
            mass = np.asarray(mass, dtype=np.float64)
            # restricted softmax over the variant-summed option logits
            shifted = logit - logit.max(axis=1, keepdims=True)
            prob = np.exp(shifted)
            prob /= prob.sum(axis=1, keepdims=True)
            for i in range(len(chunk)):
                out.append(OptionReadout(options=opts, logit=logit[i], mass=mass[i], prob=prob[i]))
        return out

    # -- (2) generation -----------------------------------------------------
    def generate(
        self,
        contexts: Sequence[str],
        max_new_tokens: int,
        temperature: float = 0.0,
        seed: int | None = None,
        batch_size: int | None = None,
    ) -> list[Generation]:
        """Greedy by default; ``temperature > 0`` switches to sampling (tier P2)."""
        bs = batch_size or self.cfg.generate_batch
        results: list[Generation] = []

        gen_kwargs: dict[str, Any] = {"pad_token_id": int(self.tokenizer.pad_token_id)}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)

        for start in range(0, len(contexts), bs):
            chunk = contexts[start : start + bs]
            ids = [self.chat.encode(c) for c in chunk]
            batch = self._pad(ids)
            prompt_width = batch["input_ids"].shape[1]

            if seed is not None:
                self.torch.manual_seed(seed + start)

            with self.lm.generate(batch, max_new_tokens=max_new_tokens, **gen_kwargs):
                sequences = self.lm.generator.output.save()

            seq = np.asarray(sequences.cpu() if hasattr(sequences, "cpu") else sequences)
            for row in seq[:, prompt_width:]:
                token_ids = [int(x) for x in row]
                truncated = not any(x in self._eos_ids for x in token_ids)
                token_ids = self._strip_tail(token_ids)
                results.append(
                    Generation(
                        text=self.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                        token_ids=token_ids,
                        truncated=truncated,
                    )
                )
        return results

    def _strip_tail(self, token_ids: list[int]) -> list[int]:
        """Drop everything from the first end-of-turn token onward.

        ``reply_last`` is defined as the final generated token *pre-EOS*, so the
        terminator itself is not part of the reply span.
        """
        for i, tid in enumerate(token_ids):
            if tid in self._eos_ids:
                return token_ids[:i]
        return token_ids

    # -- (3) pooled residual capture ---------------------------------------
    def capture(
        self,
        context_ids: Sequence[Sequence[int]],
        reply_ids: Sequence[Sequence[int]] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Pooled residuals for every layer, in all three views.

        One teacher-forced forward over ``context + reply``.  Causal masking makes
        the ``pre_reply`` slice of this pass identical to a context-only forward, and
        feeding back the *generated token ids* (not a re-tokenised string) makes the
        reply span exactly the tokens generation produced.

        Returns ``{view: (N, n_layers, hidden)}`` fp32.  With no replies only
        ``pre_reply`` is meaningful; the other two views come back as NaN.
        """
        import nnsight

        t = self.torch
        bs = batch_size or self.cfg.capture_batch
        n = len(context_ids)
        replies: list[list[int]] = (
            [list(r) for r in reply_ids] if reply_ids is not None else [[] for _ in range(n)]
        )
        if len(replies) != n:
            raise ValueError("context_ids and reply_ids must be the same length")

        views = ("pre_reply", "reply_mean", "reply_last")
        out = {v: np.full((n, self.n_layers, self.hidden_size), np.nan, dtype=np.float32) for v in views}

        for start in range(0, n, bs):
            ctx_chunk = [list(c) for c in context_ids[start : start + bs]]
            rep_chunk = replies[start : start + bs]
            full = [c + r for c, r in zip(ctx_chunk, rep_chunk)]
            batch = self._pad(full)
            width = batch["input_ids"].shape[1]

            # Left padding puts every sequence flush right, so the reply span is
            # always the final len(reply) positions and pre_reply sits just before it.
            reply_len = t.tensor([len(r) for r in rep_chunk], dtype=t.long)
            pre_idx = (width - 1 - reply_len).to(self._device)
            last_idx = t.where(reply_len > 0, t.full_like(reply_len, width - 1), width - 1 - reply_len)
            last_idx = last_idx.to(self._device)

            positions = t.arange(width).unsqueeze(0)
            reply_mask = (positions >= (width - reply_len).unsqueeze(1)).to(self._device)

            with self.lm.trace(batch):
                pooled = [
                    nnsight.apply(
                        _pool_views,
                        self.lm.model.layers[layer].output,
                        pre_idx,
                        reply_mask,
                        last_idx,
                    ).save()
                    for layer in range(self.n_layers)
                ]

            # (n_layers, B, 3, H) -> (B, n_layers, H) per view
            stacked = np.stack([np.asarray(p, dtype=np.float32) for p in pooled], axis=0)
            stacked = np.transpose(stacked, (1, 0, 2, 3))
            for vi, view in enumerate(views):
                block = stacked[:, :, vi, :]
                if view != "pre_reply":
                    empty = np.array([len(r) == 0 for r in rep_chunk])
                    block = block.copy()
                    block[empty] = np.nan
                out[view][start : start + len(ctx_chunk)] = block

        return out

    # -- convenience --------------------------------------------------------
    @property
    def stop_token_ids(self) -> list[int]:
        """Every id that terminates an assistant turn, per the model's own config."""
        return sorted(self._eos_ids)

    def encode_contexts(self, contexts: Iterable[str]) -> list[list[int]]:
        return [self.chat.encode(c) for c in contexts]


# ---------------------------------------------------------------------------
# Elicitation readout layer
# ---------------------------------------------------------------------------
class Elicitor:
    """Asks the bank's elicitation questions and reads the answer off the logits.

    Bank-contract source: ``elicitation`` (prompt text and ``state_options``) and
    ``loader_contract.validation_invariants`` #6 (option sets must be first-token
    separable — enforced by :func:`silentdoubt.bank.assert_tokenizer_invariant`,
    which shares the variant expansion used here).

    Both the gates (§7) and the per-turn battery (§5) go through this class, so
    there is exactly one definition of "what P(no) means" in the codebase.
    """

    #: 0-9 for the digit-expectation readouts (confidence, valence).
    DIGITS: tuple[str, ...] = tuple(str(d) for d in range(10))
    YES_NO: tuple[str, ...] = ("yes", "no")
    AB: tuple[str, ...] = ("A", "B")

    def __init__(self, model: SubjectModel, state_options: Sequence[str]) -> None:
        from .bank import first_token_ids

        self.model = model
        self.sets: dict[str, list[str]] = {
            "yes_no": list(self.YES_NO),
            "ab": list(self.AB),
            "digits": list(self.DIGITS),
            "state_options": list(state_options),
        }
        variants = lambda w: first_token_ids(model.tokenizer, w)  # noqa: E731
        self.tables = {
            name: model.option_id_table(options, variants) for name, options in self.sets.items()
        }

    def read(self, contexts: Sequence[str], set_name: str, batch_size: int | None = None) -> list[OptionReadout]:
        id_matrix, id_mask = self.tables[set_name]
        return self.model.read_options(
            contexts, self.sets[set_name], id_matrix, id_mask, batch_size=batch_size
        )

    def yes_no(self, contexts: Sequence[str], **kw) -> list[OptionReadout]:
        return self.read(contexts, "yes_no", **kw)

    def state(self, contexts: Sequence[str], **kw) -> list[OptionReadout]:
        return self.read(contexts, "state_options", **kw)

    def choice(self, contexts: Sequence[str], **kw) -> list[OptionReadout]:
        return self.read(contexts, "ab", **kw)

    def digit(self, contexts: Sequence[str], **kw) -> list[OptionReadout]:
        """Digit read; ``OptionReadout.expectation(range(10))`` gives E[0-9]."""
        return self.read(contexts, "digits", **kw)

    @staticmethod
    def digit_expectation(readout: OptionReadout) -> float:
        return readout.expectation(range(10))


__all__ = [
    "SubjectModel",
    "ChatFormatter",
    "Elicitor",
    "OptionReadout",
    "Generation",
    "FOLD",
]
