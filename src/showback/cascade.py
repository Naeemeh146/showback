"""CascadeRouter: FrugalGPT-style LLM cascading for cost-efficient agents.

This module implements the model-cascade strategy from:

    FrugalGPT: How to Use Large Language Models While Reducing Cost and
    Improving Performance (Lingjiao Chen, Matei Zaharia, James Zou),
    arXiv:2305.05176, Transactions on Machine Learning Research (TMLR), 2024.

Paper-to-code design mapping:

* **tiers** — the cascade's models ordered cheap → expensive. Each tier is a
  ``(model, scorer, threshold)`` triple: a model plus the scoring function and
  acceptance threshold used for its answers, mirroring FrugalGPT's per-model
  scoring function + decision threshold.
* **escalation** — answers flow up the cascade: a tier's answer is accepted
  only when its score meets that tier's threshold; otherwise the router
  escalates to the next, stronger (and pricier) tier.
* **last-tier fallback** — the final, most capable tier is the backstop: its
  answer is accepted regardless of score, exactly as FrugalGPT always runs the
  strongest model last to guarantee a response.
* **score-then-pay ordering** — every attempted tier is recorded in the
  :class:`SpendTracker` ledger, so ``total_cost`` reflects all tiers tried,
  not just the accepted one.

The scorers shipped here (``nonempty_scorer``, ``keyword_scorer``) are
heuristic stand-ins. Real deployments should plug a trained reliability
classifier — e.g. a fine-tuned DistilBERT over input/output pairs, as in the
FrugalGPT paper — via ``functools.partial``::

    from functools import partial
    cheap_scorer = partial(distilbert_reliability, model=trained_clf)
    router = CascadeRouter([("cheap-7b", cheap_scorer, 0.9), ...], tracker)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from showback.guard import BudgetGuard
from showback.tracker import SpendTracker

#: (prompt, answer) -> reliability in [0, 1].
Scorer = Callable[[str, str], float]

#: (model, prompt) -> (answer_text, input_tokens, output_tokens).
CallFn = Callable[[str, str], tuple[str, int, int]]


@dataclass
class CascadeResult:
    """Outcome of one :meth:`CascadeRouter.route` call."""

    answer: str
    model: str  # the model whose answer was accepted
    tiers_tried: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)  # aligned with tiers_tried
    total_cost: float = 0.0  # USD recorded in the tracker for this route() call


class CascadeRouter:
    """Route a prompt through a cheap → expensive model cascade.

    Args:
        tiers: non-empty list of ``(model, scorer, threshold)`` triples,
            ordered cheap → expensive. ``scorer`` maps ``(prompt, answer)``
            to a reliability in ``[0, 1]``; ``threshold`` is the acceptance
            bar for that tier and must itself be in ``[0, 1]``.
        tracker: :class:`SpendTracker` where every attempted tier is recorded.
        guard: optional :class:`BudgetGuard`; :meth:`route` calls
            ``guard.check()`` before each tier so an over-budget cascade is
            aborted before any further spend.
    """

    def __init__(
        self,
        tiers: list[tuple[str, Scorer, float]],
        tracker: SpendTracker,
        guard: BudgetGuard | None = None,
    ) -> None:
        if not tiers:
            raise ValueError("tiers must be a non-empty list")
        if not isinstance(tracker, SpendTracker):
            raise ValueError("tracker must be a SpendTracker")
        validated: list[tuple[str, Scorer, float]] = []
        for i, tier in enumerate(tiers):
            if not isinstance(tier, tuple) or len(tier) != 3:
                raise ValueError(f"tier {i} must be a (model, scorer, threshold) tuple")
            model, scorer, threshold = tier
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"tier {i}: model must be a non-empty string")
            if not callable(scorer):
                raise ValueError(f"tier {i}: scorer must be callable")
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ValueError(f"tier {i}: threshold must be a float in [0, 1]")
            threshold = float(threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"tier {i}: threshold must be a float in [0, 1]")
            validated.append((model, scorer, threshold))
        self.tiers = validated
        self.tracker = tracker
        self.guard = guard

    def route(self, prompt: str, call_fn: CallFn, **tags: object) -> CascadeResult:
        """Run the cascade and return the accepted answer.

        For each tier in cheap → expensive order: check the guard (when set),
        call the model via ``call_fn``, score the answer, record the attempt
        in the tracker, and accept it when the score meets the tier's
        threshold. The last tier is the fallback — its answer is accepted
        regardless of score. Exceptions raised by ``call_fn`` propagate
        unchanged.
        """
        tiers_tried: list[str] = []
        scores: list[float] = []
        cost_before = self.tracker.total_cost()
        last_index = len(self.tiers) - 1

        for i, (model, scorer, threshold) in enumerate(self.tiers):
            if self.guard is not None:
                self.guard.check()
            answer, in_tok, out_tok = call_fn(model, prompt)
            score = scorer(prompt, answer)
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"scorer for tier {model!r} must return a float")
            score = float(score)
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"scorer for tier {model!r} returned {score}: out of [0, 1]")
            tiers_tried.append(model)
            scores.append(score)
            self.tracker.record_llm(
                model,
                in_tok,
                out_tok,
                note=f"cascade tier={model} score={score}",
                **tags,
            )
            if score >= threshold or i == last_index:
                return CascadeResult(
                    answer=answer,
                    model=model,
                    tiers_tried=tiers_tried,
                    scores=scores,
                    total_cost=round(self.tracker.total_cost() - cost_before, 6),
                )
        raise RuntimeError("unreachable: cascade has at least one tier")  # pragma: no cover


def nonempty_scorer(prompt: str, answer: str, min_length: int = 1) -> float:
    """1.0 if the stripped answer has at least ``min_length`` chars, else 0.0.

    Usable directly as a :data:`Scorer` thanks to the default; use
    ``functools.partial`` for a custom ``min_length``.
    """
    _ = prompt
    return 1.0 if len(answer.strip()) >= min_length else 0.0


def keyword_scorer(prompt: str, answer: str, keywords: tuple[str, ...] | list[str]) -> float:
    """Fraction of ``keywords`` found in the answer (case-insensitive).

    Raises :class:`ValueError` when ``keywords`` is empty.
    """
    _ = prompt
    if not keywords:
        raise ValueError("keywords must be non-empty")
    lowered = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in lowered)
    return found / len(keywords)


__all__ = [
    "Scorer",
    "CallFn",
    "CascadeResult",
    "CascadeRouter",
    "nonempty_scorer",
    "keyword_scorer",
]
