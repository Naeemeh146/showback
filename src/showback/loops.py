"""LoopDetector: deterministic loop detection over the agent action stream.

An infinite or uselessly repetitive agent loop burns money without progress, so
it deserves a dedicated detector instead of being discovered when the budget
finally runs out. This module provides the deterministic primitive:

* Cemri et al., "Why Do Multi-Agent LLM Systems Fail?" (MAST)
  (arXiv:2503.13657, NeurIPS 2025) find that step repetition (~15.7% of failures) and
  failure-to-terminate (~12.4%) are the dominant agent failure modes. The
  implementable primitive from that finding is a *repetition detector* over the
  action stream — exactly what :class:`LoopDetector` is.
* Dubey, "Real-Time Detection and Repair of LLM Agent Failures"
  (arXiv:2608.02464) advocates a deterministic verification layer
  sitting beside the LLM. :class:`LoopDetector` is such a layer: no model
  calls, no network, pure state machine.

ML-based telemetry (e.g. CUSUM drift detection over action distributions) is an
explicit roadmap item, not v1.

Documented recipe — abort the run through the same accounting path as a
budget breach::

    detector = LoopDetector()
    try:
        for step in plan:
            detector.observe(tool_name, args_sig)
            ...
    except LoopDetected as e:
        tracker.record_tool("loop_detected", 0.0, note=f"pattern={e.pattern}")
        raise loop_to_budget_exceeded(e, guard) from e
"""

from __future__ import annotations

from showback.guard import BudgetExceeded, BudgetGuard


class LoopDetected(Exception):
    """Raised when a loop is detected in the observed action stream.

    Attributes:
        history: normalized signatures observed so far, oldest first.
        pattern: the repeating unit that tripped detection.
        kind: "repeat" (same action ``max_repeats`` times in a row) or "cycle"
            (a length-L block seen twice in a row).
    """

    def __init__(
        self,
        history: list[str],
        pattern: tuple[str, ...],
        kind: str,
    ) -> None:
        self.history = list(history)
        self.pattern = tuple(pattern)
        self.kind = kind
        if kind == "repeat":
            run = 1
            for sig in reversed(history[:-1]):
                if sig != self.pattern[0]:
                    break
                run += 1
            detail = f"{self.pattern[0]!r} x{run}"
        else:
            detail = " -> ".join(repr(sig) for sig in self.pattern) + " (twice)"
        super().__init__(f"Loop detected ({kind}): {detail}.")


class LoopDetector:
    """Deterministic repetition detector over normalized action signatures.

    Each observed ``(action, args_signature)`` pair is normalized to
    ``f"{action.strip().lower()}|{args_signature.strip().lower()}"`` and
    appended to history. After every observation the detector checks:

    1. **Exact repeat** — the last ``max_repeats`` signatures are all equal.
    2. **Cycle** — for each ``L`` in ``cycle_lengths`` (increasing), the last
       ``2*L`` signatures form the same length-L block twice.

    Exact repeats are checked before cycles.
    """

    def __init__(
        self,
        max_repeats: int = 3,
        cycle_lengths: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        if max_repeats < 2:
            raise ValueError(f"max_repeats must be >= 2, got {max_repeats!r}.")
        for length in cycle_lengths:
            if length < 1:
                raise ValueError(f"cycle_lengths entries must be >= 1, got {length!r}.")
        self.max_repeats = max_repeats
        self.cycle_lengths = tuple(cycle_lengths)
        self._history: list[str] = []

    def observe(self, action: str, args_signature: str = "") -> None:
        """Record one action and raise :class:`LoopDetected` if it loops."""
        signature = f"{action.strip().lower()}|{args_signature.strip().lower()}"
        self._history.append(signature)
        history = self._history

        if len(history) >= self.max_repeats:
            window = history[-self.max_repeats :]
            if all(sig == window[0] for sig in window):
                raise LoopDetected(history, (window[0],), "repeat")

        for length in sorted(self.cycle_lengths):
            if len(history) >= 2 * length:
                block = history[-length:]
                # A run of identical signatures is the repeat detector's job,
                # not a cycle — otherwise two identical observes would always
                # trip a length-1 cycle before max_repeats is reached.
                window = history[-2 * length :]
                if all(sig == window[0] for sig in window):
                    continue
                if history[-2 * length : -length] == block:
                    raise LoopDetected(history, tuple(block), "cycle")

    def reset(self) -> None:
        """Clear the observation history so the detector starts fresh."""
        self._history.clear()

    @property
    def history(self) -> tuple[str, ...]:
        """Normalized signatures observed so far, oldest first."""
        return tuple(self._history)


def loop_to_budget_exceeded(exc: LoopDetected, guard: BudgetGuard) -> BudgetExceeded:
    """Convert a detected loop into a :class:`BudgetExceeded`.

    Lets the run abort through the same accounting path as a budget breach —
    ``guard.budget_usd`` as the cap and ``guard.spent_usd`` as the spent value
    at detection time. The original :class:`LoopDetected` is available to the
    caller for ``raise ... from exc`` chaining.

    ``showback.guard`` does not import this module, so this is cycle-free.
    """
    return BudgetExceeded(guard.budget_usd, guard.spent_usd, guard.match_tags)
