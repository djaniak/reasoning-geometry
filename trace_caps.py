"""Length-cap resolution shared by the trace-level analyses.

Every capped-trace count depends on a cap that matches the generation budget the
traces were actually collected under.  Two mistakes silently produce a count of
zero rather than an error, and both make a truncated population wear a cap-free
label:

* the cap is missing, so a guard of the form ``max_new_tokens and length >= cap``
  short-circuits and excludes nothing;
* the cap is copied from a different model (the per-model budgets in
  ``params.yaml`` differ by 8x), so no trace can reach it.

The second mistake is what produced a reported Qwen "cap-free" population of 498
against a true value of 392: the run was given DeepSeek's 8192 instead of Qwen's
1024.  Both are raised here so that a cap-free claim is never inferred from a
silent zero.
"""

from __future__ import annotations

from typing import Iterable


def resolve_cap(
    max_new_tokens: int | None,
    lengths: Iterable[float | int | None],
    *,
    context: str = "",
) -> int:
    """Return the generation cap, or raise if it cannot be trusted.

    ``lengths`` are the observed trace lengths the cap will be applied to.  A cap
    above every observed length can never mark a trace as capped, which is
    indistinguishable from having been handed the wrong model's budget, so it is
    rejected rather than reported as a cap-free population.
    """
    where = f" ({context})" if context else ""
    if max_new_tokens is None:
        raise ValueError(
            f"max_new_tokens is required to count capped traces{where}. "
            "Pass the budget this model was collected under (see the per-model "
            "max_new_tokens entries in params.yaml); without it no trace is ever "
            "counted as capped and a truncated population is labelled cap-free."
        )
    cap = int(max_new_tokens)
    observed = [int(value) for value in lengths if value is not None]
    if observed and max(observed) < cap:
        raise ValueError(
            f"max_new_tokens={cap} exceeds every observed trace length "
            f"(max {max(observed)}){where}, so no trace can be counted as capped. "
            "This is the signature of a cap copied from another model; pass the "
            "budget these traces were actually collected under."
        )
    return cap
