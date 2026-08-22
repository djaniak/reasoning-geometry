"""Is the target's own tail geometry worth its cost against buying peer samples?

``peer_difficulty_control`` treats the two other models' pass rates as a control
and says, in its own docstring, that ``B0 + peer`` is "a control, never a
baseline the headline has to beat".  The 2026-08-21 review rejects that framing:
Hamidieh et al. deploy a scale-matched model ensemble as an uncertainty method,
so a reviewer is entitled to read the same resource as a competing baseline --
and on the frozen numbers the peer features already beat ``B1`` on two of three
targets.  The rejoinder is not that the peer control is inadmissible.  It is
that the existing comparison is not cost matched: the peer block consumes
sixteen extra generations from two other models, and ``rmd_tail_q20`` consumes
none.

So this module puts both on one axis.  Every rung is scored against what it
costs at decision time, in model calls and in generated tokens, and the rungs
are reported in cost order rather than in the order that flatters either side.

The cost model, stated once so it can be disagreed with:

* Every rung pays the target's eight generations.  ``B0`` needs them for the
  vote and the length statistics; they are the thing being scored.
* ``B1`` adds ``rmd_tail_q20``, read from hidden states of *those same*
  generations.  Its marginal generation cost over ``B0`` is **zero calls and
  zero tokens**.  It is not free -- it needs the states retained and a
  Mahalanobis readout over them -- but that cost is not a generation, and
  pretending otherwise would be the mirror image of the error being corrected.
* A peer rung at ``m`` samples from ``k`` peers adds ``k * m`` calls and
  ``k * m`` times that peer's mean trace length in tokens.

Which makes the comparison sharp rather than ambiguous: **no peer rung is cost
matched to ``B1``, because ``B1`` is free at the margin and the cheapest peer
rung is one extra generation.**  The question the review asks -- "does a cheap
peer win?" -- therefore reduces to: does ``B0 + one peer at one sample``, the
cheapest thing on the ladder that is not already paid for, beat ``B1``?  If it
does, the cheap peer wins outright and the paper must say so.  If it does not,
``B1`` wins at strictly lower cost, which is a stronger statement than winning
at matched cost.

Populations follow the budget-indexed correction: ``full_population`` is
primary, because ``cap_free_valid_plurality`` conditions on a difficulty-related
event and the frozen headline is 11-20% larger than the unconditional number.
The cap-free row is kept as the continuity check -- at ``k=2, m=8`` this
reproduces ``peer_difficulty_control``'s ``B0_plus_peer``, and disagreement
there is a bug here.

Two things this is not.  It is not a sweep for the best peer configuration: the
whole ladder is reported, the sizes are fixed at ``1/2/4/8`` before running, and
no rung is promoted on the strength of its result.  And it is not a claim that
peer sampling is deployable -- it is not, and ``peer_difficulty_control`` says
why -- only that a reviewer who *does* treat it as deployable must also count
what it costs.

Sub-sampled rungs carry an extra source of variance the frozen bootstrap does
not model: which sibling you happened to draw.  That is reported as a separate
axis -- the spread of the point estimate across independent draws -- rather than
folded into the interval, because the frozen convention resamples prompts with
the pipeline held fixed and mixing a second source into it would make these
intervals incomparable with every other interval in the paper.  Combining them
is the outer-refit rung's job, not this one's.

Sign convention: **AURC, lower is better**, so a negative ``left - right`` delta
favours the left-hand readout.

Not a DVC stage: it re-reads cached OOF rows and imports the frozen aggregation,
folds, populations, readout, bootstrap and seed convention rather than restating
any of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from applications.incremental_abstention import (
    BASE_FEATURE_NAMES,
    _finite,
    _group_rows,
    _read_oof,
    _winning_answer,
    aggregate_prompt_features,
    crossfit_logistic_predictions,
    paired_bootstrap_delta,
    prompt_metrics,
    select_layer_rows,
)
from baselines.closest_baselines import _populations
from controls.difficulty_control import _delta_seed
from controls.peer_difficulty_control import (
    METRICS,
    PEER_PREFIX,
    assert_shared_prompt_ids,
    oracle_aurc,
    peer_pass_rates,
    prompt_golds,
)

#: Sibling counts on the ladder, fixed before running.  Eight is the full cached
#: set and reproduces the frozen control; one is the cheapest thing that can be
#: bought.  The two in between are there so the reader sees a curve rather than
#: two points, not so that a winner can be picked off it.
LADDER_SIZES: tuple[int, ...] = (1, 2, 4, 8)

#: Independent re-draws of *which* siblings a sub-sampled rung got.  At ``m=8``
#: there is nothing to draw and the rung is evaluated once.
N_DRAWS = 25

#: Free at the margin, in generations.  Named as a constant because the entire
#: cost argument turns on it and it should be greppable.
RMD_EXTRA_CALLS = 0

#: Column prefix for the *deployable* peer signal.  ``PEER_PREFIX`` carries the
#: graded one -- the fraction of a peer's siblings that were **correct**, which
#: needs the gold answer and therefore cannot be computed at decision time.  A
#: peer ensemble you could actually run gives you answers, not grades, so the
#: deployable quantity is how many of the peer's samples **agree with the answer
#: the target is about to return**.  Both are on the ladder: the graded rung is
#: an upper bound on any peer method, and the agreement rung is the baseline a
#: reviewer could deploy.
AGREE_PREFIX = "peer_agreement__"


def peer_sample_rates(
    rows: Iterable[Mapping], size: int, *, seed: int, draw: int
) -> dict[int, float]:
    """Pass rate over ``size`` siblings drawn per prompt, without replacement.

    The full-cache rate is ``peer_pass_rates``; this is what you would have
    measured had you bought only ``size`` generations from that peer.  Drawing
    is per prompt and seeded on ``(seed, draw, prompt_id)``, so a draw is
    reproducible, independent across prompts, and independent across peers
    without the peers having to coordinate.

    Denominator is the drawn siblings, matching ``peer_pass_rates``: a trace
    with no extractable answer did not solve the problem, so it counts as a
    failure rather than being dropped.
    """
    rates: dict[int, float] = {}
    for prompt_id, group in sorted(_group_rows(rows).items()):
        values = [
            value for row in group if (value := _finite(row.get("is_correct"))) is not None
        ]
        if not values:
            rates[prompt_id] = float("nan")
            continue
        take = min(int(size), len(values))
        rng = np.random.default_rng((int(seed), int(draw), int(prompt_id)))
        chosen = rng.choice(len(values), size=take, replace=False)
        rates[prompt_id] = float(np.mean([values[index] for index in chosen]))
    return rates


def target_winners(rows: Iterable[Mapping]) -> dict[int, str | None]:
    """The answer the target would return per prompt: its plurality winner.

    ``_winning_answer`` is the frozen definition the outcome itself is scored
    from, so agreement is measured against the answer that is actually on the
    line and not against a second notion of what the model said.
    """
    return {
        prompt_id: _winning_answer(group)
        for prompt_id, group in _group_rows(rows).items()
    }


def peer_agreement_rates(
    rows: Iterable[Mapping],
    winners: Mapping[int, str | None],
    size: int,
    *,
    seed: int,
    draw: int,
) -> dict[int, float]:
    """Fraction of ``size`` drawn peer siblings that returned the target's answer.

    The deployable counterpart to :func:`peer_sample_rates`: no gold answer is
    consulted, only whether another model arrived at the same string.  A peer
    trace with no extractable answer counts as disagreement rather than being
    dropped, matching the graded rung's denominator -- it did not confirm the
    target's answer, and at decision time you would have paid for it anyway.

    A prompt where the target itself produced no plurality answer scores zero,
    the same convention ``aggregate_prompt_features`` uses for
    ``vote_agreement``: there is no answer for a peer to confirm.

    Drawn under the same ``(seed, draw, prompt_id)`` stream as the graded rung,
    so at a given ``draw`` the two rungs bought *the same generations* and
    differ only in what is read off them.
    """
    rates: dict[int, float] = {}
    for prompt_id, group in sorted(_group_rows(rows).items()):
        winner = winners.get(prompt_id)
        answers = [row.get("predicted_answer") for row in group]
        if not answers:
            rates[prompt_id] = float("nan")
            continue
        if winner is None:
            rates[prompt_id] = 0.0
            continue
        take = min(int(size), len(answers))
        rng = np.random.default_rng((int(seed), int(draw), int(prompt_id)))
        chosen = rng.choice(len(answers), size=take, replace=False)
        rates[prompt_id] = float(
            np.mean([str(answers[index]) == winner for index in chosen])
        )
    return rates


def tokens_per_sibling(rows: Iterable[Mapping]) -> float:
    """Mean generated tokens in one of this model's cached traces.

    Capped traces enter at the cap, which is what they cost: the budget was
    spent whether or not an answer came out of it.
    """
    lengths = [
        value for row in rows if (value := _finite(row.get("trace_length"))) is not None
    ]
    return float(np.mean(lengths)) if lengths else float("nan")


def ladder_rungs(
    peer_labels: Sequence[str], sizes: Sequence[int] = LADDER_SIZES
) -> dict[str, dict]:
    """Every readout on the ladder, with the peers and sample count it buys.

    ``B0`` and ``B1`` are the two rungs that buy nothing.  Each peer appears
    alone and the peers appear together, so "one peer" is reported for both
    choices rather than for whichever one happens to do better.

    Each purchased configuration appears twice: ``graded`` reads the peer's
    correctness, which needs the gold answer and so bounds rather than measures
    a deployable method, and ``agree`` reads only whether the peer returned the
    target's answer.  Same generations, same cost, different readout.
    """
    peer_labels = tuple(peer_labels)
    rungs: dict[str, dict] = {
        "B0": {"features": BASE_FEATURE_NAMES, "peers": (), "size": 0, "kind": "none"},
        "B1": {
            "features": BASE_FEATURE_NAMES + ("rmd_tail_q20",),
            "peers": (),
            "size": 0,
            "kind": "none",
        },
    }
    groups = [(label, (label,)) for label in peer_labels]
    if len(peer_labels) > 1:
        groups.append(("both", peer_labels))
    for kind, prefix in (("graded", PEER_PREFIX), ("agree", AGREE_PREFIX)):
        for name, members in groups:
            for size in sizes:
                rungs[f"B0_{kind}_{name}_m{size}"] = {
                    "features": BASE_FEATURE_NAMES
                    + tuple(prefix + member for member in members),
                    "peers": members,
                    "size": int(size),
                    "kind": kind,
                }
    return rungs


def rung_cost(
    rung: Mapping,
    *,
    target_calls: int,
    target_tokens: float,
    peer_tokens: Mapping[str, float],
) -> dict:
    """Model calls and generated tokens one prompt costs under this rung.

    ``extra_*`` is the margin over ``B0``, which is the number the comparison
    actually turns on: every rung pays for the target's own siblings, so the
    absolute totals differ by less than the decision does.
    """
    size = int(rung["size"])
    extra_calls = size * len(rung["peers"])
    extra_tokens = float(sum(size * peer_tokens[peer] for peer in rung["peers"]))
    return {
        "extra_calls": extra_calls,
        "extra_tokens": extra_tokens,
        "total_calls": int(target_calls) + extra_calls,
        "total_tokens": float(target_tokens) + extra_tokens,
    }


def _predictions(
    columns: Mapping[str, np.ndarray],
    spec: Sequence[str],
    outcomes: np.ndarray,
    folds: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    matrix = np.column_stack([columns[name] for name in spec])
    return crossfit_logistic_predictions(matrix, outcomes, folds, seed=seed)


def analyze_population(
    features: Mapping[int, Mapping],
    prompt_ids: Sequence[int],
    rungs: Mapping[str, Mapping],
    draws: Mapping[tuple[str, str, int, int], Mapping[int, float]],
    draw_counts: Mapping[int, int],
    costs: Mapping[str, Mapping],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Score every rung, then contrast each against ``B1`` and against ``B0``.

    A sub-sampled rung is fitted once per draw.  Its headline AURC is the mean
    over draws and its interval is taken on the *median* draw -- named, so the
    reader knows it is neither the best draw nor an arbitrary one -- with the
    full spread of point estimates reported beside it.  The spread, not the
    interval, is what says whether a rung's verdict depends on luck.
    """
    outcomes = np.asarray([features[i]["outcome"] for i in prompt_ids], dtype=float)
    folds = np.asarray([features[i]["fold"] for i in prompt_ids])
    base_names = sorted(
        {name for rung in rungs.values() for name in rung["features"]}
        - {name for name in _peer_column_names(rungs)}
    )
    base_columns = {
        name: np.asarray([features[i][name] for i in prompt_ids], dtype=float)
        for name in base_names
    }

    predictions: dict[str, list[np.ndarray]] = {}
    for name, rung in rungs.items():
        n_rung_draws = draw_counts.get(int(rung["size"]), 1) if rung["peers"] else 1
        fitted = []
        for draw in range(n_rung_draws):
            columns = dict(base_columns)
            prefix = PEER_PREFIX if rung["kind"] == "graded" else AGREE_PREFIX
            for peer in rung["peers"]:
                rates = draws[(rung["kind"], peer, int(rung["size"]), draw)]
                column = np.asarray(
                    [rates.get(i, float("nan")) for i in prompt_ids], dtype=float
                )
                missing = int((~np.isfinite(column)).sum())
                if missing:
                    raise ValueError(
                        f"peer {peer!r} has no pass rate for {missing} prompts in this "
                        f"population; the ladder assumes the three collects share every "
                        f"prompt id, which assert_shared_prompt_ids checks upstream"
                    )
                columns[prefix + peer] = column
            fitted.append(
                _predictions(columns, rung["features"], outcomes, folds, seed=seed)
            )
        predictions[name] = fitted

    floor = oracle_aurc(outcomes)
    scored = {}
    for name, fitted in predictions.items():
        per_draw = [prompt_metrics(values, outcomes) for values in fitted]
        aurcs = np.asarray([entry["aurc"] for entry in per_draw], dtype=float)
        median_draw = int(np.argsort(aurcs)[len(aurcs) // 2])
        scored[name] = {
            "kind": rungs[name]["kind"],
            "n_draws": len(fitted),
            "median_draw": median_draw,
            "aurc_mean": float(np.mean(aurcs)),
            "aurc_sd": float(np.std(aurcs, ddof=1)) if len(aurcs) > 1 else 0.0,
            "aurc_min": float(np.min(aurcs)),
            "aurc_max": float(np.max(aurcs)),
            "auacc_mean": float(np.mean([entry["auacc"] for entry in per_draw])),
            "aurc_headroom": float(np.mean(aurcs) - floor),
            "cost": dict(costs[name]),
        }

    contrasts = {}
    for name in rungs:
        if name == "B1":
            continue
        for left, right, label in (
            ("B1", name, f"B1_minus_{name}"),
            (name, "B0", f"{name}_minus_B0"),
        ):
            # ``B0`` still gets its ``B1_minus_B0`` reproduction row; what it
            # cannot have is a contrast against itself.
            if left == right:
                continue
            contrasts[label] = _contrast(
                predictions,
                scored,
                left,
                right,
                outcomes,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

    return {
        "n_prompts": len(prompt_ids),
        "base_accuracy": float(np.mean(outcomes)),
        "oracle_aurc": floor,
        "rungs": scored,
        "contrasts": contrasts,
    }


def _peer_column_names(rungs: Mapping[str, Mapping]) -> set[str]:
    return {
        name
        for rung in rungs.values()
        for name in rung["features"]
        if name.startswith(PEER_PREFIX) or name.startswith(AGREE_PREFIX)
    }


def _contrast(
    predictions: Mapping[str, Sequence[np.ndarray]],
    scored: Mapping[str, Mapping],
    left: str,
    right: str,
    outcomes: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """One ``left - right`` delta, with the across-draw spread carried beside it.

    The interval comes from the median draw of whichever side is sub-sampled.
    The spread comes from pairing every draw against the deterministic side, so
    a rung whose sign flips between draws is visible as such instead of being
    averaged into a confident-looking point estimate.
    """
    label = f"{left}_minus_{right}"
    left_draws, right_draws = predictions[left], predictions[right]
    left_index = scored[left]["median_draw"]
    right_index = scored[right]["median_draw"]

    per_draw = []
    for draw in range(max(len(left_draws), len(right_draws))):
        first = left_draws[min(draw, len(left_draws) - 1)]
        second = right_draws[min(draw, len(right_draws) - 1)]
        per_draw.append(
            float(
                prompt_metrics(first, outcomes)["aurc"]
                - prompt_metrics(second, outcomes)["aurc"]
            )
        )
    signs = {int(np.sign(value)) for value in per_draw}

    entry = {
        "n_draws": len(per_draw),
        "aurc_delta_per_draw_mean": float(np.mean(per_draw)),
        "aurc_delta_per_draw_min": float(np.min(per_draw)),
        "aurc_delta_per_draw_max": float(np.max(per_draw)),
        # The verdict survives the draw only if every draw agrees on direction.
        # With one draw this is trivially true and means nothing; ``n_draws``
        # is carried so it cannot be read as evidence when it is not.
        "sign_stable_across_draws": len(signs) == 1,
        "interval_from_draw": {"left": left_index, "right": right_index},
    }
    for metric in METRICS:
        entry[metric] = paired_bootstrap_delta(
            left_draws[left_index],
            right_draws[right_index],
            outcomes,
            metric=metric,
            n_bootstrap=n_bootstrap,
            seed=_delta_seed(seed, label, metric),
        )
    return entry


def analyze_model(
    label: str,
    rows: Sequence[Mapping],
    layer: int,
    data_dir: str | Path,
    peer_rows: Mapping[str, Sequence[Mapping]],
    *,
    populations: Sequence[str],
    sizes: Sequence[int] = LADDER_SIZES,
    n_draws: int = N_DRAWS,
    expected_traces: int = 8,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    features = aggregate_prompt_features(
        rows, data_dir=str(data_dir), expected_traces=expected_traces
    )
    peer_labels = tuple(sorted(peer_rows))
    rungs = ladder_rungs(peer_labels, sizes)

    # A rung that buys the whole cache has nothing to draw; one that buys less
    # is evaluated once per draw.  Keyed by size so the deterministic case stays
    # a property of the data rather than a hard-coded ``8``.
    draw_counts = {
        int(size): 1 if int(size) >= int(expected_traces) else int(n_draws)
        for size in sizes
    }
    draw_counts[0] = 1
    winners = target_winners(rows)
    draws: dict[tuple[str, str, int, int], Mapping[int, float]] = {}
    for peer, rows_for_peer in peer_rows.items():
        for size in sizes:
            for draw in range(draw_counts[int(size)]):
                if int(size) >= int(expected_traces):
                    # The full cache: nothing to draw, and the graded rate must
                    # equal the frozen control's definition exactly or the
                    # continuity check against it is empty.
                    graded = peer_pass_rates(rows_for_peer)
                else:
                    graded = peer_sample_rates(rows_for_peer, size, seed=seed, draw=draw)
                draws[("graded", peer, int(size), draw)] = graded
                draws[("agree", peer, int(size), draw)] = peer_agreement_rates(
                    rows_for_peer, winners, size, seed=seed, draw=draw
                )

    target_tokens = tokens_per_sibling(rows) * expected_traces
    peer_tokens = {
        peer: tokens_per_sibling(rows_for_peer) for peer, rows_for_peer in peer_rows.items()
    }
    costs = {
        name: rung_cost(
            rung,
            target_calls=expected_traces,
            target_tokens=target_tokens,
            peer_tokens=peer_tokens,
        )
        for name, rung in rungs.items()
    }

    available = _populations(features)
    body = {
        "label": label,
        "layer": layer,
        "peers": list(peer_labels),
        "target_tokens_per_prompt": target_tokens,
        "peer_tokens_per_sibling": peer_tokens,
        "rung_features": {name: list(rung["features"]) for name, rung in rungs.items()},
        "populations": {},
    }
    for population in populations:
        prompt_ids = [
            i for i in available[population] if features[i]["fold"] is not None
        ]
        if len(prompt_ids) < 2:
            continue
        body["populations"][population] = analyze_population(
            features,
            prompt_ids,
            rungs,
            draws,
            draw_counts,
            costs,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    return body


def cheapest_peer_verdict(body: Mapping, population: str) -> dict:
    """Does one extra generation from one peer already beat ``B1``?

    The ladder's whole point in one reading.  ``B1`` costs nothing at the
    margin, so if the cheapest purchasable rung loses to it, the target's own
    geometry wins at strictly lower cost; if it wins, a cheap peer wins and the
    paper says so.  Reported per single peer and per readout, since "one peer"
    is a choice, the two choices need not agree, and the graded reading is not
    available at decision time while the agreement reading is.
    """
    entry = body["populations"].get(population)
    if entry is None:
        return {"population": population, "available": False}
    verdicts = {}
    for name, rung in entry["rungs"].items():
        if rung["cost"]["extra_calls"] != 1:
            continue
        contrast = entry["contrasts"][f"B1_minus_{name}"]
        delta = contrast["aurc"]
        verdicts[name] = {
            "kind": rung["kind"],
            "deployable": rung["kind"] == "agree",
            "aurc_delta_B1_minus_rung": delta["point_estimate"],
            "ci": [delta["ci_low"], delta["ci_high"]],
            "excludes_zero": _excludes_zero(delta),
            "sign_stable_across_draws": contrast["sign_stable_across_draws"],
            # Negative favours B1: the free rung is ahead of the bought one.
            "winner": _winner(delta),
        }
    return {"population": population, "available": True, "rungs": verdicts}


def saturation_flags(body: Mapping, population: str, *, threshold: float = 0.9) -> dict:
    """Rungs that have removed nearly all the risk the oracle leaves removable.

    AURC does not bottom out at zero, and a rung sitting on the floor cannot be
    told apart from a better one by a delta.  Where the fraction of ``B0``'s
    headroom removed passes ``threshold``, "the peer wins" and "there was
    nothing left to win" stop being distinguishable, and the comparison on that
    model is reported as saturated rather than as a result.
    """
    entry = body["populations"].get(population)
    if entry is None:
        return {"population": population, "available": False}
    floor = entry["oracle_aurc"]
    base_headroom = entry["rungs"]["B0"]["aurc_mean"] - floor
    if base_headroom <= 0:
        return {"population": population, "available": True, "degenerate": True}
    removed = {
        name: float((entry["rungs"]["B0"]["aurc_mean"] - rung["aurc_mean"]) / base_headroom)
        for name, rung in entry["rungs"].items()
    }
    return {
        "population": population,
        "available": True,
        "oracle_aurc": floor,
        "B0_headroom": base_headroom,
        "headroom_fraction_removed": removed,
        "saturated_rungs": sorted(
            name for name, share in removed.items() if share >= threshold
        ),
        "threshold": threshold,
    }


def _excludes_zero(delta: Mapping) -> bool | None:
    low, high = delta.get("ci_low"), delta.get("ci_high")
    if low is None or high is None:
        return None
    return bool(low > 0 or high < 0)


def _winner(delta: Mapping) -> str:
    if not _excludes_zero(delta):
        return "tie"
    return "B1" if delta["point_estimate"] < 0 else "peer"


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}"


def _band(delta: Mapping) -> str:
    if delta.get("ci_low") is None:
        return "--"
    return f"[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]"


def _ordered_rungs(entry: Mapping, *, kind: str | None = None, drop: Sequence[str] = ()):
    items = [
        (name, rung)
        for name, rung in entry["rungs"].items()
        if name not in drop and (kind is None or rung["kind"] == kind)
    ]
    return sorted(items, key=lambda item: (item[1]["cost"]["extra_calls"], item[0]))


def write_report(
    results: Sequence[Mapping], path: str | Path, *, headline: str, sizes: Sequence[int]
) -> None:
    lines = [
        "# Peer baseline and cost ladder",
        "",
        "Every rung scored against what it costs at decision time. `B1` adds "
        "`rmd_tail_q20` to `B0` and buys **no** extra generations -- it reads "
        "hidden states of the eight the target already produced -- so it is free "
        "at the margin and no peer rung is cost matched to it.",
        "",
        "Two readouts of the same purchased generations:",
        "",
        "* **`graded`** -- the fraction of the peer's samples that were *correct*. "
        "Needs the gold answer, so it is **not computable at decision time**. It "
        "is an upper bound on what any peer method could deliver, not a baseline.",
        "* **`agree`** -- the fraction of the peer's samples that returned the "
        "*target's own answer*. No gold needed; this is the peer ensemble a "
        "reviewer could actually deploy.",
        "",
        "AURC, lower is better. A negative `B1 - rung` favours `B1`.",
        "",
        f"Headline population: `{headline}`. Ladder sizes: "
        f"{', '.join(str(size) for size in sizes)} siblings per peer.",
        "",
        "## 1. Floors and the two free rungs",
        "",
        "AURC does not bottom out at zero. Where a rung approaches the oracle "
        "floor, a delta can no longer separate \"this is better\" from \"there "
        "was nothing left to remove\".",
        "",
        "| Model | n | Base acc. | Oracle AURC | `B0` | `B1` | B0 headroom |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for body in results:
        entry = body["populations"].get(headline)
        if entry is None:
            continue
        floor = entry["oracle_aurc"]
        lines.append(
            f"| {body['label']} | {entry['n_prompts']} | {entry['base_accuracy']:.3f} | "
            f"{floor:.4f} | {entry['rungs']['B0']['aurc_mean']:.4f} | "
            f"{entry['rungs']['B1']['aurc_mean']:.4f} | "
            f"{entry['rungs']['B0']['aurc_mean'] - floor:.4f} |"
        )

    lines += [
        "",
        "## 2. The ladder in cost order",
        "",
        "`Extra` is the margin over `B0` per prompt. `AURC` is the mean over "
        f"{N_DRAWS} independent re-draws of which siblings were bought "
        "(sub-sampled rungs only; the full-cache rungs are deterministic). "
        "`removed` is the fraction of `B0`'s headroom the rung takes out -- at "
        "1.00 the rung is on the oracle floor.",
        "",
        "| Model | Rung | Kind | Extra calls | Extra tokens | AURC | across-draw range | removed |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for body in results:
        entry = body["populations"].get(headline)
        if entry is None:
            continue
        shares = saturation_flags(body, headline).get("headroom_fraction_removed", {})
        for name, rung in _ordered_rungs(entry):
            spread = (
                f"[{rung['aurc_min']:.4f}, {rung['aurc_max']:.4f}]"
                if rung["n_draws"] > 1
                else "--"
            )
            lines.append(
                f"| {body['label']} | `{name}` | {rung['kind']} | "
                f"{rung['cost']['extra_calls']} | {rung['cost']['extra_tokens']:.0f} | "
                f"{_fmt(rung['aurc_mean'])} | {spread} | "
                f"{_fmt(shares.get(name), 2)} |"
            )

    lines += [
        "",
        "## 3. Does a bought rung beat the free one?",
        "",
        "`B1 - rung` on AURC. Negative favours `B1`, the rung that costs no "
        "extra generations. The interval is the frozen prompt bootstrap on the "
        "median draw; `sign stable` reports whether every draw agreed on the "
        "direction, which is the separate question of whether the verdict "
        "depends on which siblings you happened to buy.",
        "",
        "| Model | Rung | Kind | Extra calls | B1 - rung | 95% CI | excludes 0 | sign stable | winner |",
        "|---|---|---|---:|---:|---|:--:|:--:|---|",
    ]
    for body in results:
        entry = body["populations"].get(headline)
        if entry is None:
            continue
        for name, rung in _ordered_rungs(entry, drop=("B1",)):
            contrast = entry["contrasts"][f"B1_minus_{name}"]
            delta = contrast["aurc"]
            stable = (
                "--" if contrast["n_draws"] == 1
                else ("yes" if contrast["sign_stable_across_draws"] else "no")
            )
            lines.append(
                f"| {body['label']} | `{name}` | {rung['kind']} | "
                f"{rung['cost']['extra_calls']} | {_fmt(delta['point_estimate'])} | "
                f"{_band(delta)} | {'yes' if _excludes_zero(delta) else 'no'} | "
                f"{stable} | {_winner(delta)} |"
            )

    lines += [
        "",
        "## 4. The one-extra-generation question",
        "",
        "The cheapest thing on the ladder that is not already paid for is one "
        "sample from one peer. If it beats `B1`, a cheap peer wins outright. If "
        "it does not, `B1` wins at strictly lower cost. Only the `agree` rows "
        "are a fair answer to that question -- the `graded` rows need the gold "
        "answer and are reported as the bound they are.",
        "",
        "| Model | Rung | Kind | Deployable | B1 - rung | 95% CI | sign stable | winner |",
        "|---|---|---|:--:|---:|---|:--:|---|",
    ]
    for body in results:
        verdict = cheapest_peer_verdict(body, headline)
        if not verdict.get("available"):
            continue
        for name, rung in sorted(
            verdict["rungs"].items(), key=lambda item: (item[1]["kind"], item[0])
        ):
            low, high = rung["ci"]
            band = "--" if low is None else f"[{low:+.4f}, {high:+.4f}]"
            lines.append(
                f"| {body['label']} | `{name}` | {rung['kind']} | "
                f"{'yes' if rung['deployable'] else 'no'} | "
                f"{_fmt(rung['aurc_delta_B1_minus_rung'])} | {band} | "
                f"{'yes' if rung['sign_stable_across_draws'] else 'no'} | "
                f"{rung['winner']} |"
            )

    lines += [
        "",
        "## 5. Saturated comparisons",
        "",
        "Rungs that have removed at least 90% of `B0`'s headroom. On these the "
        "delta against `B1` is not evidence about which readout is better; there "
        "is almost nothing left for either to remove.",
        "",
    ]
    for body in results:
        flags = saturation_flags(body, headline)
        if not flags.get("available") or flags.get("degenerate"):
            continue
        names = flags["saturated_rungs"]
        lines.append(
            f"* **{body['label']}** -- headroom {flags['B0_headroom']:.4f} above a "
            f"floor of {flags['oracle_aurc']:.4f}; "
            + (
                f"{len(names)} rung(s) saturated: "
                + ", ".join(f"`{name}`" for name in names)
                if names
                else "no rung saturated"
            )
        )

    lines += [
        "",
        "## What the cost model does not charge for",
        "",
        "`B1` needs the target's hidden states retained and a Mahalanobis "
        "readout fitted over them. That is real work and real memory; it is not "
        "a generation, and it does not scale with the number of models you are "
        "willing to run. Charging it as tokens would be the mirror image of the "
        "error this rung exists to correct, so it is named here and left "
        "uncosted rather than silently folded in.",
        "",
        "The `graded` rungs are not charged for the gold answer they consume, "
        "because it cannot be bought at decision time at any price. They bound "
        "the peer family from above and are reported for that reason only.",
        "",
    ]
    Path(path).write_text("\n".join(lines))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL:OOF_CSV:DATA_DIR",
        help="repeatable; colon-separated triple. At least two are needed, since "
        "each model's peers are the others.",
    )
    parser.add_argument(
        "--population",
        action="append",
        default=None,
        help="default: full_population (primary), then cap_free_valid_plurality "
        "(continuity with the frozen peer control).",
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--expected_traces", type=int, default=8)
    parser.add_argument("--n_draws", type=int, default=N_DRAWS)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/peer_cost_ladder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    populations = tuple(
        args.population or ("full_population", "cap_free_valid_plurality")
    )
    specs = [spec.split(":", 2) for spec in args.model]
    if len(specs) < 2:
        raise SystemExit("need at least two models: a target's peers are the others")

    loaded = {}
    for label, oof_csv, data_dir in specs:
        rows, layer = select_layer_rows(_read_oof(oof_csv), args.layer, context=str(oof_csv))
        loaded[label] = {"rows": rows, "layer": layer, "data_dir": data_dir}
    shared = assert_shared_prompt_ids(
        {label: prompt_golds(body["rows"]) for label, body in loaded.items()}
    )

    results = [
        analyze_model(
            label,
            loaded[label]["rows"],
            loaded[label]["layer"],
            loaded[label]["data_dir"],
            {peer: loaded[peer]["rows"] for peer in loaded if peer != label},
            populations=populations,
            n_draws=args.n_draws,
            expected_traces=args.expected_traces,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        for label, _, _ in specs
    ]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "cost_model": {
            "target_calls": args.expected_traces,
            "rmd_extra_calls_over_B0": RMD_EXTRA_CALLS,
            "peer_extra_calls": "n_peers * siblings_bought",
            "peer_extra_tokens": "n_peers * siblings_bought * that peer's mean trace length",
            "uncosted": "hidden-state retention and the Mahalanobis readout for B1; "
            "real work, but not a generation",
        },
        "ladder_sizes": list(LADDER_SIZES),
        "n_draws": args.n_draws,
        "n_shared_prompt_ids": len(shared),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "readouts": {
            "graded": "fraction of the peer's drawn siblings with is_correct == 1; "
            "needs the gold answer, so it bounds the peer family rather than "
            "measuring a deployable method",
            "agree": "fraction of the peer's drawn siblings whose predicted_answer "
            "equals the target's plurality answer; no gold consulted, so this is "
            "the peer ensemble a reviewer could deploy",
        },
        "cheapest_peer_verdict": {
            body["label"]: cheapest_peer_verdict(body, populations[0]) for body in results
        },
        "saturation": {
            body["label"]: saturation_flags(body, populations[0]) for body in results
        },
        "models": results,
    }
    (output / "peer_cost_ladder_results.json").write_text(json.dumps(payload, indent=2))
    write_report(
        results,
        output / "peer_cost_ladder_report.md",
        headline=populations[0],
        sizes=LADDER_SIZES,
    )
    print(f"wrote {output}/peer_cost_ladder_report.md")


if __name__ == "__main__":
    main()
