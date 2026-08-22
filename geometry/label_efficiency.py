"""Label-efficiency curves: one-class geometry against a supervised hidden-state probe.

The abstention increment rests on ``rmd_tail_q20``, a *one-class* statistic --
its reference manifold is fitted on correct traces alone and never sees an
incorrect one.  ``probe_hidden_tail_q20`` is the discriminative alternative: a
supervised LDA on the same PCA-projected tail means, fitted on both classes.  At
the full label budget the probe wins (FINDINGS.md, 2026-07-29: +0.025 pooled AUC
on Qwen, +0.033 prompt-centered on DeepSeek).  So the deployment claim cannot be
"geometry beats a probe".  It can only be "geometry costs fewer labels" -- and
that is a claim about a *curve*, not about a point.  This module measures it.

Everything the label budget can pay for is refitted at each budget, from that
budget's prompts only:

  * PCA(128) and the correct-trace Gaussian   -- the RMD reference
  * the background Gaussian                   -- the RMD denominator
  * the LDA on projected tail means           -- ``probe_hidden_tail_q20``
  * the LDA on individual tail tokens         -- ``probe_token_tail_q20``
  * the logistic abstention readout           -- B0, B0+rmd, B0+probe

**Two probes, because one of them confounds two things at once.**  RMD scores a
trace by computing a distance *per tail token* and averaging the token scores.
``probe_hidden_tail_q20`` averages first and classifies once, so it differs from
RMD in supervision *and* in pooling order simultaneously, and a gap between them
cannot be attributed to either.  ``probe_token_tail_q20`` closes that: it fits
the LDA on individual tail tokens and averages the token scores, matching RMD's
pooling order exactly, so supervision is the only remaining difference.

Nothing else downstream consumes a label, so this is the whole cost of both.

**Design.**  Each replicate draws one permutation of the prompts.  Training sets
are *nested* along it, so a curve is one within-replicate trajectory rather than
five independent draws.  The evaluation set is *fixed within a replicate* -- the
headline-population prompts lying outside the largest budget -- so AURC levels
are comparable across budgets, and every readout at every budget is scored on
identical prompts.  That pairing is what makes the comparison of interest, the
delta between two readouts, essentially free of evaluation noise.

**Deviations from the frozen pipeline.**  Two, both applied uniformly across
budgets and readouts so neither side can be favoured:

  1. Every reference fit sees a fixed per-trace token subsample
     (``--max_tokens_per_trace``) instead of the whole sequence.  The frozen
     pipeline already caps pooled reference tokens (params.yaml:
     ``max_reference_tokens: 2000000``); this is the same device moved per
     trace, so that the token count still grows with the label budget.
  2. Only the 20% tail block is retained for scoring.  Both readouts under
     comparison read exactly that block, so this is lossless for them -- and it
     is what keeps a 40 GiB layer resident across a hundred refits.
  3. The PCA solver is pinned to ``randomized``.  The frozen helper switches to
     ``full`` below 200k pooled tokens, a threshold this sweep walks across --
     see :func:`fit_correct_reference`.

Neither is the frozen configuration, so numbers here are not interchangeable
with the frozen artifacts; the largest budget is reported alongside the frozen
value as the sanity check.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from analysis.analyze import (
    _concatenate_hidden_tokens,  # applies the pooled reference-token cap and dtype
    _fit_lw_precision,
    compute_relative_mahal_distances,
    extend_reference_with_background_safe,
    load_all_traces,
    resolve_pca_n_components,
    set_compute_dtype,
    set_max_reference_tokens,
)
from baselines.best_of_n import group_traces_by_problem
from applications.incremental_abstention import (
    BASE_FEATURE_NAMES,
    _impute_and_scale,  # the frozen readout's imputation/scaling, reused verbatim
    _read_oof,
    aggregate_prompt_features,
    prompt_metrics,
    select_layer_rows,
)
from applications.prompt_decomposition import (
    fit_hidden_state_probe,
    is_unparsed,
    region_indices,
    score_hidden_state_probe,
)


#: Labelled-prompt budgets. 400 of ~500 prompts is what bounds the evaluation
#: set to the ~80 prompts left over; the smaller budgets are free of that cost.
DEFAULT_BUDGETS = (25, 50, 100, 200, 400)

GEOMETRY_FEATURE = "rmd_tail_q20"
PROBE_FEATURE = "probe_hidden_tail_q20"
TOKEN_PROBE_FEATURE = "probe_token_tail_q20"
QUADRATIC_FEATURE = "qmd_tail_q20"

#: Every feature the geometry side produces per trace, in report order.
GEOMETRY_FEATURES = (
    GEOMETRY_FEATURE,
    PROBE_FEATURE,
    TOKEN_PROBE_FEATURE,
    QUADRATIC_FEATURE,
)

#: The tail block is sliced out of each trace up front, so the region argument
#: handed to the frozen probe helpers is "full" *over that block* -- which is
#: identically the tail_q20 region mean of the whole trace.
TAIL_REGION = "tail_q20"

READOUT_SPECS: dict[str, tuple[str, ...]] = {
    "B0": BASE_FEATURE_NAMES,
    "B0_rmd": BASE_FEATURE_NAMES + (GEOMETRY_FEATURE,),
    "B0_probe": BASE_FEATURE_NAMES + (PROBE_FEATURE,),
    "B0_token_probe": BASE_FEATURE_NAMES + (TOKEN_PROBE_FEATURE,),
    "B0_qmd": BASE_FEATURE_NAMES + (QUADRATIC_FEATURE,),
    "B0_both": BASE_FEATURE_NAMES + (GEOMETRY_FEATURE, PROBE_FEATURE),
}

#: ``(left, right)`` pairs reported as ``left - right``.
#:
#: The four comparators form a ladder in which each rung releases one variable:
#:
#:   ``rmd``         positives-only contrast, quadratic, score-then-pool
#:   ``qmd``         both classes,            quadratic, score-then-pool
#:   ``token_probe`` both classes,            linear,    score-then-pool
#:   ``probe``       both classes,            linear,    pool-then-score
#:
#: so ``B0_rmd - B0_qmd`` isolates the use of negative labels, ``B0_qmd -
#: B0_token_probe`` isolates the decision function's functional form, and
#: ``B0_token_probe - B0_probe`` isolates pooling order.  ``B0_rmd - B0_probe``
#: is the original comparison, which moves all three at once.
PAIRED_DELTAS = (
    ("B0_rmd", "B0"),
    ("B0_probe", "B0"),
    ("B0_rmd", "B0_probe"),
    ("B0_both", "B0_rmd"),
    ("B0_token_probe", "B0"),
    ("B0_rmd", "B0_token_probe"),
    ("B0_qmd", "B0"),
    ("B0_rmd", "B0_qmd"),
    ("B0_qmd", "B0_token_probe"),
)

HEADLINE_POPULATION = "cap_free_valid_plurality"


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Trace reduction
# ---------------------------------------------------------------------------

def reference_subsample_indices(
    n_tokens: int, max_tokens: int | None, *, seed: int, trace_id: int
) -> np.ndarray:
    """Deterministic per-trace token subsample shared by every reference fit.

    Fixed per trace rather than per fit, so that two budgets differ only in
    *which prompts* they see and never in which tokens of a shared prompt.
    """
    if n_tokens <= 0:
        raise ValueError(f"trace {trace_id} has no tokens")
    if max_tokens is None or n_tokens <= int(max_tokens):
        return np.arange(n_tokens, dtype=int)
    rng = np.random.default_rng([int(seed), int(trace_id), int(n_tokens)])
    return np.sort(rng.choice(n_tokens, size=int(max_tokens), replace=False))


def prepare_trace_views(
    traces: Sequence[Mapping],
    layer: int,
    *,
    max_tokens_per_trace: int | None,
    seed: int = 42,
) -> list[dict]:
    """Reduce each trace to the two token blocks this study needs.

    ``hiddens`` carries the reference subsample under the layer key, so the
    frozen fitting helpers accept a view unchanged; ``tail`` carries the scoring
    block.  The caller is expected to drop the source traces afterwards -- the
    reduction is roughly twenty-fold, and it is what makes a hundred refits
    against a resident layer possible at all.
    """
    views = []
    for trace in traces:
        trace_id = int(trace["trace_id"])
        hidden = trace.get("hiddens", {}).get(layer)
        if hidden is None:
            raise ValueError(f"trace {trace_id} is missing layer {layer}")
        entropies = trace.get("entropies")
        if entropies is None:
            raise ValueError(f"trace {trace_id} is missing entropies")
        entropies = np.asarray(entropies, dtype=float)
        hidden = np.asarray(hidden)
        if hidden.shape[0] != entropies.shape[0]:
            raise ValueError(
                f"trace {trace_id}: {hidden.shape[0]} hidden rows against "
                f"{entropies.shape[0]} entropies"
            )
        tail = region_indices(entropies, TAIL_REGION)
        reference = reference_subsample_indices(
            int(entropies.shape[0]),
            max_tokens_per_trace,
            seed=seed,
            trace_id=trace_id,
        )
        views.append(
            {
                "trace_id": trace_id,
                "prompt_id": int(trace["idx"]),
                "sample_id": int(trace.get("sample_id", 0)),
                "is_correct": bool(trace["is_correct"]),
                "predicted_answer": trace.get("predicted_answer"),
                "hiddens": {layer: hidden[reference].copy()},
                "tail": hidden[tail].copy(),
                # Named `entropies` because the frozen probe helpers read that key;
                # it is the tail slice, matching the tail block row for row.
                "entropies": entropies[tail].copy(),
                "trace_length": int(entropies.shape[0]),
            }
        )
    return views


def group_views_by_prompt(views: Sequence[Mapping]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for view in views:
        grouped.setdefault(int(view["prompt_id"]), []).append(dict(view))
    return {
        prompt_id: sorted(group, key=lambda view: (view["sample_id"], view["trace_id"]))
        for prompt_id, group in sorted(grouped.items())
    }


# ---------------------------------------------------------------------------
# Fitting and scoring at one budget
# ---------------------------------------------------------------------------

def fit_correct_reference(correct_traces: Sequence[Mapping], layer: int, pca_dim: int):
    """:func:`analyze.fit_mahalanobis_reference` with the PCA solver pinned.

    The frozen version picks ``svd_solver`` by token count -- ``"randomized"``
    above 200k pooled tokens, ``"full"`` below.  Every frozen run sits on one
    side of that line, so it never matters there.  A label-efficiency sweep
    walks straight across it: the smaller budgets would be fitted by a different
    decomposition than the larger ones, putting a methodological step change in
    the middle of the very curve being measured.  Randomized is also what the
    frozen runs actually used, and ``full`` turns out to cost minutes per fit on
    a 3584-wide matrix, which is what surfaced this in the first place.
    """
    correct_hiddens = _concatenate_hidden_tokens(list(correct_traces), layer)
    pca = PCA(
        n_components=resolve_pca_n_components(correct_hiddens, pca_dim),
        random_state=42,
        svd_solver="randomized",
    )
    pca.fit(correct_hiddens)
    projected = pca.transform(correct_hiddens)
    mu = projected.mean(axis=0)
    return pca, mu, _fit_lw_precision(projected - mu)


def incorrect_side_views(train_views: Sequence[Mapping]) -> list[Mapping]:
    """The negative class for :func:`fit_quadratic_reference`.

    Unparsed traces are dropped.  They are auto-labelled incorrect upstream, so
    a negative class that keeps them is partly a *truncation* class, and the
    resulting score would separate correct from unparsed rather than correct
    from wrong.  RMD is not exposed to that: its second Gaussian pools every
    training trace, so unparsed tokens sit in both terms of the difference and
    largely cancel.  The positive side needs no such filter -- an unparsed trace
    has no answer to be right about, so it is never in ``correct``.
    """
    return [
        view for view in train_views if not view["is_correct"] and not is_unparsed(view)
    ]


def fit_quadratic_reference(base_reference, train_views: Sequence[Mapping], layer: int):
    """RMD's estimator with a *discriminative* second Gaussian.

    ``rmd_tail_q20`` scores a token as ``d(correct) - d(all training tokens)``.
    Only the first Gaussian consumes correctness labels; the second is an
    unconditional background, which is what makes the feature positives-only.
    This replaces that background with a Gaussian over *incorrect* traces, so
    the score becomes ``d(correct) - d(incorrect)`` -- a two-class quadratic
    discriminant with per-class covariances.

    Everything else is shared with the RMD reference by construction: the same
    ``base_reference`` supplies the PCA basis and the correct-trace Gaussian
    unchanged, and the projection helper is the one the background uses.  The
    only free variable against ``rmd_tail_q20`` is therefore whether the
    negative class was labelled, which is what the comparison is for.

    Returns ``None`` on a negative class too small or too degenerate to fit,
    matching :func:`analyze.extend_reference_with_background_safe`.
    """
    return extend_reference_with_background_safe(
        base_reference, incorrect_side_views(train_views), layer
    )


def fit_token_probe(
    train_views: Sequence[Mapping],
    projected_tails: Mapping[int, np.ndarray],
    *,
    max_tokens_per_trace: int | None,
    seed: int = 42,
) -> dict:
    """Supervised LDA over *individual* tail tokens, pooled across traces.

    The pooling-order match to RMD.  Each tail token becomes one training row
    carrying its trace's label, and a trace is later scored by averaging the
    per-token decision values -- the same "score every token, then average"
    order that ``compute_relative_mahal_distances`` plus a mean produces for
    ``rmd_tail_q20``.  ``probe_hidden_tail_q20`` instead averages the tokens
    first and classifies once, so against RMD it varies supervision and pooling
    order together; this estimator varies only supervision.

    Labels are per *trace*, so every token of a correct trace is labelled
    correct.  That is the honest reading of the supervision actually available
    at this budget -- nothing labels individual tokens -- and it is also what
    makes the comparison fair, since the one-class reference pools tokens from
    correct traces on exactly the same basis.

    Unparsed traces are excluded, matching :func:`fit_hidden_state_probe`: they
    are auto-labelled incorrect upstream, so keeping them would let the probe
    win by detecting truncation.  The per-trace token subsample is the same
    device the reference fits use, so both sides of the comparison consume a
    comparable number of tokens at a given label budget.
    """
    blocks, labels = [], []
    n_traces = 0
    for view in train_views:
        if is_unparsed(view):
            continue
        trace_id = int(view["trace_id"])
        projected = np.asarray(projected_tails[trace_id], dtype=float)
        if projected.ndim != 2 or not len(projected):
            continue
        index = reference_subsample_indices(
            int(projected.shape[0]),
            max_tokens_per_trace,
            seed=seed,
            trace_id=trace_id,
        )
        block = projected[index]
        blocks.append(block)
        labels.append(np.full(len(block), int(bool(view["is_correct"])), dtype=int))
        n_traces += 1

    if not blocks:
        return {"scaler": None, "classifier": None, "n_train": 0,
                "n_train_traces": 0, "skipped": "no_parseable_traces"}
    matrix = np.concatenate(blocks, axis=0)
    y = np.concatenate(labels, axis=0)
    if len(np.unique(y)) < 2:
        return {"scaler": None, "classifier": None, "n_train": int(len(y)),
                "n_train_traces": n_traces, "skipped": "single_class"}

    scaler = StandardScaler().fit(matrix)
    # Same solver and shrinkage as the frozen region-mean probe, so the two
    # differ in what a row *is* and in nothing else.
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
        scaler.transform(matrix), y
    )
    return {
        "scaler": scaler,
        "classifier": classifier,
        "n_train": int(len(y)),
        "n_train_traces": n_traces,
        "skipped": None,
    }


def score_token_probe(projected: np.ndarray, fit: Mapping) -> float:
    """Mean per-token decision value over a trace's tail (higher = more correct)."""
    classifier, scaler = fit.get("classifier"), fit.get("scaler")
    if classifier is None or scaler is None:
        raise ValueError("no usable token-level probe was fitted")
    projected = np.asarray(projected, dtype=float)
    if projected.ndim != 2 or not len(projected):
        return float("nan")
    # classes_ is [0, 1], so decision_function points toward is_correct=1.
    return float(np.mean(classifier.decision_function(scaler.transform(projected))))


def fit_budget(
    views_by_prompt: Mapping[int, list[dict]],
    train_ids: Sequence[int],
    layer: int,
    pca_dim: int,
    *,
    max_tokens_per_trace: int | None = None,
    seed: int = 42,
) -> dict:
    """Fit every label-consuming geometry object on one budget's prompts."""
    train = [view for prompt_id in train_ids for view in views_by_prompt[int(prompt_id)]]
    correct = [view for view in train if view["is_correct"]]
    if not correct:
        raise RuntimeError(f"no correct traces in a budget of {len(train_ids)} prompts")

    reference = fit_correct_reference(correct, layer, pca_dim)
    if reference is None:
        raise RuntimeError(
            f"could not fit the correct-trace reference on {len(correct)} traces"
        )
    rmd_reference = extend_reference_with_background_safe(reference, train, layer)
    if rmd_reference is None:
        raise RuntimeError(
            f"could not fit the RMD background on {len(train)} traces"
        )

    pca = reference[0]
    projected_tails = {
        int(view["trace_id"]): pca.transform(
            np.asarray(view["tail"], dtype=np.float32)
        )
        for view in train
    }
    # region="full" over an already-sliced tail block is the tail_q20 region mean.
    probe = fit_hidden_state_probe(
        views_by_prompt, list(train_ids), projected_tails, region="full"
    )
    if probe["classifier"] is None:
        raise RuntimeError(f"no usable hidden-state probe ({probe['skipped']})")
    token_probe = fit_token_probe(
        train, projected_tails, max_tokens_per_trace=max_tokens_per_trace, seed=seed
    )
    if token_probe["classifier"] is None:
        raise RuntimeError(f"no usable token-level probe ({token_probe['skipped']})")
    incorrect = incorrect_side_views(train)
    qmd_reference = fit_quadratic_reference(reference, train, layer)
    if qmd_reference is None:
        raise RuntimeError(
            f"could not fit the quadratic negative class on {len(incorrect)} "
            f"parseable incorrect traces"
        )
    return {
        "pca": pca,
        "rmd_reference": rmd_reference,
        "qmd_reference": qmd_reference,
        "probe": probe,
        "token_probe": token_probe,
        "n_train_prompts": int(len(train_ids)),
        "n_train_traces": int(len(train)),
        "n_correct_traces": int(len(correct)),
        "n_incorrect_traces": int(len(incorrect)),
        "n_probe_traces": int(probe["n_train"]),
        "n_token_probe_rows": int(token_probe["n_train"]),
    }


def score_views(views: Sequence[Mapping], fit: Mapping) -> dict[int, dict[str, float]]:
    """Per-trace tail geometry under one budget's fit, both features at once."""
    pca = fit["pca"]
    scores: dict[int, dict[str, float]] = {}
    for view in views:
        tail = np.asarray(view["tail"], dtype=np.float32)
        relative = compute_relative_mahal_distances(tail, *fit["rmd_reference"])
        projected = pca.transform(tail)
        scores[int(view["trace_id"])] = {
            # Negated so higher is better, matching every other scorer.
            GEOMETRY_FEATURE: -float(np.mean(relative)),
            PROBE_FEATURE: score_hidden_state_probe(
                projected, view["entropies"], fit["probe"], "full"
            ),
            # Scored over the whole tail, not the fit subsample: RMD is too, so
            # the two features see identical token sets at evaluation time.
            TOKEN_PROBE_FEATURE: score_token_probe(projected, fit["token_probe"]),
            # Same call as the RMD line above, against a reference whose second
            # Gaussian saw only incorrect traces.
            QUADRATIC_FEATURE: -float(
                np.mean(compute_relative_mahal_distances(tail, *fit["qmd_reference"]))
            ),
        }
    return scores


def prompt_geometry(
    views_by_prompt: Mapping[int, list[dict]],
    prompt_ids: Sequence[int],
    trace_scores: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    """Average each trace score over the prompt's siblings, as the frozen run does."""
    aggregated: dict[int, dict[str, float]] = {}
    for prompt_id in prompt_ids:
        group = views_by_prompt[int(prompt_id)]
        aggregated[int(prompt_id)] = {
            name: float(
                np.mean([trace_scores[int(view["trace_id"])][name] for view in group])
            )
            for name in GEOMETRY_FEATURES
        }
    return aggregated


def crossfit_train_geometry(
    views_by_prompt: Mapping[int, list[dict]],
    train_ids: Sequence[int],
    layer: int,
    pca_dim: int,
    *,
    inner_folds: int,
    seed: int,
    max_tokens_per_trace: int | None = None,
    in_sample_fit: Mapping | None = None,
) -> dict[int, dict[str, float]]:
    """Held-out geometry scores for the *training* prompts.

    The logistic readout is fitted on these.  Scoring a training prompt with a
    reference that was fitted on it is optimistic for both features, but not
    equally so -- the discriminative probe has far more capacity to absorb its
    own training prompts than a one-class Gaussian does -- so an in-sample fit
    here would read as a probe advantage that the held-out evaluation never
    sees.  With ``inner_folds < 2`` the in-sample scores are used instead and
    the caller records that the curve is not honest on the training side.
    """
    train_ids = [int(prompt_id) for prompt_id in train_ids]
    if inner_folds < 2:
        if in_sample_fit is None:
            raise ValueError("inner_folds < 2 requires an in-sample fit to reuse")
        views = [view for prompt_id in train_ids for view in views_by_prompt[prompt_id]]
        return prompt_geometry(
            views_by_prompt, train_ids, score_views(views, in_sample_fit)
        )

    folds = min(int(inner_folds), len(train_ids))
    order = np.random.default_rng(int(seed)).permutation(len(train_ids))
    assignments = np.empty(len(train_ids), dtype=int)
    assignments[order] = np.arange(len(train_ids)) % folds

    geometry: dict[int, dict[str, float]] = {}
    for fold in range(folds):
        held_out = [train_ids[i] for i in range(len(train_ids)) if assignments[i] == fold]
        inner_train = [
            train_ids[i] for i in range(len(train_ids)) if assignments[i] != fold
        ]
        if not held_out or not inner_train:
            continue
        fit = fit_budget(
            views_by_prompt, inner_train, layer, pca_dim,
            max_tokens_per_trace=max_tokens_per_trace, seed=seed,
        )
        views = [view for prompt_id in held_out for view in views_by_prompt[prompt_id]]
        geometry.update(prompt_geometry(views_by_prompt, held_out, score_views(views, fit)))
        del fit
        gc.collect()
    return geometry


# ---------------------------------------------------------------------------
# Readouts and metrics
# ---------------------------------------------------------------------------

def feature_matrix(
    base_features: Mapping[int, Mapping],
    geometry: Mapping[int, Mapping[str, float]],
    prompt_ids: Sequence[int],
    names: Sequence[str],
) -> np.ndarray:
    rows = []
    for prompt_id in prompt_ids:
        entry = dict(base_features[int(prompt_id)])
        entry.update(geometry.get(int(prompt_id), {}))
        rows.append([float(entry.get(name, float("nan"))) for name in names])
    return np.asarray(rows, dtype=float)


def fit_predict_logistic(
    train_features: np.ndarray,
    train_outcomes: np.ndarray,
    eval_features: np.ndarray,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Train-on-budget, predict-on-held-out logistic readout.

    Deliberately not :func:`incremental_abstention.crossfit_logistic_predictions`:
    that cross-fits *within* one population, whereas a label budget is an
    explicit train/evaluate split.  The imputation and scaling are the frozen
    module's, so B0 is the same object it is everywhere else.
    """
    from sklearn.linear_model import LogisticRegression

    train_scaled, eval_scaled = _impute_and_scale(train_features, eval_features)
    outcomes = np.asarray(train_outcomes, dtype=float)
    finite = np.isfinite(outcomes)
    if finite.sum() < len(outcomes):
        train_scaled = train_scaled[finite]
        outcomes = outcomes[finite]
    labels = outcomes.astype(int)
    if len(labels) == 0:
        return np.full(len(eval_scaled), float("nan"))
    if len(np.unique(labels)) < 2:
        return np.full(len(eval_scaled), float(np.mean(labels)))
    model = LogisticRegression(max_iter=2000, random_state=seed)
    try:
        model.fit(train_scaled, labels)
    except ValueError:
        return np.full(len(eval_scaled), float(np.mean(labels)))
    return model.predict_proba(eval_scaled)[:, 1]


def chance_aurc(n: int, base_accuracy: float) -> float:
    """The flat risk-coverage level on ``n`` prompts at that base accuracy.

    AURC is ``(1 - 1/n) - AUACC``, and a risk-coverage curve pinned at the base
    rate across every coverage level scores exactly this.  Subtracting it is
    what makes a *level* comparable across evaluation sets of different size and
    difficulty; a raw AURC is not.

    It is a recentering, not an exact null.  An actually-random ranking of a
    finite sample lands slightly above this -- the lowest coverage levels average
    one or two prompts, so the curve is noisy rather than flat there -- by around
    0.01 at n=100.  That offset is a property of the evaluation set, so it is
    shared by every readout scored on it and cancels in the paired deltas, which
    is where the comparison of interest lives.
    """
    if n <= 0:
        return float("nan")
    return float((1.0 - 1.0 / n) - float(base_accuracy))


def safe_auroc(outcomes: np.ndarray, scores: np.ndarray) -> float | None:
    outcomes = np.asarray(outcomes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    usable = np.isfinite(outcomes) & np.isfinite(scores)
    if usable.sum() < 2 or len(np.unique(outcomes[usable])) < 2:
        return None
    return float(roc_auc_score(outcomes[usable].astype(int), scores[usable]))


def summarize_replicates(values: Sequence[float | None]) -> dict:
    """Median and a 10--90 band over label draws, plus the mean.

    The spread reported here is variation over *which prompts got labelled* --
    the quantity a deployment cares about -- not bootstrap noise on a fixed
    sample, so it is summarized by percentiles over replicates rather than by a
    resampled confidence interval.
    """
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"n_replicates": 0, "median": None, "mean": None, "p10": None, "p90": None}
    array = np.asarray(finite, dtype=float)
    return {
        "n_replicates": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def sign_summary(values: Sequence[float | None], *, negative_is_better: bool) -> dict:
    """How often the delta landed on the expected side, with a two-sided sign test."""
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"n_replicates": 0, "win_rate": None, "p_sign": None}
    wins = sum(
        (value < 0) if negative_is_better else (value > 0) for value in finite
    )
    n = len(finite)
    # Two-sided exact binomial against p=0.5, computed directly to avoid a scipy
    # dependency in a module that otherwise only needs numpy and sklearn.
    tail = sum(math.comb(n, k) for k in range(min(wins, n - wins) + 1)) / (2.0 ** n)
    return {
        "n_replicates": int(n),
        "win_rate": float(wins) / n,
        "p_sign": float(min(1.0, 2.0 * tail)),
    }


POOLED_QUANTITIES: tuple[tuple[str, bool], ...] = (
    ("delta_aurc_B0_rmd_minus_B0", True),
    ("delta_aurc_B0_probe_minus_B0", True),
    ("delta_aurc_B0_rmd_minus_B0_probe", True),
    ("delta_aurc_B0_both_minus_B0_rmd", True),
    ("auroc_rmd_minus_probe", False),
    # The pooling-matched pair. Absent from results files written before
    # ``probe_token_tail_q20`` existed; the summarizers return an empty entry
    # for a quantity no replicate carries, so an old JSON still reads.
    ("delta_aurc_B0_rmd_minus_B0_token_probe", True),
    ("auroc_rmd_minus_token_probe", False),
    # The negative-label pair. ``qmd_tail_q20`` is RMD's own estimator with the
    # unconditional background swapped for an incorrect-trace Gaussian, so this
    # is the one comparison in which supervision moves alone.
    ("delta_aurc_B0_rmd_minus_B0_qmd", True),
    ("delta_aurc_B0_qmd_minus_B0_token_probe", True),
    ("delta_aurc_B0_qmd_minus_B0", True),
    ("auroc_rmd_minus_qmd", False),
)

#: Pooled AUROC gaps derived from the per-feature columns rather than stored.
_AUROC_DELTA_AGAINST: dict[str, str] = {
    "auroc_rmd_minus_probe": PROBE_FEATURE,
    "auroc_rmd_minus_token_probe": TOKEN_PROBE_FEATURE,
    "auroc_rmd_minus_qmd": QUADRATIC_FEATURE,
}


def _pooled_value(row: Mapping, key: str) -> float | None:
    """One replicate's value for a pooled quantity, or None if it is missing.

    The ``auroc_rmd_minus_*`` gaps are derived rather than stored: the
    per-replicate row carries each feature's AUROC, and the difference is the
    base-rate-free view of the same comparison the AURC deltas make.
    """
    against = _AUROC_DELTA_AGAINST.get(key)
    if against is not None:
        rmd = row.get(f"auroc_{GEOMETRY_FEATURE}")
        probe = row.get(f"auroc_{against}")
        if rmd is None or probe is None:
            return None
        return float(rmd) - float(probe)
    value = row.get(key)
    return None if value is None else float(value)


def pooled_sign_table(models: Sequence[Mapping], budgets: Sequence[int]) -> list[dict]:
    """The three models pooled, one observation per label draw.

    Each per-model table rests on ten draws against an evaluation set of under a
    hundred prompts, which cannot separate two readouts differing by 0.02 AURC.
    The models are separate datasets, so the *direction* pools even where no
    single model resolves it.

    What does not pool is the p-value: draws inside a model share an evaluation
    set and are not independent, so ``p_sign`` on thirty observations overstates
    its own resolution.  ``models_agreeing`` is the honest statistic -- it asks
    how many of three independent datasets put their own median on the same
    side, and is reported alongside for exactly that reason.
    """
    table: list[dict] = []
    for budget in budgets:
        quantities: dict[str, dict] = {}
        for key, negative_is_better in POOLED_QUANTITIES:
            per_model: dict[str, dict] = {}
            pooled: list[float | None] = []
            for model in models:
                values = [
                    _pooled_value(row, key)
                    for row in model["replicate_rows"]
                    if int(row["budget"]) == int(budget)
                ]
                per_model[model["label"]] = sign_summary(
                    values, negative_is_better=negative_is_better
                )
                pooled.extend(values)
            quantities[key] = {
                "pooled": sign_summary(pooled, negative_is_better=negative_is_better),
                "median": summarize_replicates(pooled)["median"],
                "per_model": per_model,
                "models_agreeing": sum(
                    1
                    for summary in per_model.values()
                    if summary["win_rate"] is not None and summary["win_rate"] > 0.5
                ),
                "n_models": len(per_model),
            }
        table.append({"budget": int(budget), "quantities": quantities})
    return table


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------

def replicate_split(
    prompt_ids: Sequence[int],
    eval_pool: Sequence[int],
    *,
    max_budget: int,
    seed: int,
    replicate: int,
) -> tuple[list[int], list[int]]:
    """One permutation: nested training sets, and one evaluation set for all budgets.

    The evaluation set is the complement of the *largest* budget, so that every
    budget in this replicate is scored on identical prompts and a difference
    between two curves cannot be an evaluation-set difference.
    """
    ids = [int(prompt_id) for prompt_id in prompt_ids]
    if max_budget >= len(ids):
        raise ValueError(
            f"largest budget {max_budget} leaves no prompts to evaluate on "
            f"({len(ids)} available)"
        )
    rng = np.random.default_rng([int(seed), int(replicate)])
    permuted = [ids[index] for index in rng.permutation(len(ids))]
    held_out = set(permuted[max_budget:])
    evaluation = [prompt_id for prompt_id in sorted(eval_pool) if prompt_id in held_out]
    if not evaluation:
        raise ValueError("no evaluation prompts survive the population filter")
    return permuted, evaluation


def run_replicate(
    *,
    replicate: int,
    budgets: Sequence[int],
    permutation: Sequence[int],
    eval_ids: Sequence[int],
    views_by_prompt: Mapping[int, list[dict]],
    base_features: Mapping[int, Mapping],
    layer: int,
    pca_dim: int,
    inner_folds: int,
    seed: int,
    max_tokens_per_trace: int | None = None,
) -> list[dict]:
    eval_views = [view for prompt_id in eval_ids for view in views_by_prompt[int(prompt_id)]]
    outcomes = np.asarray(
        [float(base_features[int(prompt_id)]["outcome"]) for prompt_id in eval_ids],
        dtype=float,
    )
    base_accuracy = float(np.mean(outcomes))
    rows = []

    for budget in budgets:
        started = time.time()
        train_ids = [int(prompt_id) for prompt_id in permutation[:budget]]
        fit = fit_budget(
            views_by_prompt, train_ids, layer, pca_dim,
            max_tokens_per_trace=max_tokens_per_trace, seed=seed,
        )
        eval_geometry = prompt_geometry(
            views_by_prompt, eval_ids, score_views(eval_views, fit)
        )
        train_geometry = crossfit_train_geometry(
            views_by_prompt,
            train_ids,
            layer,
            pca_dim,
            inner_folds=inner_folds,
            seed=seed + 7919 * replicate + budget,
            max_tokens_per_trace=max_tokens_per_trace,
            in_sample_fit=fit,
        )
        train_outcomes = np.asarray(
            [float(base_features[prompt_id]["outcome"]) for prompt_id in train_ids],
            dtype=float,
        )

        row = {
            "replicate": int(replicate),
            "budget": int(budget),
            "n_eval": int(len(eval_ids)),
            "eval_base_accuracy": base_accuracy,
            "n_train_traces": fit["n_train_traces"],
            "n_correct_traces": fit["n_correct_traces"],
            "n_incorrect_traces": fit["n_incorrect_traces"],
            "n_probe_traces": fit["n_probe_traces"],
            "n_token_probe_rows": fit["n_token_probe_rows"],
            "train_base_accuracy": float(np.mean(train_outcomes)),
        }

        for name in GEOMETRY_FEATURES:
            row[f"auroc_{name}"] = safe_auroc(
                outcomes,
                np.asarray([eval_geometry[int(pid)][name] for pid in eval_ids], dtype=float),
            )

        for readout, names in READOUT_SPECS.items():
            probabilities = fit_predict_logistic(
                feature_matrix(base_features, train_geometry, train_ids, names),
                train_outcomes,
                feature_matrix(base_features, eval_geometry, eval_ids, names),
                seed=seed,
            )
            metrics = prompt_metrics(probabilities, outcomes)
            row[f"aurc_{readout}"] = metrics["aurc"]
            row[f"auacc_{readout}"] = metrics["auacc"]
            row[f"excess_aurc_{readout}"] = (
                None
                if metrics["aurc"] is None
                else float(metrics["aurc"] - chance_aurc(int(metrics["n"]), base_accuracy))
            )

        for left, right in PAIRED_DELTAS:
            key = f"delta_aurc_{left}_minus_{right}"
            if row[f"aurc_{left}"] is None or row[f"aurc_{right}"] is None:
                row[key] = None
            else:
                row[key] = float(row[f"aurc_{left}"] - row[f"aurc_{right}"])

        row["seconds"] = float(time.time() - started)
        _status(
            f"  replicate {replicate} budget {budget:>3}: "
            f"AUROC rmd={_fmt(row[f'auroc_{GEOMETRY_FEATURE}'])} "
            f"probe={_fmt(row[f'auroc_{PROBE_FEATURE}'])} "
            f"tokprobe={_fmt(row[f'auroc_{TOKEN_PROBE_FEATURE}'])} "
            f"qmd={_fmt(row[f'auroc_{QUADRATIC_FEATURE}'])} "
            f"AURC B0={_fmt(row['aurc_B0'])} "
            f"+rmd={_fmt(row['aurc_B0_rmd'])} +probe={_fmt(row['aurc_B0_probe'])} "
            f"+tokprobe={_fmt(row['aurc_B0_token_probe'])} "
            f"+qmd={_fmt(row['aurc_B0_qmd'])} "
            f"[{row['seconds']:.0f}s]"
        )
        rows.append(row)
        del fit
        gc.collect()
    return rows


def aggregate_curves(rows: Sequence[Mapping], budgets: Sequence[int]) -> dict:
    """Collapse per-replicate rows into one entry per budget.

    Every column is read with ``.get``.  Rows come either from this run or from
    a finished results JSON replayed through ``--report_from``, and a JSON
    written before a comparator existed carries none of its columns; missing
    means "no replicate measured this", which is what ``summarize_replicates``
    already reports for an all-``None`` list.
    """

    def auroc_gap(at_budget: Sequence[Mapping], against: str) -> dict:
        """``rmd_tail_q20``'s solo AUROC minus one comparator's, per replicate."""
        return summarize_replicates(
            [
                None
                if row.get(f"auroc_{GEOMETRY_FEATURE}") is None
                or row.get(f"auroc_{against}") is None
                else row[f"auroc_{GEOMETRY_FEATURE}"] - row[f"auroc_{against}"]
                for row in at_budget
            ]
        )

    curves = []
    for budget in budgets:
        at_budget = [row for row in rows if int(row["budget"]) == int(budget)]
        entry = {
            "budget": int(budget),
            "n_replicates": len(at_budget),
            "n_eval": summarize_replicates([row["n_eval"] for row in at_budget]),
            "eval_base_accuracy": summarize_replicates(
                [row["eval_base_accuracy"] for row in at_budget]
            ),
            "n_train_traces": summarize_replicates(
                [row["n_train_traces"] for row in at_budget]
            ),
            "feature_auroc": {
                name: summarize_replicates(
                    [row.get(f"auroc_{name}") for row in at_budget]
                )
                for name in GEOMETRY_FEATURES
            },
            "aurc": {
                readout: summarize_replicates(
                    [row.get(f"aurc_{readout}") for row in at_budget]
                )
                for readout in READOUT_SPECS
            },
            "excess_aurc": {
                readout: summarize_replicates(
                    [row.get(f"excess_aurc_{readout}") for row in at_budget]
                )
                for readout in READOUT_SPECS
            },
            "delta_aurc": {},
            "feature_auroc_delta": auroc_gap(at_budget, PROBE_FEATURE),
            # The same gap against the pooling-matched probe. Where the two
            # deltas disagree, the difference is pooling order, not supervision.
            "feature_auroc_delta_token": auroc_gap(at_budget, TOKEN_PROBE_FEATURE),
            # And against the quadratic discriminant, where only supervision
            # differs -- the gap that is left once pooling and form are matched.
            "feature_auroc_delta_qmd": auroc_gap(at_budget, QUADRATIC_FEATURE),
        }
        for left, right in PAIRED_DELTAS:
            key = f"{left}_minus_{right}"
            values = [row.get(f"delta_aurc_{key}") for row in at_budget]
            entry["delta_aurc"][key] = {
                **summarize_replicates(values),
                **sign_summary(values, negative_is_better=True),
            }
        curves.append(entry)
    return {"curves": curves}


def crossing_budget(curves: Sequence[Mapping]) -> dict:
    """Where the one-class statistic stops being the better use of a label.

    Reported off the median paired delta ``B0_rmd - B0_probe``: negative means
    geometry is the cheaper feature at that budget (AURC, lower is better).
    Linear interpolation in log2(budget) between the bracketing budgets.
    """
    points = [
        (int(entry["budget"]), entry["delta_aurc"]["B0_rmd_minus_B0_probe"]["median"])
        for entry in curves
        if entry["delta_aurc"]["B0_rmd_minus_B0_probe"]["median"] is not None
    ]
    if len(points) < 2:
        return {"crossed": None, "budget": None, "note": "too few budgets to interpolate"}
    for (left_n, left_value), (right_n, right_value) in zip(points, points[1:]):
        if left_value < 0.0 <= right_value:
            span = right_value - left_value
            weight = 0.0 if span == 0 else (0.0 - left_value) / span
            budget = 2.0 ** (
                math.log2(left_n) + weight * (math.log2(right_n) - math.log2(left_n))
            )
            return {
                "crossed": True,
                "budget": float(budget),
                "bracket": [left_n, right_n],
                "note": "geometry is the better feature below this budget",
            }
    if all(value < 0.0 for _, value in points):
        return {
            "crossed": False,
            "budget": None,
            "note": "geometry is ahead at every budget tested",
        }
    if all(value >= 0.0 for _, value in points):
        return {
            "crossed": False,
            "budget": None,
            "note": "the probe is ahead at every budget tested",
        }
    return {"crossed": False, "budget": None, "note": "the deltas are not monotone"}


def analyze_model(
    *,
    label: str,
    oof_csv: str | Path,
    data_dir: str | Path,
    layer: int | None,
    pca_dim: int,
    budgets: Sequence[int],
    replicates: int,
    inner_folds: int,
    max_tokens_per_trace: int | None,
    max_new_tokens: int | None,
    expected_traces: int,
    load_workers: int,
    seed: int,
    show_progress: bool,
) -> dict:
    rows, selected_layer = select_layer_rows(
        _read_oof(oof_csv), layer, context=str(oof_csv)
    )
    base_features = aggregate_prompt_features(
        rows,
        max_new_tokens=max_new_tokens,
        data_dir=str(data_dir),
        expected_traces=expected_traces,
    )
    eval_pool = sorted(
        prompt_id
        for prompt_id, entry in base_features.items()
        if entry["valid_plurality"] and entry["cap_count"] == 0
    )

    _status(f"[{label}] loading layer {selected_layer} from {data_dir}")
    traces = load_all_traces(
        str(data_dir),
        [selected_layer],
        max_workers=load_workers,
        show_progress=show_progress,
        include_auxiliary=True,
        auxiliary_fields={"entropies"},
        hidden_dtype=np.float16,
    )
    _status(f"[{label}] reducing {len(traces)} traces to tail + reference blocks")
    views = prepare_trace_views(
        traces, selected_layer, max_tokens_per_trace=max_tokens_per_trace, seed=seed
    )
    del traces
    gc.collect()
    views_by_prompt = group_views_by_prompt(views)
    resident_gib = sum(
        view["hiddens"][selected_layer].nbytes + view["tail"].nbytes for view in views
    ) / (1024**3)
    del views
    gc.collect()

    prompt_ids = sorted(set(views_by_prompt) & set(base_features))
    eval_pool = [prompt_id for prompt_id in eval_pool if prompt_id in views_by_prompt]
    budgets = sorted({int(budget) for budget in budgets})
    usable_budgets = [budget for budget in budgets if budget < len(prompt_ids)]
    skipped_budgets = [budget for budget in budgets if budget not in usable_budgets]
    if not usable_budgets:
        raise ValueError(
            f"[{label}] every budget in {budgets} is at least the prompt count "
            f"({len(prompt_ids)}); nothing would be left to evaluate on"
        )
    max_budget = max(usable_budgets)
    _status(
        f"[{label}] {len(prompt_ids)} prompts, {len(eval_pool)} in the "
        f"{HEADLINE_POPULATION} pool, {resident_gib:.1f} GiB resident"
    )

    rows_out: list[dict] = []
    for replicate in range(int(replicates)):
        permutation, eval_ids = replicate_split(
            prompt_ids,
            eval_pool,
            max_budget=max_budget,
            seed=seed,
            replicate=replicate,
        )
        rows_out.extend(
            run_replicate(
                replicate=replicate,
                budgets=usable_budgets,
                permutation=permutation,
                eval_ids=eval_ids,
                views_by_prompt=views_by_prompt,
                base_features=base_features,
                layer=selected_layer,
                pca_dim=pca_dim,
                inner_folds=inner_folds,
                seed=seed,
                max_tokens_per_trace=max_tokens_per_trace,
            )
        )

    aggregated = aggregate_curves(rows_out, usable_budgets)
    return {
        "label": label,
        "oof_csv": str(oof_csv),
        "data_dir": str(data_dir),
        "layer": int(selected_layer),
        "pca_dim": int(pca_dim),
        "population": HEADLINE_POPULATION,
        "n_prompts": int(len(prompt_ids)),
        "n_eval_pool": int(len(eval_pool)),
        "budgets": [int(budget) for budget in usable_budgets],
        "skipped_budgets": [int(budget) for budget in skipped_budgets],
        "replicates": int(replicates),
        "inner_folds": int(inner_folds),
        "train_side_crossfit": bool(inner_folds >= 2),
        "max_tokens_per_trace": None if max_tokens_per_trace is None else int(max_tokens_per_trace),
        "seed": int(seed),
        "resident_gib": float(resident_gib),
        "replicate_rows": rows_out,
        **aggregated,
        "crossing": crossing_budget(aggregated["curves"]),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _band(summary: Mapping | None, digits: int = 3) -> str:
    if not summary or summary.get("median") is None:
        return "n/a"
    return (
        f"{_fmt(summary['median'], digits)} "
        f"[{_fmt(summary['p10'], digits)}, {_fmt(summary['p90'], digits)}]"
    )


def _model_section(result: Mapping) -> list[str]:
    lines = [
        f"## {result['label']}",
        "",
        f"Layer {result['layer']}, PCA {result['pca_dim']}, "
        f"{result['n_prompts']} prompts ({result['n_eval_pool']} in the "
        f"`{result['population']}` evaluation pool), "
        f"{result['replicates']} label draw{'' if result['replicates'] == 1 else 's'}, "
        f"{result['inner_folds']} inner folds on the training side, "
        f"{result['max_tokens_per_trace']} reference tokens per trace.",
        "",
    ]
    if result["skipped_budgets"]:
        lines += [
            f"Budgets skipped as larger than the prompt count: "
            f"{result['skipped_budgets']}.",
            "",
        ]
    if not result["train_side_crossfit"]:
        lines += [
            "**The training-side geometry is in-sample** (`--inner_folds` below 2), "
            "which flatters the discriminative probe more than the one-class "
            "reference. Treat the readout rows as an upper bound on the probe.",
            "",
        ]

    lines += [
        "### Feature AUROC against the budget",
        "",
        "Prompt-level AUROC of each feature alone, the base-rate-free view. "
        "Median over label draws, 10--90 band.",
        "",
        "`probe_token_tail_q20` is the pooling-matched probe: LDA per tail "
        "token, token scores averaged, which is the order `rmd_tail_q20` uses. "
        "`qmd_tail_q20` goes one step further and matches the decision "
        "function too -- it is RMD's own quadratic with the unconditional "
        "background replaced by an incorrect-trace Gaussian, so `rmd − qmd` is "
        "the gap left when only supervision differs.",
        "",
        "| labelled prompts | `rmd_tail_q20` | `probe_hidden_tail_q20` | "
        "`probe_token_tail_q20` | `qmd_tail_q20` | rmd − probe | "
        "rmd − token probe | rmd − qmd |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for entry in result["curves"]:
        lines.append(
            f"| {entry['budget']} "
            f"| {_band(entry['feature_auroc'][GEOMETRY_FEATURE])} "
            f"| {_band(entry['feature_auroc'][PROBE_FEATURE])} "
            f"| {_band(entry['feature_auroc'][TOKEN_PROBE_FEATURE])} "
            f"| {_band(entry['feature_auroc'][QUADRATIC_FEATURE])} "
            f"| {_band(entry['feature_auroc_delta'])} "
            f"| {_band(entry['feature_auroc_delta_token'])} "
            f"| {_band(entry['feature_auroc_delta_qmd'])} |"
        )

    lines += [
        "",
        "### AURC against the budget",
        "",
        "Lower is better. `excess` subtracts the chance AURC "
        "`(1 − 1/n) − base`, which is what makes a level comparable at all.",
        "",
        "| labelled prompts | eval n | base acc | B0 | B0+rmd | B0+probe | "
        "B0+token probe | B0+qmd | excess B0+rmd | excess B0+probe |",
        "|---:|---:|---|---|---|---|---|---|---|---|",
    ]
    for entry in result["curves"]:
        lines.append(
            f"| {entry['budget']} "
            f"| {_fmt(entry['n_eval']['median'], 0)} "
            f"| {_fmt(entry['eval_base_accuracy']['median'])} "
            f"| {_band(entry['aurc']['B0'])} "
            f"| {_band(entry['aurc']['B0_rmd'])} "
            f"| {_band(entry['aurc']['B0_probe'])} "
            f"| {_band(entry['aurc']['B0_token_probe'])} "
            f"| {_band(entry['aurc']['B0_qmd'])} "
            f"| {_band(entry['excess_aurc']['B0_rmd'])} "
            f"| {_band(entry['excess_aurc']['B0_probe'])} |"
        )

    lines += [
        "",
        "### Paired AURC deltas against the budget",
        "",
        "Paired inside a replicate: identical evaluation prompts, identical "
        "labelled prompts, one logistic each. Negative favours the left readout. "
        "`wins` is the share of label draws landing on that side.",
        "",
        "| labelled prompts | B0+rmd − B0 | B0+probe − B0 | B0+token probe − B0 | "
        "B0+rmd − B0+probe | wins | sign p | B0+rmd − B0+token probe | wins | sign p |",
        "|---:|---|---|---|---|---:|---:|---|---:|---:|",
    ]
    for entry in result["curves"]:
        head_to_head = entry["delta_aurc"]["B0_rmd_minus_B0_probe"]
        matched = entry["delta_aurc"]["B0_rmd_minus_B0_token_probe"]
        lines.append(
            f"| {entry['budget']} "
            f"| {_band(entry['delta_aurc']['B0_rmd_minus_B0'])} "
            f"| {_band(entry['delta_aurc']['B0_probe_minus_B0'])} "
            f"| {_band(entry['delta_aurc']['B0_token_probe_minus_B0'])} "
            f"| {_band(head_to_head)} "
            f"| {_fmt(head_to_head['win_rate'], 2)} "
            f"| {_fmt(head_to_head['p_sign'], 3)} "
            f"| {_band(matched)} "
            f"| {_fmt(matched['win_rate'], 2)} "
            f"| {_fmt(matched['p_sign'], 3)} |"
        )

    lines += [
        "",
        "### The supervision ladder",
        "",
        "Each rung releases one variable, so a gap that survives every rung is "
        "the one attributable to that rung alone. Negative favours the left "
        "readout.",
        "",
        "| rung | what it releases |",
        "|---|---|",
        "| `B0+rmd − B0+qmd` | whether the negative class was labelled |",
        "| `B0+qmd − B0+token probe` | quadratic against linear decision function |",
        "| `B0+token probe − B0+probe` | score-then-pool against pool-then-score |",
        "",
        "| labelled prompts | rmd − qmd | wins | sign p | qmd − token probe | "
        "wins | sign p | qmd − B0 |",
        "|---:|---|---:|---:|---|---:|---:|---|",
    ]
    for entry in result["curves"]:
        supervision = entry["delta_aurc"]["B0_rmd_minus_B0_qmd"]
        form = entry["delta_aurc"]["B0_qmd_minus_B0_token_probe"]
        lines.append(
            f"| {entry['budget']} "
            f"| {_band(supervision)} "
            f"| {_fmt(supervision['win_rate'], 2)} "
            f"| {_fmt(supervision['p_sign'], 3)} "
            f"| {_band(form)} "
            f"| {_fmt(form['win_rate'], 2)} "
            f"| {_fmt(form['p_sign'], 3)} "
            f"| {_band(entry['delta_aurc']['B0_qmd_minus_B0'])} |"
        )

    crossing = result["crossing"]
    lines += [
        "",
        "### Crossing",
        "",
        (
            f"{crossing['note']}."
            if crossing.get("budget") is None
            else f"Median `B0_rmd − B0_probe` crosses zero at about "
            f"**{crossing['budget']:.0f} labelled prompts** "
            f"(bracketed by {crossing['bracket']}); {crossing['note']}."
        ),
        "",
    ]
    return lines


def _pooled_cell(entry: Mapping) -> str:
    pooled = entry["pooled"]
    if not pooled["n_replicates"]:
        return "n/a"
    wins = int(round(pooled["win_rate"] * pooled["n_replicates"]))
    return (
        f"{_fmt(entry['median'])} · {wins}/{pooled['n_replicates']} · "
        f"p={_fmt(pooled['p_sign'])}"
    )


def _pooled_section(result: Mapping) -> list[str]:
    models = result["models"]
    if len(models) < 2:
        return []
    table = pooled_sign_table(models, result["budgets"])
    n_models = len(models)
    lines = [
        "## Across the models",
        "",
        f"Every label draw from all {n_models} models, pooled per budget. Cells "
        "read `median delta · draws on that side · sign p`; negative favours the "
        "left readout except in the last column, where positive favours geometry. "
        "`agree` counts models whose own median lands on the geometry side of "
        "`B0+rmd − B0+probe`.",
        "",
        "The models are separate datasets so the direction pools, but the draws "
        "inside a model share an evaluation set, so the pooled `p` is a summary "
        "of consistency and not a test on independent observations. `agree` is "
        "the statistic that does not depend on that assumption.",
        "",
        "| labelled prompts | `B0+rmd − B0` | `B0+probe − B0` | `B0+rmd − B0+probe` | agree | "
        "`B0+rmd − B0+token probe` | agree | `B0+both − B0+rmd` | AUROC rmd − probe |",
        "|---:|---|---|---|---:|---|---:|---|---|",
    ]
    for row in table:
        quantities = row["quantities"]
        versus_probe = quantities["delta_aurc_B0_rmd_minus_B0_probe"]
        versus_token = quantities["delta_aurc_B0_rmd_minus_B0_token_probe"]
        lines.append(
            f"| {row['budget']} "
            f"| {_pooled_cell(quantities['delta_aurc_B0_rmd_minus_B0'])} "
            f"| {_pooled_cell(quantities['delta_aurc_B0_probe_minus_B0'])} "
            f"| {_pooled_cell(versus_probe)} "
            f"| {versus_probe['models_agreeing']}/{versus_probe['n_models']} "
            f"| {_pooled_cell(versus_token)} "
            f"| {versus_token['models_agreeing']}/{versus_token['n_models']} "
            f"| {_pooled_cell(quantities['delta_aurc_B0_both_minus_B0_rmd'])} "
            f"| {_pooled_cell(quantities['auroc_rmd_minus_probe'])} |"
        )

    lines += [
        "",
        "Pooled supervision ladder. `rmd − qmd` holds pooling order *and* the "
        "decision function fixed, so it is the only column in this report where "
        "supervision moves alone.",
        "",
        "| labelled prompts | `B0+rmd − B0+qmd` | agree | "
        "`B0+qmd − B0+token probe` | agree | `B0+qmd − B0` | AUROC rmd − qmd |",
        "|---:|---|---:|---|---:|---|---|",
    ]
    for row in table:
        quantities = row["quantities"]
        supervision = quantities["delta_aurc_B0_rmd_minus_B0_qmd"]
        form = quantities["delta_aurc_B0_qmd_minus_B0_token_probe"]
        lines.append(
            f"| {row['budget']} "
            f"| {_pooled_cell(supervision)} "
            f"| {supervision['models_agreeing']}/{supervision['n_models']} "
            f"| {_pooled_cell(form)} "
            f"| {form['models_agreeing']}/{form['n_models']} "
            f"| {_pooled_cell(quantities['delta_aurc_B0_qmd_minus_B0'])} "
            f"| {_pooled_cell(quantities['auroc_rmd_minus_qmd'])} |"
        )
    lines.append("")
    return lines


def write_report(result: Mapping, path: str | Path) -> None:
    lines = [
        "# Label-efficiency curves: one-class geometry versus a supervised probe",
        "",
        "`rmd_tail_q20` fits a Gaussian on correct traces only; "
        "`probe_hidden_tail_q20` fits an LDA on both classes over the same "
        "PCA-projected tail means. At the full label budget the probe is ahead, "
        "so the only deployment claim geometry can carry is that it needs fewer "
        "labels. These curves are that claim, or its refutation.",
        "",
        "At each budget the PCA basis, the correct-trace Gaussian, the background "
        "Gaussian, the LDA, and the logistic readout are all refitted from that "
        "budget's prompts alone. Training sets are nested along one permutation "
        "per replicate; the evaluation set is the headline-population complement "
        "of the largest budget and is held fixed across budgets, so every number "
        "in a row is scored on the same prompts.",
        "",
    ]
    for model in result["models"]:
        lines.extend(_model_section(model))
    lines.extend(_pooled_section(result))
    lines += [
        "## Scope",
        "",
        "- Reference fits see a fixed per-trace token subsample "
        f"(`max_tokens_per_trace`), not the whole sequence. The frozen pipeline "
        "caps pooled reference tokens instead; this is the same device moved per "
        "trace so the token count still grows with the budget. Applied "
        "identically to both features.",
        "- Only the 20% tail block is retained, which is lossless for both "
        "features under comparison and for nothing else.",
        "- The PCA solver is pinned to `randomized`. The frozen helper picks it "
        "by token count, switching to `full` below 200k pooled tokens -- a "
        "threshold that falls inside this sweep, so leaving it alone would put a "
        "change of decomposition in the middle of the curve.",
        "- The evaluation set is small by construction: the largest budget takes "
        "most of the prompts, and what is left is what can be scored. The spread "
        "quoted is over label draws, which is the relevant variation, but it "
        "carries that evaluation noise inside it.",
        "- Numbers here are not interchangeable with the frozen artifacts.",
        "",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_replicate_csv(result: Mapping, path: str | Path) -> None:
    rows = [row for model in result["models"] for row in model["replicate_rows"]]
    if not rows:
        return
    fieldnames = ["label"] + list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in result["models"]:
            for row in model["replicate_rows"]:
                writer.writerow({"label": model["label"], **row})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_model_spec(raw: str) -> dict:
    """``LABEL:OOF_CSV:DATA_DIR[:LAYER]``."""
    parts = raw.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            f"expected LABEL:OOF_CSV:DATA_DIR[:LAYER], got {raw!r}"
        )
    spec = {"label": parts[0], "oof_csv": parts[1], "data_dir": parts[2], "layer": None}
    if len(parts) == 4:
        spec["layer"] = int(parts[3])
    return spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        dest="models",
        help="LABEL:OOF_CSV:DATA_DIR[:LAYER], repeatable",
    )
    parser.add_argument(
        "--report_from",
        default=None,
        help="rebuild the report and replicate CSV from a finished results JSON "
        "instead of refitting; the JSON carries every per-replicate row, so a "
        "change to the write-up does not cost another sweep",
    )
    parser.add_argument("--output_dir", default="results/label_efficiency")
    parser.add_argument(
        "--budgets",
        default=",".join(str(budget) for budget in DEFAULT_BUDGETS),
        help="comma-separated labelled-prompt budgets",
    )
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument(
        "--inner_folds",
        type=int,
        default=3,
        help="folds used to score the training prompts honestly; below 2 uses "
        "in-sample scores and flatters the probe",
    )
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument(
        "--max_tokens_per_trace",
        type=int,
        default=256,
        help="per-trace token cap for every reference fit (0 for no cap)",
    )
    parser.add_argument(
        "--max_reference_tokens",
        type=int,
        default=2_000_000,
        help="pooled reference-token cap, matching params.yaml",
    )
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--expected_traces", type=int, default=8)
    parser.add_argument("--load_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute_dtype", default="float32")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    if args.report_from:
        result = json.loads(Path(args.report_from).read_text(encoding="utf-8"))
        output_dir.mkdir(parents=True, exist_ok=True)
        write_report(result, output_dir / "label_efficiency_report.md")
        write_replicate_csv(result, output_dir / "label_efficiency_replicates.csv")
        _status(f"rebuilt {output_dir}/label_efficiency_report.md")
        return
    if not args.models:
        raise SystemExit("--model is required unless --report_from is given")

    set_compute_dtype(np.dtype(args.compute_dtype))
    set_max_reference_tokens(args.max_reference_tokens or None)
    budgets = [int(part) for part in str(args.budgets).split(",") if part.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for spec in args.models:
        started = time.time()
        models.append(
            analyze_model(
                label=spec["label"],
                oof_csv=spec["oof_csv"],
                data_dir=spec["data_dir"],
                layer=spec["layer"],
                pca_dim=args.pca_dim,
                budgets=budgets,
                replicates=args.replicates,
                inner_folds=args.inner_folds,
                max_tokens_per_trace=args.max_tokens_per_trace or None,
                max_new_tokens=args.max_new_tokens,
                expected_traces=args.expected_traces,
                load_workers=args.load_workers,
                seed=args.seed,
                show_progress=args.progress,
            )
        )
        _status(f"[{spec['label']}] done in {time.time() - started:.0f}s")
        gc.collect()

    result = {
        "budgets": budgets,
        "replicates": int(args.replicates),
        "inner_folds": int(args.inner_folds),
        "seed": int(args.seed),
        "models": models,
    }
    (output_dir / "label_efficiency_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_report(result, output_dir / "label_efficiency_report.md")
    write_replicate_csv(result, output_dir / "label_efficiency_replicates.csv")
    _status(f"wrote {output_dir}/label_efficiency_report.md")


if __name__ == "__main__":
    main()
