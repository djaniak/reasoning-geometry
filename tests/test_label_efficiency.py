import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import (
    compute_relative_mahal_distances,
    extend_reference_with_background_safe,
)
from incremental_abstention import prompt_metrics
from prompt_decomposition import region_indices, score_localized_rmd, trace_region_mean

import label_efficiency as le
from label_efficiency import (
    GEOMETRY_FEATURE,
    PROBE_FEATURE,
    QUADRATIC_FEATURE,
    TOKEN_PROBE_FEATURE,
    chance_aurc,
    fit_correct_reference,
    crossing_budget,
    feature_matrix,
    fit_predict_logistic,
    fit_quadratic_reference,
    fit_token_probe,
    group_views_by_prompt,
    incorrect_side_views,
    score_token_probe,
    pooled_sign_table,
    prepare_trace_views,
    prompt_geometry,
    reference_subsample_indices,
    replicate_split,
    sign_summary,
    summarize_replicates,
)


def _trace(trace_id: int, prompt_id: int, n_tokens: int, *, correct: bool = True) -> dict:
    rng = np.random.default_rng(trace_id)
    return {
        "trace_id": trace_id,
        "idx": prompt_id,
        "sample_id": trace_id % 8,
        "is_correct": correct,
        "predicted_answer": "42",
        "entropies": rng.random(n_tokens).astype(np.float32),
        "hiddens": {7: rng.random((n_tokens, 5)).astype(np.float16)},
    }


def _token_probe_views(n_traces: int = 8, n_tokens: int = 40, dim: int = 4):
    """Separable tail tokens: correct traces sit at +1, incorrect at -1."""
    rng = np.random.default_rng(0)
    views, projected = [], {}
    for trace_id in range(n_traces):
        correct = trace_id % 2 == 0
        views.append(
            {
                "trace_id": trace_id,
                "prompt_id": trace_id // 2,
                "is_correct": correct,
                "predicted_answer": "42",
            }
        )
        centre = 1.0 if correct else -1.0
        projected[trace_id] = rng.normal(centre, 0.25, size=(n_tokens, dim))
    return views, projected


REFERENCE_LAYER = 7


def _synthetic_result(n_models: int = 2, budgets=(25, 50), replicates: int = 4) -> dict:
    """A results dict shaped exactly like a finished run's.

    Every column is generated from the module's own spec lists rather than
    hard-coded, so a comparator added to ``READOUT_SPECS`` or ``PAIRED_DELTAS``
    is automatically covered here.
    """
    rng = np.random.default_rng(7)
    models = []
    for index in range(n_models):
        rows = []
        for replicate in range(replicates):
            for budget in budgets:
                row = {
                    "replicate": replicate,
                    "budget": budget,
                    "n_eval": 80,
                    "eval_base_accuracy": 0.7,
                    "n_train_traces": budget * 8,
                    "n_correct_traces": budget * 5,
                    "n_incorrect_traces": budget * 3,
                    "n_probe_traces": budget * 8,
                    "n_token_probe_rows": budget * 8 * 16,
                    "train_base_accuracy": 0.7,
                    "seconds": 1.0,
                }
                for name in le.GEOMETRY_FEATURES:
                    row[f"auroc_{name}"] = float(rng.uniform(0.5, 0.9))
                for readout in le.READOUT_SPECS:
                    aurc = float(rng.uniform(0.1, 0.3))
                    row[f"aurc_{readout}"] = aurc
                    row[f"auacc_{readout}"] = 1.0 - aurc
                    row[f"excess_aurc_{readout}"] = aurc - 0.3
                for left, right in le.PAIRED_DELTAS:
                    row[f"delta_aurc_{left}_minus_{right}"] = float(
                        rng.uniform(-0.05, 0.05)
                    )
                rows.append(row)
        curves = le.aggregate_curves(rows, budgets)["curves"]
        models.append(
            {
                "label": f"model{index}",
                "layer": 21,
                "pca_dim": 128,
                "population": "cap_free_valid_plurality",
                "n_prompts": 500,
                "n_eval_pool": 392,
                "budgets": list(budgets),
                "skipped_budgets": [],
                "replicates": replicates,
                "inner_folds": 3,
                "train_side_crossfit": True,
                "max_tokens_per_trace": 256,
                "seed": 42,
                "resident_gib": 1.0,
                "replicate_rows": rows,
                "curves": curves,
                "crossing": le.crossing_budget(curves),
            }
        )
    return {
        "budgets": list(budgets),
        "replicates": replicates,
        "inner_folds": 3,
        "seed": 42,
        "models": models,
    }


def test_report_renders_every_quantity_it_references(tmp_path):
    """The pooled table reads `quantities[...]` by name, and `pooled_sign_table`
    only builds the names in POOLED_QUANTITIES. A comparator added to the report
    but not registered there raises KeyError *after* the whole sweep has run and
    the results JSON has been written -- which is exactly what happened once."""
    path = tmp_path / "report.md"

    le.write_report(_synthetic_result(), path)

    text = path.read_text(encoding="utf-8")
    assert "The supervision ladder" in text
    assert "`B0+rmd − B0+qmd`" in text


def test_report_renders_a_results_file_written_before_a_comparator_existed(tmp_path):
    """`--report_from` replays finished runs. A JSON predating a comparator
    carries none of its columns, and must still render rather than raise."""
    result = _synthetic_result()
    stale = f"delta_aurc_B0_rmd_minus_B0_{QUADRATIC_FEATURE}"
    for model in result["models"]:
        for row in model["replicate_rows"]:
            row.pop(f"auroc_{QUADRATIC_FEATURE}", None)
            row.pop("aurc_B0_qmd", None)
            row.pop("delta_aurc_B0_rmd_minus_B0_qmd", None)
            row.pop(stale, None)
        model["curves"] = le.aggregate_curves(
            model["replicate_rows"], result["budgets"]
        )["curves"]

    le.write_report(result, tmp_path / "report.md")

    assert (tmp_path / "report.md").exists()


def _reference_views(n_traces: int = 8, n_tokens: int = 40, dim: int = 5):
    """Views the frozen reference fitters accept, correct at +1, incorrect at −1.

    ``hiddens`` is what the Gaussians are fitted on and ``tail`` is what gets
    scored, matching :func:`label_efficiency.prepare_trace_views`.
    """
    rng = np.random.default_rng(1)
    views = []
    for trace_id in range(n_traces):
        correct = trace_id % 2 == 0
        centre = 1.0 if correct else -1.0
        views.append(
            {
                "trace_id": trace_id,
                "prompt_id": trace_id // 2,
                "is_correct": correct,
                "predicted_answer": "42",
                "hiddens": {
                    REFERENCE_LAYER: rng.normal(
                        centre, 0.3, size=(n_tokens, dim)
                    ).astype(np.float32)
                },
                "tail": rng.normal(centre, 0.3, size=(n_tokens // 2, dim)).astype(
                    np.float32
                ),
            }
        )
    return views


def _positive_reference(views, pca_dim: int = 3):
    return fit_correct_reference(
        [view for view in views if view["is_correct"]], REFERENCE_LAYER, pca_dim
    )


def _tail_score(view, reference) -> float:
    """``qmd_tail_q20`` as ``score_views`` computes it."""
    return -float(
        np.mean(compute_relative_mahal_distances(view["tail"], *reference))
    )


# ---------------------------------------------------------------------------
# The quadratic discriminant: RMD's estimator with a labelled negative class
# ---------------------------------------------------------------------------

def test_incorrect_side_drops_correct_and_unparsed_traces():
    """Unparsed traces are auto-labelled incorrect upstream. A negative class
    that keeps them is partly a truncation class, and the score would separate
    correct from unparsed rather than correct from wrong."""
    views = _reference_views(n_traces=8)
    views[1]["predicted_answer"] = ""
    views[5]["predicted_answer"] = None

    negative = incorrect_side_views(views)

    assert [view["trace_id"] for view in negative] == [3, 7]


def test_quadratic_reference_reuses_the_positive_side_unchanged():
    """The PCA basis and the correct-trace Gaussian must be the *same* objects
    the RMD reference uses. Refitting either would let projection noise show up
    as a supervision effect, which is the one thing this comparison measures."""
    views = _reference_views()
    base = _positive_reference(views)

    quadratic = fit_quadratic_reference(base, views, REFERENCE_LAYER)

    assert quadratic[0] is base[0]
    assert np.array_equal(quadratic[1], base[1])
    assert np.array_equal(quadratic[2], base[2])


def test_quadratic_reference_does_not_reproduce_the_rmd_background():
    """The second Gaussian has to actually change. If the incorrect-only fit
    coincided with the all-trace one, `qmd_tail_q20` would be a rename of
    `rmd_tail_q20` and every delta between them would be noise."""
    views = _reference_views()
    base = _positive_reference(views)

    background = extend_reference_with_background_safe(base, views, REFERENCE_LAYER)
    quadratic = fit_quadratic_reference(base, views, REFERENCE_LAYER)

    assert not np.allclose(quadratic[3], background[3])


def test_quadratic_reference_scores_correct_tails_above_incorrect():
    views = _reference_views()
    quadratic = fit_quadratic_reference(
        _positive_reference(views), views, REFERENCE_LAYER
    )

    correct = [_tail_score(v, quadratic) for v in views if v["is_correct"]]
    incorrect = [_tail_score(v, quadratic) for v in views if not v["is_correct"]]

    assert min(correct) > max(incorrect)


def test_quadratic_reference_is_none_without_a_parseable_negative_class():
    """`fit_budget` turns this into an explicit error rather than fitting a
    degenerate second Gaussian."""
    views = _reference_views()
    for view in views:
        if not view["is_correct"]:
            view["predicted_answer"] = ""

    assert fit_quadratic_reference(
        _positive_reference(views), views, REFERENCE_LAYER
    ) is None


# ---------------------------------------------------------------------------
# The pooling-matched token probe
# ---------------------------------------------------------------------------

def test_token_probe_trains_on_tokens_not_traces():
    """The row unit is the token. That is the whole point of this estimator --
    it is what puts it in RMD's pooling order rather than the region-mean
    probe's, so a fit that collapsed each trace to one row would silently
    reintroduce the confound the feature exists to remove."""
    views, projected = _token_probe_views(n_traces=8, n_tokens=40)

    fit = fit_token_probe(views, projected, max_tokens_per_trace=None)

    assert fit["skipped"] is None
    assert fit["n_train_traces"] == 8
    assert fit["n_train"] == 8 * 40


def test_token_probe_honours_the_per_trace_token_cap():
    views, projected = _token_probe_views(n_traces=4, n_tokens=100)

    fit = fit_token_probe(views, projected, max_tokens_per_trace=10)

    assert fit["n_train"] == 4 * 10


def test_token_probe_excludes_unparsed_traces():
    """Unparsed traces are auto-labelled incorrect upstream, so keeping them
    would let the probe win by detecting truncation -- the same exclusion
    `fit_hidden_state_probe` makes."""
    views, projected = _token_probe_views(n_traces=8, n_tokens=20)
    views[1]["predicted_answer"] = ""
    views[3]["predicted_answer"] = None

    fit = fit_token_probe(views, projected, max_tokens_per_trace=None)

    assert fit["n_train_traces"] == 6
    assert fit["n_train"] == 6 * 20


def test_token_probe_scores_separable_tails_in_the_right_direction():
    views, projected = _token_probe_views(n_traces=8, n_tokens=40)
    fit = fit_token_probe(views, projected, max_tokens_per_trace=None)

    correct = score_token_probe(projected[0], fit)
    incorrect = score_token_probe(projected[1], fit)

    assert correct > incorrect


def test_token_probe_score_is_the_mean_of_the_token_scores():
    """Scoring must average *after* classifying. Averaging the tokens first and
    classifying once is the other probe; if this collapsed to that, the two
    features would be the same estimator under two names."""
    views, projected = _token_probe_views(n_traces=8, n_tokens=30)
    fit = fit_token_probe(views, projected, max_tokens_per_trace=None)
    tail = projected[2]

    per_token = fit["classifier"].decision_function(fit["scaler"].transform(tail))

    assert score_token_probe(tail, fit) == pytest.approx(float(np.mean(per_token)))


def test_token_probe_reports_a_single_class_instead_of_fitting_one():
    views, projected = _token_probe_views(n_traces=4, n_tokens=10)
    for view in views:
        view["is_correct"] = True

    fit = fit_token_probe(views, projected, max_tokens_per_trace=None)

    assert fit["classifier"] is None
    assert fit["skipped"] == "single_class"


# ---------------------------------------------------------------------------
# Trace reduction
# ---------------------------------------------------------------------------

def test_reference_subsample_keeps_everything_under_the_cap():
    indices = reference_subsample_indices(30, 256, seed=42, trace_id=3)

    assert indices.tolist() == list(range(30))


def test_reference_subsample_is_capped_sorted_and_stable_per_trace():
    first = reference_subsample_indices(1000, 64, seed=42, trace_id=3)
    again = reference_subsample_indices(1000, 64, seed=42, trace_id=3)
    other = reference_subsample_indices(1000, 64, seed=42, trace_id=4)

    assert first.size == 64
    assert first.tolist() == sorted(set(first.tolist()))
    # The same trace must contribute the same tokens at every budget, or two
    # budgets would differ in which tokens a shared prompt supplied and not only
    # in how many prompts they saw.
    assert first.tolist() == again.tolist()
    assert first.tolist() != other.tolist()


def test_the_retained_tail_block_is_the_tail_q20_region():
    """The whole reduction rests on this: region "full" over the sliced tail block
    is identically the tail_q20 region of the untouched trace."""
    trace = _trace(0, 0, 50)
    entropies = np.asarray(trace["entropies"], dtype=float)
    full = np.asarray(trace["hiddens"][7], dtype=float)

    view = prepare_trace_views([trace], 7, max_tokens_per_trace=None)[0]
    tail = np.asarray(view["tail"], dtype=float)

    assert tail.shape[0] == 10  # ceil(0.20 * 50)
    assert np.allclose(
        tail.mean(axis=0), trace_region_mean(full, entropies, "tail_q20")
    )
    assert np.allclose(view["entropies"], entropies[region_indices(entropies, "tail_q20")])


def test_the_tail_mean_of_a_distance_sequence_matches_the_frozen_scorer():
    entropies = np.random.default_rng(0).random(40)
    distances = np.arange(40, dtype=float)
    tail = region_indices(entropies, "tail_q20")

    assert -float(distances[tail].mean()) == pytest.approx(
        score_localized_rmd(distances, entropies, "tail_q20")
    )


def test_a_trace_whose_hidden_states_and_entropies_disagree_is_rejected():
    trace = _trace(0, 0, 20)
    trace["entropies"] = trace["entropies"][:19]

    with pytest.raises(ValueError, match="hidden rows"):
        prepare_trace_views([trace], 7, max_tokens_per_trace=None)


def test_views_group_by_prompt_in_sibling_order():
    traces = [_trace(2, 0, 20), _trace(1, 0, 20), _trace(3, 1, 20)]

    grouped = group_views_by_prompt(prepare_trace_views(traces, 7, max_tokens_per_trace=None))

    assert sorted(grouped) == [0, 1]
    assert [view["trace_id"] for view in grouped[0]] == [1, 2]


def test_the_pca_solver_does_not_change_with_the_budget():
    """The frozen helper picks the solver by token count and flips below 200k.
    A budget sweep crosses that threshold, so an unpinned solver would decompose
    the small budgets differently from the large ones -- a step change sitting in
    the middle of the curve being measured."""
    tiny = [_trace(i, 0, 40) for i in range(3)]
    ample = [_trace(i, 0, 40) for i in range(30)]

    for traces in (tiny, ample):
        pca, mu, precision = fit_correct_reference(traces, 7, 4)
        assert pca.svd_solver == "randomized"
        assert mu.shape == (4,)
        assert precision.shape == (4, 4)


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------

def test_the_evaluation_set_is_disjoint_from_every_budget():
    prompt_ids = list(range(60))
    eval_pool = list(range(0, 60, 2))

    permutation, evaluation = replicate_split(
        prompt_ids, eval_pool, max_budget=40, seed=42, replicate=0
    )

    assert set(evaluation).isdisjoint(permutation[:40])
    assert set(evaluation) <= set(eval_pool)
    # Nested training sets: a smaller budget is a prefix of a larger one, so the
    # curve is one trajectory rather than five unrelated draws.
    assert permutation[:10] == permutation[:40][:10]


def test_replicates_differ_and_repeat_exactly():
    prompt_ids = list(range(60))
    pool = list(range(60))

    first, _ = replicate_split(prompt_ids, pool, max_budget=40, seed=42, replicate=0)
    again, _ = replicate_split(prompt_ids, pool, max_budget=40, seed=42, replicate=0)
    second, _ = replicate_split(prompt_ids, pool, max_budget=40, seed=42, replicate=1)

    assert first == again
    assert first != second


def test_a_budget_that_swallows_the_prompts_is_refused():
    with pytest.raises(ValueError, match="leaves no prompts"):
        replicate_split(list(range(40)), list(range(40)), max_budget=40, seed=42, replicate=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_chance_aurc_tracks_a_random_ranking_without_pretending_to_be_exact():
    """A recentering, not a null: a random ranking of a finite sample lands a
    little above the flat-curve level, because the lowest coverage levels average
    one or two prompts and are noisy rather than flat."""
    rng = np.random.default_rng(0)
    outcomes = np.concatenate([np.ones(70), np.zeros(30)])
    drawn = [
        prompt_metrics(np.full(100, 0.5), rng.permutation(outcomes))["aurc"]
        for _ in range(500)
    ]

    gap = float(np.mean(drawn)) - chance_aurc(100, 0.7)

    assert 0.0 < gap < 0.02


def test_recentering_takes_the_base_rate_out_of_a_level():
    """Why excess AURC is reported at all: two evaluation sets scored equally
    well are not comparable on the raw level, because AURC inherits the base
    rate exactly as AUACC does."""
    rng = np.random.default_rng(7)

    def scored(base: float) -> tuple[float, float]:
        outcomes = (rng.random(400) < base).astype(float)
        probabilities = np.clip(0.5 + 0.25 * (2 * outcomes - 1) + rng.normal(0, 0.8, 400), 0, 1)
        metrics = prompt_metrics(probabilities, outcomes)
        chance = chance_aurc(metrics["n"], float(outcomes.mean()))
        return metrics["aurc"], metrics["aurc"] - chance

    (easy_aurc, easy_excess), (hard_aurc, hard_excess) = scored(0.85), scored(0.55)

    assert abs(easy_aurc - hard_aurc) > 0.2
    assert abs(easy_excess - hard_excess) < 0.05


def test_prompt_geometry_averages_over_the_siblings():
    grouped = group_views_by_prompt(
        prepare_trace_views(
            [_trace(0, 5, 20), _trace(1, 5, 20)], 7, max_tokens_per_trace=None
        )
    )
    scores = {
        0: {
            GEOMETRY_FEATURE: 1.0,
            PROBE_FEATURE: -2.0,
            TOKEN_PROBE_FEATURE: 4.0,
            QUADRATIC_FEATURE: -1.0,
        },
        1: {
            GEOMETRY_FEATURE: 3.0,
            PROBE_FEATURE: 2.0,
            TOKEN_PROBE_FEATURE: 6.0,
            QUADRATIC_FEATURE: -3.0,
        },
    }

    aggregated = prompt_geometry(grouped, [5], scores)

    assert aggregated[5][GEOMETRY_FEATURE] == pytest.approx(2.0)
    assert aggregated[5][PROBE_FEATURE] == pytest.approx(0.0)
    assert aggregated[5][TOKEN_PROBE_FEATURE] == pytest.approx(5.0)
    assert aggregated[5][QUADRATIC_FEATURE] == pytest.approx(-2.0)


def test_feature_matrix_takes_geometry_from_the_budget_not_the_frozen_column():
    """The base table is read straight from the frozen OOF file, which carries a
    full-label-budget `rmd_tail_q20`. Letting it survive would score every budget
    with the same feature and flatten the curve to nothing."""
    base = {1: {"length": -1.0, GEOMETRY_FEATURE: 99.0}}
    geometry = {1: {GEOMETRY_FEATURE: 0.5}}

    matrix = feature_matrix(base, geometry, [1], ["length", GEOMETRY_FEATURE])

    assert matrix.tolist() == [[-1.0, 0.5]]


def test_feature_matrix_marks_an_absent_feature_missing_rather_than_zero():
    matrix = feature_matrix({1: {"length": -1.0}}, {}, [1], ["length", PROBE_FEATURE])

    assert matrix[0, 0] == -1.0
    assert np.isnan(matrix[0, 1])


def test_a_single_class_budget_predicts_the_base_rate_instead_of_failing():
    train = np.arange(8, dtype=float)[:, None]
    outcomes = np.ones(8)

    predictions = fit_predict_logistic(train, outcomes, np.zeros((3, 1)))

    assert predictions.tolist() == [1.0, 1.0, 1.0]


def test_the_readout_ranks_a_separable_holdout():
    train = np.linspace(-3, 3, 40)[:, None]
    outcomes = (train[:, 0] > 0).astype(float)

    predictions = fit_predict_logistic(train, outcomes, np.asarray([[-2.0], [2.0]]))

    assert predictions[1] > predictions[0]


def test_summaries_ignore_missing_replicates():
    summary = summarize_replicates([1.0, None, 3.0, float("nan")])

    assert summary["n_replicates"] == 2
    assert summary["median"] == pytest.approx(2.0)


def test_the_sign_test_counts_lower_is_better_as_a_win():
    losses = sign_summary([-0.1, -0.2, -0.3, 0.05], negative_is_better=True)

    assert losses["win_rate"] == pytest.approx(0.75)
    # 4 draws, 1 on the wrong side: two-sided exact binomial.
    assert losses["p_sign"] == pytest.approx(2 * (1 + 4) / 16)


def _pooled_model(label: str, deltas: list[float], aurocs: list[tuple[float, float]]) -> dict:
    return {
        "label": label,
        "replicate_rows": [
            {
                "budget": 25,
                "delta_aurc_B0_rmd_minus_B0": -0.01,
                "delta_aurc_B0_probe_minus_B0": 0.01,
                "delta_aurc_B0_rmd_minus_B0_probe": delta,
                "delta_aurc_B0_both_minus_B0_rmd": 0.0,
                f"auroc_{GEOMETRY_FEATURE}": rmd,
                f"auroc_{PROBE_FEATURE}": probe,
            }
            for delta, (rmd, probe) in zip(deltas, aurocs)
        ],
    }


def test_pooling_counts_every_draw_and_the_models_that_agree():
    models = [
        _pooled_model("a", [-0.02, -0.03], [(0.7, 0.6), (0.7, 0.6)]),
        _pooled_model("b", [-0.01, -0.04], [(0.6, 0.7), (0.6, 0.7)]),
        _pooled_model("c", [+0.01, +0.02], [(0.6, 0.7), (0.6, 0.7)]),
    ]

    table = pooled_sign_table(models, [25])
    versus_probe = table[0]["quantities"]["delta_aurc_B0_rmd_minus_B0_probe"]

    assert versus_probe["pooled"]["n_replicates"] == 6
    assert versus_probe["pooled"]["win_rate"] == pytest.approx(4 / 6)
    # Two of three datasets put their own median on the geometry side; that count
    # is the claim, because draws inside one model share an evaluation set.
    assert versus_probe["models_agreeing"] == 2
    assert versus_probe["n_models"] == 3
    assert versus_probe["per_model"]["c"]["win_rate"] == 0.0


def test_the_pooled_auroc_delta_is_derived_from_the_two_feature_columns():
    """The base-rate-free view of the same comparison. It is not stored per
    replicate, so pooling has to reconstruct it rather than read it."""
    models = [_pooled_model("a", [-0.02, -0.02], [(0.80, 0.70), (0.60, 0.75)])]

    entry = pooled_sign_table(models, [25])[0]["quantities"]["auroc_rmd_minus_probe"]

    # Positive is the geometry side here, the opposite convention to the AURC
    # deltas, and the sign test has to follow the metric rather than the column.
    assert entry["median"] == pytest.approx(-0.025)
    assert entry["pooled"]["win_rate"] == pytest.approx(0.5)


def test_a_budget_absent_from_a_model_does_not_silently_shrink_the_pool():
    models = [_pooled_model("a", [-0.02, -0.03], [(0.7, 0.6), (0.7, 0.6)])]

    table = pooled_sign_table(models, [25, 50])

    assert table[0]["quantities"]["delta_aurc_B0_rmd_minus_B0_probe"]["pooled"]["n_replicates"] == 2
    assert table[1]["quantities"]["delta_aurc_B0_rmd_minus_B0_probe"]["pooled"]["n_replicates"] == 0
    assert table[1]["quantities"]["delta_aurc_B0_rmd_minus_B0_probe"]["models_agreeing"] == 0


def _curve(budget: int, delta: float | None) -> dict:
    return {
        "budget": budget,
        "delta_aurc": {"B0_rmd_minus_B0_probe": {"median": delta}},
    }


def test_the_crossing_is_interpolated_in_log_budget():
    crossing = crossing_budget([_curve(100, -0.02), _curve(400, +0.02)])

    assert crossing["crossed"] is True
    assert crossing["bracket"] == [100, 400]
    assert crossing["budget"] == pytest.approx(200.0)


def test_geometry_ahead_everywhere_is_reported_as_no_crossing():
    crossing = crossing_budget([_curve(25, -0.05), _curve(400, -0.01)])

    assert crossing["crossed"] is False
    assert "geometry is ahead" in crossing["note"]


def test_the_probe_ahead_everywhere_is_reported_as_no_crossing():
    crossing = crossing_budget([_curve(25, 0.01), _curve(400, 0.05)])

    assert crossing["crossed"] is False
    assert "probe is ahead" in crossing["note"]
