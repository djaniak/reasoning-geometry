import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controls.refit_stability import (
    MODEL_SPECS,
    PEER_DEPLOYABLE_CONTRAST,
    PEER_RESIDUAL_CONTRAST,
    Step,
    collect_seed,
    main,
    plan_seed,
    run_step,
    summarize_quantity,
)

QWEN = next(spec for spec in MODEL_SPECS if spec.label == "qwen")


def _plan(tmp_path, seed=101, specs=MODEL_SPECS):
    return plan_seed(seed, specs, tmp_path, n_bootstrap=10, load_workers=2)


def test_every_stage_carries_the_refit_seed(tmp_path):
    """The whole point of the test is that one partition drives every fitting
    step. A stage left at 42 while the others move is not a refit -- it is a
    refit of some stages evaluated against a frozen one, which is the very
    confound the review is asking about."""
    for step in _plan(tmp_path, seed=777):
        assert "--seed" in step.cmd, step.name
        assert step.cmd[step.cmd.index("--seed") + 1] == "777", step.name


def test_the_decomposition_runs_before_what_reads_its_oof(tmp_path):
    """Order is not cosmetic: abstention and the probe both read the OOF csv
    the decomposition writes."""
    names = [step.name for step in _plan(tmp_path)]
    for spec in MODEL_SPECS:
        assert names.index(f"decomposition/{spec.label}") < names.index(
            f"abstention/{spec.label}"
        )
        assert names.index(f"decomposition/{spec.label}") < names.index(
            f"probe/{spec.label}"
        )


def test_the_peer_rung_runs_last_because_it_needs_every_model(tmp_path):
    """A peer rung compares a target against peers fitted on the same
    partition, so it cannot start until all three models are through."""
    steps = _plan(tmp_path)
    assert steps[-1].name == "peer"
    for spec in MODEL_SPECS:
        assert any(spec.label in part for part in steps[-1].cmd)


def test_the_peer_rung_is_dropped_when_a_single_model_is_selected(tmp_path):
    steps = plan_seed(101, [QWEN], tmp_path, n_bootstrap=10, load_workers=2)
    assert [step.name for step in steps] == [
        "decomposition/qwen",
        "abstention/qwen",
        "probe/qwen",
    ]


def test_the_probe_joins_this_seeds_oof_not_the_frozen_one(tmp_path):
    """If the probe read the committed OOF, rmd_tail_q20 and
    probe_hidden_tail_q20 would sit frozen in a table whose other rows moved."""
    step = next(s for s in _plan(tmp_path, seed=303) if s.name == "probe/qwen")
    oof = step.cmd[step.cmd.index("--oof") + 1]
    assert "seed_303" in oof
    assert "results/qwen_bestofn_full" not in oof


def test_seeds_get_separate_working_directories(tmp_path):
    a = next(s for s in _plan(tmp_path, seed=1) if s.name == "decomposition/qwen")
    b = next(s for s in _plan(tmp_path, seed=2) if s.name == "decomposition/qwen")
    assert a.produces != b.produces
    assert a.marker != b.marker


def _step(tmp_path, cmd, produces="out.json"):
    return Step(
        name="probe/fake",
        cmd=cmd,
        marker=tmp_path / ".done",
        produces=tmp_path / produces,
        log=tmp_path / "log.txt",
    )


def test_a_finished_step_is_skipped_on_re_entry(tmp_path):
    """The sweep is meant to be resumable after a killed terminal."""
    cmd = [sys.executable, "-c", "raise SystemExit(1)"]
    (tmp_path / ".done").write_text(json.dumps({"cmd": cmd}))
    (tmp_path / "out.json").write_text("{}")
    step = _step(tmp_path, cmd)
    assert step.done
    assert run_step(step) is False


def test_a_marker_from_a_different_command_is_not_reused(tmp_path):
    (tmp_path / ".done").write_text(json.dumps({"cmd": ["old", "command"]}))
    (tmp_path / "out.json").write_text("{}")
    step = _step(tmp_path, ["new", "command"])
    assert step.done is False


def test_a_step_whose_output_was_deleted_runs_again(tmp_path):
    """A marker alone is not evidence: the artifact is what downstream reads."""
    (tmp_path / ".done").write_text("{}")
    step = _step(
        tmp_path,
        [sys.executable, "-c", "open('out.json','w').write('{}')"],
    )
    assert step.done is False


def test_a_failing_step_raises_and_leaves_no_marker(tmp_path):
    """A marker written after a failure would make the resume logic skip a
    stage that never produced anything."""
    step = _step(tmp_path, [sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(RuntimeError, match="exited 3"):
        run_step(step)
    assert not (tmp_path / ".done").exists()


def test_a_step_that_exits_clean_without_its_artifact_still_fails(tmp_path):
    step = _step(tmp_path, [sys.executable, "-c", "pass"])
    with pytest.raises(RuntimeError, match="did not write"):
        run_step(step)
    assert not (tmp_path / ".done").exists()


def test_a_successful_step_records_the_command_it_ran(tmp_path):
    """Resuming into a marker produced by different arguments would silently
    mix two protocols in one table."""
    cmd = [
        sys.executable,
        "-c",
        f"open({str(tmp_path / 'out.json')!r},'w').write('{{}}')",
    ]
    step = _step(tmp_path, cmd)
    assert run_step(step) is True
    assert json.loads((tmp_path / ".done").read_text())["cmd"] == cmd


def test_dry_run_executes_nothing(tmp_path):
    step = _step(tmp_path, [sys.executable, "-c", "open('out.json','w')"])
    assert run_step(step, dry_run=True) is False
    assert not (tmp_path / "out.json").exists()


def test_a_sign_flip_is_reported():
    """The review's decision rule: a residual that changes sign across refits
    is demoted, however tight its within-refit interval was."""
    stable = summarize_quantity([-0.052, -0.048, -0.061])
    assert stable["sign_stable"] is True

    flipped = summarize_quantity([-0.052, 0.007, -0.061])
    assert flipped["sign_stable"] is False
    assert flipped["n_positive"] == 1
    assert flipped["n_negative"] == 2


def test_spread_and_drift_from_the_frozen_partition():
    summary = summarize_quantity([-0.05, -0.02, -0.09], frozen=-0.05)
    assert summary["spread"] == pytest.approx(0.07)
    assert summary["max_abs_drift_from_frozen"] == pytest.approx(0.04)
    assert summary["n"] == 3


def test_an_empty_quantity_summarises_to_nothing_rather_than_raising():
    assert summarize_quantity([]) == {"n": 0}


def _write_artifacts(root, seed, label, *, probe=True, abstention=True, peer=True):
    model_dir = root / f"seed_{seed}" / label
    if abstention:
        path = model_dir / "abstention"
        path.mkdir(parents=True, exist_ok=True)
        (path / "math500_incremental_abstention_results.json").write_text(
            json.dumps(
                {
                    "populations": {
                        "full_population": {
                            "n_prompts": 500,
                            "paired_deltas": {
                                "B1_minus_B0_aurc": {"point_estimate": -0.052},
                                "B1_minus_B0_auacc": {"point_estimate": 0.052},
                            },
                        }
                    }
                }
            )
        )
    if probe:
        path = model_dir / "probe"
        path.mkdir(parents=True, exist_ok=True)
        (path / "last_token_probe_results.json").write_text(
            json.dumps(
                {
                    "models": {
                        label: {
                            "populations": {
                                "parseable": {
                                    "selected_layers": [7, 7, 21, 7, 7],
                                    "scores": {
                                        "last_token_probe": {
                                            "point": {
                                                "pooled_auroc": 0.90,
                                                "macro_prompt_auroc": 0.64,
                                                "pooled_minus_macro": 0.26,
                                                "n_mixed_prompts": 117,
                                            }
                                        },
                                        "rmd_tail_q20": {
                                            "point": {
                                                "pooled_auroc": 0.81,
                                                "pooled_minus_macro": 0.21,
                                                "n_mixed_prompts": 117,
                                            }
                                        },
                                    },
                                }
                            }
                        }
                    }
                }
            )
        )
    if peer:
        path = root / f"seed_{seed}" / "peer"
        path.mkdir(parents=True, exist_ok=True)
        (path / "peer_cost_ladder_results.json").write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "label": label,
                            "populations": {
                                "full_population": {
                                    "contrasts": {
                                        PEER_RESIDUAL_CONTRAST: {
                                            "aurc": {"point_estimate": 0.056}
                                        },
                                        PEER_DEPLOYABLE_CONTRAST: {
                                            "aurc": {"point_estimate": 0.031}
                                        },
                                        "B1_minus_B0": {"aurc": {"point_estimate": -0.052}},
                                    }
                                }
                            },
                        }
                    ]
                }
            )
        )


def test_collect_reads_the_three_tracked_quantities(tmp_path):
    _write_artifacts(tmp_path, 42, "qwen")
    record = collect_seed(42, [QWEN], tmp_path)

    entry = record["models"]["qwen"]
    assert entry["b1_minus_b0_aurc"] == pytest.approx(-0.052)
    assert entry["probe_pooled_minus_macro"] == pytest.approx(0.26)
    assert entry["selected_layers"] == [7, 21]
    assert record["peer"]["qwen"]["residual_aurc"] == pytest.approx(0.056)
    assert record["peer"]["qwen"]["residual_deployable_aurc"] == pytest.approx(0.031)


def test_collect_tolerates_a_sweep_stopped_part_way(tmp_path):
    """A run killed after six hours should still summarise the refits that
    finished rather than losing them to an exception."""
    _write_artifacts(tmp_path, 42, "qwen", probe=False, peer=False)
    record = collect_seed(42, [QWEN], tmp_path)

    assert record["models"]["qwen"]["b1_minus_b0_aurc"] == pytest.approx(-0.052)
    assert "probe_pooled_minus_macro" not in record["models"]["qwen"]
    assert record["peer"] == {}


def test_collect_returns_an_empty_record_for_a_seed_never_run(tmp_path):
    record = collect_seed(999, [QWEN], tmp_path)
    assert record["models"]["qwen"] == {}
    assert record["peer"] == {}


def test_collect_only_keeps_an_incomplete_sweep_out_of_the_canonical_result(
    tmp_path, monkeypatch
):
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refit_stability",
            "--collect_only",
            "--seed",
            "42",
            "--model",
            "qwen",
            "--skip_peer",
            "--work_dir",
            str(tmp_path / "empty-work"),
            "--output_dir",
            str(output),
        ],
    )

    main()

    assert not (output / "refit_stability_results.json").exists()
    partial = json.loads((output / "refit_stability_partial_results.json").read_text())
    assert partial["complete"] is False
    assert partial["complete_seeds"] == []
    assert partial["incomplete_seeds"] == [42]
