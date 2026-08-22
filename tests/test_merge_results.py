from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.merge_results import merge_dicts, validate_compatible


def test_merge_dicts_merges_layer_keys_recursively():
    base = {
        "dataset": "gsm8k",
        "layers": {
            "7": {
                "combined": {"roc_auc_mean": 0.8},
            }
        },
    }
    incoming = {
        "layers": {
            "7": {
                "normalized_combined": {"roc_auc_mean": 0.82},
            },
            "14": {
                "combined": {"roc_auc_mean": 0.78},
            },
        },
        "settings": {"analysis_family": "controls"},
    }

    merged = merge_dicts(base, incoming)

    assert merged["layers"]["7"]["combined"]["roc_auc_mean"] == 0.8
    assert merged["layers"]["7"]["normalized_combined"]["roc_auc_mean"] == 0.82
    assert merged["layers"]["14"]["combined"]["roc_auc_mean"] == 0.78
    assert merged["settings"]["analysis_family"] == "controls"


def test_validate_compatible_rejects_mismatched_counts(tmp_path: Path):
    anchor = {"dataset": "gsm8k", "n_correct": 10, "n_incorrect": 5}
    candidate = {"dataset": "gsm8k", "n_correct": 11, "n_incorrect": 5}

    try:
        validate_compatible(anchor, candidate, str(tmp_path / "candidate.json"))
    except ValueError as exc:
        assert "n_correct" in str(exc)
    else:
        raise AssertionError("Expected validate_compatible() to reject mismatched counts")
