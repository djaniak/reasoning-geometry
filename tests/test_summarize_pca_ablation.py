import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.summarize import (
    generate_combined_json,
    generate_markdown,
    load_application_alignment_result,
    load_bestofn_results,
    load_one_class_sweep_results,
    load_pca_ablation_results,
    load_prompt_decomposition_results,
    load_prompt_selection_results,
)


def test_load_pca_ablation_results_reads_expected_file(tmp_path: Path):
    root = tmp_path / "results"
    ds_dir = root / "qwen" / "gsm8k"
    ds_dir.mkdir(parents=True)

    payload = {
        "dataset": "gsm8k",
        "model": "qwen",
        "settings": {"pca_dims": [32, 64]},
        "layers": {},
    }
    (ds_dir / "gsm8k_pca_ablation_results.json").write_text(json.dumps(payload))
    loaded = load_pca_ablation_results(str(root))

    assert loaded["qwen"]["gsm8k"]["settings"]["pca_dims"] == [32, 64]


def test_generate_markdown_renders_pca_ablation_section(tmp_path: Path):
    output_path = tmp_path / "SUMMARY.md"
    pca_results = {
        "qwen": {
            "gsm8k": {
                "settings": {"pca_dims": [32, "max"]},
                "layers": {
                    "7": {
                        "dims": {
                            "32": {
                                "mahalanobis_only": {"roc_auc_mean": 0.74, "roc_auc_std": 0.01},
                                "combined": {"roc_auc_mean": 0.78, "roc_auc_std": 0.02},
                                "delta_vs_entropy": 0.03,
                                "length_controlled_delta": 0.02,
                            },
                            "max": {
                                "mahalanobis_only": {"roc_auc_mean": 0.77, "roc_auc_std": 0.02},
                                "combined": {"roc_auc_mean": 0.81, "roc_auc_std": 0.03},
                                "delta_vs_entropy": 0.06,
                                "length_controlled_delta": 0.05,
                            },
                        },
                    }
                },
            }
        }
    }

    generate_markdown({}, str(output_path), pca_ablation_results=pca_results)
    content = output_path.read_text()

    assert "## PCA-dimension ablation (base geometry)" in content
    assert (
        "| Model | Dataset | Layer | PCA dim | Mahal-only | Combined | Δ (raw) | Δ (len-ctrl) |"
        in content
    )
    assert "| qwen | gsm8k | L7 | 32 | 0.740 | 0.780 | +0.0300 | +0.0200 |" in content
    assert "| qwen | gsm8k | L7 | max | 0.770 | 0.810 | +0.0600 | +0.0500 |" in content


def _bestofn_payload():
    selector = lambda value: {
        "pass_at_1_mean": value,
        "pass_at_1_std": 0.01,
        "n_problems": 100,
    }
    return {
        "dataset": "math500",
        "n_values": {
            "8": {
                "n": 8,
                "n_problems": 100,
                "selectors": {
                    "random": selector(0.44),
                    "oracle_pass_at_n": selector(0.57),
                    "majority_vote": selector(0.57),
                    "mean_logprob": selector(0.50),
                    "entropy_only": selector(0.50),
                },
                "layers": {
                    "7": {
                        "mahalanobis_only": selector(0.53),
                        "combined": selector(0.55),
                        "rmd_only": selector(0.54),
                        "rmd_combined": selector(0.54),
                    },
                    "21": {
                        "mahalanobis_only": selector(0.50),
                        "combined": selector(0.51),
                        "rmd_only": selector(0.57),
                        "rmd_combined": selector(0.56),
                    },
                },
            }
        },
    }


def test_load_and_render_bestofn_supplemental_results(tmp_path: Path):
    root = tmp_path / "results"
    ds_dir = root / "deepseek_bestofn_pilot" / "math500"
    ds_dir.mkdir(parents=True)
    bestofn = _bestofn_payload()
    concordance = {
        "metric": "within_prompt_pairwise_concordance",
        "layers": {
            "7": {
                "mean_concordance": 0.26,
                "std_concordance": 0.30,
                "n_problems_evaluated": 31,
            }
        },
    }
    (ds_dir / "math500_best_of_n_results.json").write_text(
        json.dumps(bestofn)
    )
    (ds_dir / "math500_bestofn_concordance.json").write_text(
        json.dumps(concordance)
    )

    loaded_bestofn, loaded_concordance = load_bestofn_results(str(root))
    output_path = tmp_path / "SUMMARY.md"
    generate_markdown(
        {},
        str(output_path),
        bestofn_results=loaded_bestofn,
        concordance_results=loaded_concordance,
    )
    content = output_path.read_text()

    assert "| deepseek | pilot | math500 | 8 | 0.440 | 0.570 |" in content
    assert "0.570 (L21)" in content
    assert "| deepseek | pilot | math500 | L7 | 0.260 | 0.300 | 31 |" in content


def test_combined_json_keeps_supplemental_results(tmp_path: Path):
    output_path = tmp_path / "all_results.json"
    supplemental = {"best_of_n": {"qwen": {"pilot": _bestofn_payload()}}}

    generate_combined_json(
        {"qwen": {"math500": {"dataset": "math500"}}},
        str(output_path),
        supplemental=supplemental,
    )

    payload = json.loads(output_path.read_text())
    assert payload["qwen"]["math500"]["dataset"] == "math500"
    assert payload["_supplemental"]["best_of_n"]["qwen"]["pilot"]


def test_load_and_render_new_confidence_experiment_artifacts(tmp_path: Path):
    root = tmp_path / "results"
    bestofn_dir = root / "qwen_bestofn_full" / "math500"
    sweep_dir = root / "qwen_one_class" / "math500"
    alignment_dir = root / "application_alignment"
    bestofn_dir.mkdir(parents=True)
    sweep_dir.mkdir(parents=True)
    alignment_dir.mkdir(parents=True)

    decomposition = {
        "model": "qwen",
        "dataset": "math500",
        "layers": {
            "7": {
                "methods": {
                    "raw": {
                        "metrics": {
                            "pooled_auc": 0.41,
                            "prompt_centered_auc": 0.48,
                            "within_prompt_pair_weighted": 0.49,
                            "score_icc": 0.88,
                            "prompt_score_pass_rate_spearman": -0.20,
                        }
                    },
                    "rmd": {
                        "metrics": {
                            "pooled_auc": 0.74,
                            "prompt_centered_auc": 0.56,
                            "within_prompt_pair_weighted": 0.55,
                            "score_icc": 0.94,
                            "prompt_score_pass_rate_spearman": 0.45,
                        }
                    },
                }
            }
        },
    }
    selection = {
        "model": "qwen",
        "dataset": "math500",
        "settings": {
            "invalid_answer_policy": "count as failure",
        },
        "layers": {
            "7": {
                "answer_parsing": {
                    "n_traces": 4000,
                    "n_parsed": 3600,
                    "parse_rate": 0.9,
                    "correct_parse_rate": 1.0,
                    "incorrect_parse_rate": 0.75,
                    "n_prompts_without_parsed_answer": 3,
                },
                "selectors": {
                    "random": {"pass_at_1": 0.56},
                    "top1_rmd": {"pass_at_1": 0.61},
                }
            }
        },
    }
    sweep = {
        "model": "qwen",
        "dataset": "math500",
        "layers": {
            "7": {
                "dimensions": {
                    "1": {
                        "methods": {
                            "centroid": {
                                "pooled_roc_auc": 0.70,
                                "fold_roc_auc_mean": 0.69,
                                "fold_roc_auc_std": 0.02,
                                "pooled_pr_auc": 0.71,
                                "n_eval": 500,
                            }
                        }
                    }
                }
            }
        },
    }
    alignment = {
        "warning": "Exploratory descriptive correlations only.",
        "conditions": [
            {
                "model": "qwen",
                "layer": 7,
                "method": "rmd",
                "within_prompt_pair_weighted": 0.55,
                "score_icc": 0.94,
                "top1_gain_over_random": 0.05,
                "selective_ausc_gain_over_entropy": 0.10,
            }
        ],
        "correlations": {},
    }
    (bestofn_dir / "math500_prompt_decomposition_results.json").write_text(
        json.dumps(decomposition)
    )
    (bestofn_dir / "math500_prompt_selection_results.json").write_text(
        json.dumps(selection)
    )
    (sweep_dir / "math500_one_class_sweep_results.json").write_text(
        json.dumps(sweep)
    )
    (alignment_dir / "math500_application_alignment_results.json").write_text(
        json.dumps(alignment)
    )

    decompositions = load_prompt_decomposition_results(str(root))
    selections = load_prompt_selection_results(str(root))
    sweeps = load_one_class_sweep_results(str(root))
    loaded_alignment = load_application_alignment_result(str(root), "math500")
    output = tmp_path / "SUMMARY.md"
    generate_markdown(
        {},
        str(output),
        prompt_decomposition_results=decompositions,
        prompt_selection_results=selections,
        application_alignment_result=loaded_alignment,
        one_class_sweep_results=sweeps,
    )
    content = output.read_text()

    assert "## Prompt Decomposition" in content
    assert "| qwen | math500 | L7 | rmd | 0.740 | 0.560 | 0.550 | 0.940 | 0.450 |" in content
    assert "## OOF Prompt Selection" in content
    assert "Unparsed answers are counted as failures." in content
    assert "| qwen | math500 | L7 | 3600/4000 | 0.900 | 1.000 | 0.750 | 3 |" in content
    assert "| qwen | math500 | L7 | top1_rmd | 0.610 |" in content
    assert "## Application Alignment" in content
    assert "## One-Class Mechanism Sweep" in content


def test_prompt_decomposition_summary_marks_legacy_raw_rmd_schema(tmp_path: Path):
    output = tmp_path / "SUMMARY.md"
    legacy = {
        "deepseek": {
            "math500": {
                "model": "deepseek",
                "dataset": "math500",
                "layers": {
                    "7": {
                        "methods": {
                            "raw": {"metrics": {}},
                            "rmd": {"metrics": {}},
                        }
                    }
                },
            }
        }
    }

    generate_markdown(
        {},
        str(output),
        prompt_decomposition_results=legacy,
    )

    assert "raw/RMD-only legacy schema" in output.read_text()
