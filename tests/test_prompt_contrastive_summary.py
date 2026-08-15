import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.summarize import render_prompt_decomposition_section


def test_summary_separates_parseable_contrastive_results():
    data = {
        "layers": {
            "7": {
                "methods": {},
                "parseable_only": {
                    "methods": {
                        "contrast_full": {
                            "prompt_centered_auc": 0.61,
                            "within_prompt_macro": 0.59,
                            "n_mixed_prompts": 3,
                        }
                    }
                },
            }
        },
        "contrastive": {
            "regions": ["full"],
            "alignment_diagnostics": [
                {
                    "layer": 7,
                    "region": "full",
                    "observed_alignment": 0.8,
                    "null": {"mean": 0.3, "p_value": 0.05},
                }
            ],
        },
    }
    lines = []
    render_prompt_decomposition_section({"qwen": {"math500": data}}, lines)
    text = "\n".join(lines)
    assert "Prompt-contrastive correctness (parseable-only primary)" in text
    assert "contrast_full" in text
    assert "Observed alignment" in text
