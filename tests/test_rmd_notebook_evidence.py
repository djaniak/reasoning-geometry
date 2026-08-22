import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

LEDGER_INPUTS = (
    "results/orgad_agreement_control/orgad_agreement_control_results.json",
    "results/peer_difficulty_control/peer_difficulty_control_results.json",
    "results/allocation_precheck/allocation_precheck_results.json",
    "results/application_alignment/math500_application_alignment_results.json",
    "results/label_efficiency/label_efficiency_results.json",
    "results/label_efficiency/label_efficiency_replicates.csv",
    "results/label_efficiency_supervision_ladder/label_efficiency_results.json",
    "results/label_efficiency_supervision_ladder/label_efficiency_replicates.csv",
    "results/label_efficiency_token_pooling/label_efficiency_results.json",
    "results/label_efficiency_token_pooling/label_efficiency_replicates.csv",
)


def test_paper_notebooks_do_not_require_retired_wave1_artifacts():
    for name in (
        "notebooks/build_14_rmd_workshop_story.py",
        "notebooks/build_17_rmd_experiment_ledger.py",
    ):
        source = (ROOT / name).read_text()
        assert "math500_wave1_results.json" not in source
        assert "Table 5b" not in source


def test_paper_notebooks_name_the_qwen_checkpoint_correctly():
    for name in (
        "notebooks/build_14_rmd_workshop_story.py",
        "notebooks/build_17_rmd_experiment_ledger.py",
    ):
        source = (ROOT / name).read_text()
        assert "Qwen-1.5B" not in source
        assert "Qwen2.5-7B" in source


def test_storyboard_only_closes_the_refit_gate_for_a_complete_sweep():
    source = (ROOT / "notebooks/build_14_rmd_workshop_story.py").read_text()
    assert 'REFIT_PAYLOAD.get("complete")' in source


def test_ledger_inputs_are_present_and_admitted_by_gitignore():
    for relative in LEDGER_INPUTS:
        path = ROOT / relative
        assert path.is_file(), relative
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", relative],
            cwd=ROOT,
            check=False,
        )
        assert ignored.returncode == 1, f"{relative} is excluded from the evidence package"
