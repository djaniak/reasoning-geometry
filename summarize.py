"""
summarize.py - Aggregate result JSONs into summary markdown and combined JSON.

The results directory contains multiple schemas (probe analysis, best-of-N,
prefix analysis, prefix filtering). This summarizer renders sections only for
schemas it recognizes and skips incompatible ones.
"""
import json
import math
import os
import argparse
from pathlib import Path


def load_results(results_dir: str) -> dict:
    """Load all result JSONs into a nested dict: {model: {dataset: data}}."""
    results = {}
    base = Path(results_dir)
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        results[model] = {}
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds = ds_dir.name
            json_files = sorted([f for f in ds_dir.iterdir() if f.suffix == ".json"])
            if not json_files:
                continue
            # Prefer the canonical merged output ({dataset}_results.json) over
            # family-specific files ({dataset}_base/controls/subspace_results.json).
            canonical = ds_dir / f"{ds}_results.json"
            if canonical.exists():
                chosen = canonical
            else:
                chosen = next((f for f in json_files if f.name.endswith("_results.json")), json_files[0])
            with open(chosen) as fh:
                results[model][ds] = json.load(fh)
    return results


def load_pca_ablation_results(results_dir: str) -> dict:
    """Load PCA ablation outputs into {model: {dataset: data}}."""
    ablations = {}
    base = Path(results_dir)
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds = ds_dir.name
            path = ds_dir / f"{ds}_pca_ablation_results.json"
            if not path.exists():
                continue
            with open(path) as fh:
                ablations.setdefault(model, {})[ds] = json.load(fh)
    return ablations


def is_probe_result(data: dict) -> bool:
    """Main analyze.py schema with layer AUCs and class counts."""
    return isinstance(data, dict) and all(
        key in data for key in ("n_correct", "n_incorrect", "entropy_baseline", "layers")
    )


def is_prefix_filter_result(data: dict) -> bool:
    """prefix_filter.py schema with policy settings and threshold sweeps."""
    return isinstance(data, dict) and all(
        key in data for key in ("settings", "threshold_quantiles", "max_restarts")
    )


def is_selective_prediction_result(data: dict) -> bool:
    """selective_prediction.py schema with coverage-accuracy scorers."""
    return (
        isinstance(data, dict)
        and "scorers" in data
        and "no_abstain" in data.get("scorers", {})
    )


def load_selective_results(results_dir: str) -> dict:
    """Load selective prediction results from results/{model}_selective/{dataset}/ dirs.

    Returns {model: {dataset: data}} for entries matching the selective prediction schema.
    Separate from load_results to avoid colliding with the canonical {dataset}_results.json
    loader.
    """
    selective: dict = {}
    base = Path(results_dir)
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.endswith("_selective"):
            continue
        model = model_dir.name[: -len("_selective")]
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds = ds_dir.name
            target = ds_dir / f"{ds}_selective_prediction_results.json"
            if not target.exists():
                continue
            with open(target) as fh:
                data = json.load(fh)
            if is_selective_prediction_result(data):
                selective.setdefault(model, {})[ds] = data
    return selective


def load_bestofn_results(results_dir: str) -> tuple[dict, dict]:
    """Load Best-of-N evaluations and concordance diagnostics by model/scale/dataset."""
    bestofn: dict = {}
    concordance: dict = {}
    base = Path(results_dir)
    marker = "_bestofn_"

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or marker not in model_dir.name:
            continue
        model, scale = model_dir.name.rsplit(marker, 1)
        if scale not in {"pilot", "full"}:
            continue
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds = ds_dir.name
            result_path = ds_dir / f"{ds}_best_of_n_results.json"
            if result_path.exists():
                with open(result_path) as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and "n_values" in data:
                    bestofn.setdefault(model, {}).setdefault(scale, {})[ds] = data

            concordance_path = ds_dir / f"{ds}_bestofn_concordance.json"
            if concordance_path.exists():
                with open(concordance_path) as fh:
                    data = json.load(fh)
                if (
                    isinstance(data, dict)
                    and data.get("metric") == "within_prompt_pairwise_concordance"
                ):
                    concordance.setdefault(model, {}).setdefault(scale, {})[ds] = data

    return bestofn, concordance


def load_prompt_decomposition_results(results_dir: str) -> dict:
    """Load full Best-of-N prompt decomposition artifacts by model/dataset."""
    results = {}
    base = Path(results_dir)
    marker = "_bestofn_full"
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.endswith(marker):
            continue
        model = model_dir.name[: -len(marker)]
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds = ds_dir.name
            path = ds_dir / f"{ds}_prompt_decomposition_results.json"
            if path.exists():
                with path.open() as handle:
                    results.setdefault(model, {})[ds] = json.load(handle)
    return results


def load_prompt_selection_results(results_dir: str) -> dict:
    """Load OOF prompt-selection artifacts by model/dataset."""
    results = {}
    base = Path(results_dir)
    marker = "_bestofn_full"
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.endswith(marker):
            continue
        model = model_dir.name[: -len(marker)]
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds = ds_dir.name
            path = ds_dir / f"{ds}_prompt_selection_results.json"
            if path.exists():
                with path.open() as handle:
                    results.setdefault(model, {})[ds] = json.load(handle)
    return results


def load_application_alignment_result(
    results_dir: str, dataset: str = "math500"
) -> dict:
    path = (
        Path(results_dir)
        / "application_alignment"
        / f"{dataset}_application_alignment_results.json"
    )
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def load_one_class_sweep_results(results_dir: str) -> dict:
    """Load canonical one-class mechanism sweeps by model/dataset."""
    results = {}
    base = Path(results_dir)
    marker = "_one_class"
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.endswith(marker):
            continue
        model = model_dir.name[: -len(marker)]
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds = ds_dir.name
            path = ds_dir / f"{ds}_one_class_sweep_results.json"
            if path.exists():
                with path.open() as handle:
                    results.setdefault(model, {})[ds] = json.load(handle)
    return results


def collect_typed_results(results: dict):
    """Split raw loaded results by schema."""
    probe = {}
    prefix_filter = {}
    for model in sorted(results):
        for ds in sorted(results[model]):
            data = results[model][ds]
            if is_probe_result(data):
                probe.setdefault(model, {})[ds] = data
            elif is_prefix_filter_result(data):
                prefix_filter.setdefault(model, {})[ds] = data
    return probe, prefix_filter


def fmt(v, digits=3):
    """Format a float to fixed decimal places."""
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def pm(mean, std, digits=3):
    """Format mean ± std."""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def best_layer(layers_dict):
    """Return the layer key with highest combined AUC."""
    return max(layers_dict, key=lambda l: layers_dict[l]["combined"]["roc_auc_mean"])


def generate_markdown(
    results: dict,
    output_path: str,
    pca_ablation_results: dict | None = None,
    selective_results: dict | None = None,
    bestofn_results: dict | None = None,
    concordance_results: dict | None = None,
    prompt_decomposition_results: dict | None = None,
    prompt_selection_results: dict | None = None,
    application_alignment_result: dict | None = None,
    one_class_sweep_results: dict | None = None,
):
    """Generate a markdown summary of all results."""
    probe_results, prefix_filter_results = collect_typed_results(results)
    pca_ablation_results = pca_ablation_results or {}

    lines = []
    lines.append("# Results Summary")
    lines.append("")
    lines.append("*Auto-generated by `summarize.py` from per-stage result JSONs.*")
    lines.append("")

    # --- Overview table ---
    if probe_results:
        lines.append("## Overview")
        lines.append("")
        lines.append(
            "Best layer selected by combined AUC on the same CV folds used for evaluation "
            "(selection bias over 3 layers). See per-layer detail for the full breakdown."
        )
        lines.append("")
        lines.append("| Model | Dataset | N | Correct | Incorrect | Entropy AUC | Best Layer | Mahal AUC | Combined AUC | Δ (raw) | Δ (len-ctrl) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

        for model in sorted(probe_results):
            for ds in sorted(probe_results[model]):
                if ds.endswith("_cross"):
                    continue
                d = probe_results[model][ds]
                n_c = d["n_correct"]
                n_i = d["n_incorrect"]
                n = n_c + n_i
                ent = d["entropy_baseline"]["roc_auc_mean"]
                bl = best_layer(d["layers"])
                lr = d["layers"][bl]
                mah = lr["mahalanobis_only"]["roc_auc_mean"]
                comb = lr["combined"]["roc_auc_mean"]
                delta = lr["delta_vs_entropy"]
                lc_delta = lr.get("length_controlled_delta", None)
                lines.append(
                    f"| {model} | {ds} | {n} | {n_c} ({100*n_c/n:.0f}%) | {n_i} "
                    f"| {fmt(ent)} | L{bl} | {fmt(mah)} | **{fmt(comb)}** "
                    f"| {delta:+.3f} | {'+' if lc_delta and lc_delta >= 0 else ''}{fmt(lc_delta)} |"
                )
        lines.append("")

    # --- Primary geometry variants (unsupervised only) ---
    has_controls_or_base = bool(probe_results)
    if has_controls_or_base:
        def _dv(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "—"
            return f"{v:+.4f}"

        lines.append("## Primary geometry variants")
        lines.append("")
        lines.append(
            "Unsupervised methods only: the reference distribution is fitted on correct traces "
            "alone (one-class setup, no correctness labels required). "
            "Δ columns are AUC − entropy baseline; len-ctrl Δ is the length-controlled delta."
        )
        lines.append("")
        lines.append("| Model | Dataset | Layer | Entropy AUC | Base Δ | Base len-ctrl Δ | RMD Δ | Norm-RMD Δ |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for model in sorted(probe_results):
            for ds in sorted(probe_results[model]):
                if ds.endswith("_cross"):
                    continue
                d = probe_results[model][ds]
                entropy_auc = d["entropy_baseline"]["roc_auc_mean"]
                for layer in sorted(d["layers"], key=int):
                    lr = d["layers"][layer]
                    base_delta = lr.get("delta_vs_entropy", float("nan"))
                    lc_delta = lr.get("length_controlled_delta", None)
                    rmd_delta = lr.get("raw_rmd_delta_vs_entropy", float("nan"))
                    norm_rmd_delta = lr.get("normalized_rmd_delta_vs_entropy", float("nan"))
                    lc_str = _dv(lc_delta) if lc_delta is not None else "—"
                    lines.append(
                        f"| {model} | {ds} | L{layer} "
                        f"| {fmt(entropy_auc)} "
                        f"| {_dv(base_delta)} | {lc_str} "
                        f"| {_dv(rmd_delta)} | {_dv(norm_rmd_delta)} |"
                    )
        lines.append("")

    # --- Per-layer detail ---
    if probe_results:
        lines.append("## Per-layer detail")
        lines.append("")
        for model in sorted(probe_results):
            for ds in sorted(probe_results[model]):
                if ds.endswith("_cross"):
                    continue
                d = probe_results[model][ds]
                lines.append(f"### {model} / {ds}")
                lines.append("")
                ent = d["entropy_baseline"]["roc_auc_mean"]
                ent_std = d["entropy_baseline"]["roc_auc_std"]
                lines.append(f"Entropy-only baseline: {pm(ent, ent_std)}")
                lines.append("")
                lines.append("| Layer | Mahal-only | Combined | Δ (raw) | Δ (len-ctrl) | Conf-wrong p |")
                lines.append("|---|---|---|---|---|---|")
                for layer in sorted(d["layers"], key=int):
                    lr = d["layers"][layer]
                    mah = lr["mahalanobis_only"]["roc_auc_mean"]
                    mah_std = lr["mahalanobis_only"]["roc_auc_std"]
                    comb = lr["combined"]["roc_auc_mean"]
                    comb_std = lr["combined"]["roc_auc_std"]
                    delta = lr["delta_vs_entropy"]
                    lc = lr.get("length_controlled_delta", None)
                    cw = lr["confident_wrong"]
                    cw_p = cw.get("mannwhitney_pvalue")
                    if cw.get("skipped") or cw_p is None:
                        cw_cell = "—"
                    elif isinstance(cw_p, float) and math.isnan(cw_p):
                        cw_cell = "—"
                    else:
                        sig = "***" if cw_p < 0.01 else ("*" if cw_p < 0.05 else "ns")
                        cw_cell = f"{cw_p:.2e} {sig}"
                    lines.append(
                        f"| L{layer} | {pm(mah, mah_std)} | {pm(comb, comb_std)} "
                        f"| {delta:+.4f} | {lc:+.4f} | {cw_cell} |"
                    )
                lines.append("")

    # --- PCA dimension ablation ---
    if pca_ablation_results:
        def _pca_dim_sort_key(dim: str) -> tuple[int, int]:
            return (1, 0) if dim == "max" else (0, int(dim))

        def _delta_cell(value):
            return "—" if value is None else f"{value:+.4f}"

        lines.append("## PCA-dimension ablation (base geometry)")
        lines.append("")
        lines.append(
            "Base-only sweep over PCA truncation size. All evaluated dimensions are shown "
            "explicitly for each layer so the tradeoff is visible without collapsing to the "
            "single best PCA dimension."
        )
        lines.append("")
        lines.append("| Model | Dataset | Layer | PCA dim | Mahal-only | Combined | Δ (raw) | Δ (len-ctrl) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for model in sorted(pca_ablation_results):
            for ds in sorted(pca_ablation_results[model]):
                result = pca_ablation_results[model][ds]
                for layer in sorted(result.get("layers", {}), key=int):
                    layer_data = result["layers"][layer]
                    dims = [str(dim) for dim in result.get("settings", {}).get("pca_dims", [])]
                    if not dims:
                        dims = sorted(layer_data.get("dims", {}), key=_pca_dim_sort_key)
                    for dim in dims:
                        dim_metrics = layer_data.get("dims", {}).get(dim)
                        if not dim_metrics:
                            continue
                        mahal_auc = dim_metrics["mahalanobis_only"]["roc_auc_mean"]
                        combined_auc = dim_metrics["combined"]["roc_auc_mean"]
                        delta = dim_metrics.get("delta_vs_entropy")
                        lc_delta = dim_metrics.get("length_controlled_delta")
                        lines.append(
                            f"| {model} | {ds} | L{layer} | {dim} "
                            f"| {fmt(mahal_auc)} | {fmt(combined_auc)} "
                            f"| {_delta_cell(delta)} | {_delta_cell(lc_delta)} |"
                        )
        lines.append("")

    # --- Controls analysis (normalized & RMD) ---
    has_controls = any(
        "normalized_delta_vs_entropy" in d["layers"].get(list(d["layers"].keys())[0], {})
        for model in probe_results
        for ds in probe_results[model]
        if not ds.endswith("_cross")
        for d in [probe_results[model][ds]]
        if d.get("layers")
    )
    if has_controls:
        lines.append("## Controls analysis (normalized & robust Mahalanobis)")
        lines.append("")
        lines.append(
            "Checks whether the raw Mahalanobis signal survives length-normalization "
            "and robust distance estimation."
        )
        lines.append("")
        lines.append("| Model | Dataset | Layer | Base Δ | Norm Δ | RMD Δ | Norm-RMD Δ |")
        lines.append("|---|---|---|---|---|---|---|")
        for model in sorted(probe_results):
            for ds in sorted(probe_results[model]):
                if ds.endswith("_cross"):
                    continue
                d = probe_results[model][ds]
                for layer in sorted(d["layers"], key=int):
                    lr = d["layers"][layer]
                    base_delta = lr.get("delta_vs_entropy", float("nan"))
                    norm_delta = lr.get("normalized_delta_vs_entropy", float("nan"))
                    rmd_delta = lr.get("raw_rmd_delta_vs_entropy", float("nan"))
                    norm_rmd_delta = lr.get("normalized_rmd_delta_vs_entropy", float("nan"))
                    def _delta(v):
                        return f"{v:+.4f}" if not math.isnan(v) else "—"
                    lines.append(
                        f"| {model} | {ds} | L{layer} "
                        f"| {_delta(base_delta)} | {_delta(norm_delta)} "
                        f"| {_delta(rmd_delta)} | {_delta(norm_rmd_delta)} |"
                    )
        lines.append("")

    # --- Subspace analysis ---
    has_subspace = any(
        "contrast_delta_vs_entropy" in d["layers"].get(list(d["layers"].keys())[0], {})
        for model in probe_results
        for ds in probe_results[model]
        if not ds.endswith("_cross")
        for d in [probe_results[model][ds]]
        if d.get("layers")
    )
    if has_subspace:
        def _d(v):
            return f"{v:+.4f}" if not math.isnan(v) else "—"

        lines.append("## Subspace analysis — label-informed geometry")
        lines.append("")
        lines.append(
            "**These methods require correctness labels at fit time** and are therefore "
            "label-informed, not unsupervised. Both contrast subspace and the low-rank sweep "
            "use correct *and* incorrect training traces to find discriminative directions. "
            "Treat these results as a supervised upper bound, not a continuation of the "
            "one-class geometry story. All ranks shown (no argmax selection) to avoid "
            "within-CV selection bias."
        )
        lines.append("")

        # Contrast subspace
        lines.append("### Contrast subspace")
        lines.append("")
        lines.append("| Model | Dataset | Layer | Contrast Δ |")
        lines.append("|---|---|---|---|")
        for model in sorted(probe_results):
            for ds in sorted(probe_results[model]):
                if ds.endswith("_cross"):
                    continue
                d = probe_results[model][ds]
                for layer in sorted(d["layers"], key=int):
                    lr = d["layers"][layer]
                    contrast_delta = lr.get("contrast_delta_vs_entropy", float("nan"))
                    lines.append(f"| {model} | {ds} | L{layer} | {_d(contrast_delta)} |")
        lines.append("")

        # Low-rank sweep — all ranks
        has_sweep = any(
            "low_rank_subspace_sweep" in probe_results[model][ds]["layers"].get(
                list(probe_results[model][ds]["layers"].keys())[0], {}
            )
            for model in probe_results
            for ds in probe_results[model]
            if not ds.endswith("_cross") and probe_results[model][ds].get("layers")
        )
        if has_sweep:
            lines.append("### Low-rank subspace sweep (all ranks)")
            lines.append("")
            lines.append("| Model | Dataset | Layer | Rank | Centroid-combined Δ | Mahal-combined Δ |")
            lines.append("|---|---|---|---|---|---|")
            for model in sorted(probe_results):
                for ds in sorted(probe_results[model]):
                    if ds.endswith("_cross"):
                        continue
                    d = probe_results[model][ds]
                    entropy_mean = d["entropy_baseline"]["roc_auc_mean"]
                    for layer in sorted(d["layers"], key=int):
                        lr = d["layers"][layer]
                        sweep = lr.get("low_rank_subspace_sweep")
                        if not sweep:
                            continue
                        for rank in sorted(sweep, key=int):
                            rs = sweep[rank]
                            c_auc = rs.get("centroid_combined", {}).get("roc_auc_mean", float("nan"))
                            m_auc = rs.get("mahalanobis_combined", {}).get("roc_auc_mean", float("nan"))
                            c_delta = c_auc - entropy_mean if not math.isnan(c_auc) else float("nan")
                            m_delta = m_auc - entropy_mean if not math.isnan(m_auc) else float("nan")
                            lines.append(
                                f"| {model} | {ds} | L{layer} | {rank} "
                                f"| {_d(c_delta)} | {_d(m_delta)} |"
                            )
            lines.append("")

    # --- Cross-model transfer ---
    has_cross = any(
        ds.endswith("_cross")
        for model in probe_results
        for ds in probe_results[model]
    )
    if has_cross:
        lines.append("## Cross-model Mahalanobis transfer")
        lines.append("")
        lines.append("Reference manifold fitted on the *other* model's correct traces,")
        lines.append("then used to classify *this* model's traces.")
        lines.append("")
        lines.append("| Model | Dataset | Layer | Native Mahal | Cross Mahal | Transfer % | Cross Combined | Clf Mahal | Clf Combined |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for model in sorted(probe_results):
            for ds in sorted(probe_results[model]):
                if not ds.endswith("_cross"):
                    continue
                base_ds = ds.replace("_cross", "")
                d = probe_results[model][ds]
                native_data = probe_results.get(model, {}).get(base_ds, {})
                cross = d.get("cross_model_transfer", {})

                for layer in sorted(cross, key=int):
                    cr = cross[layer]
                    cross_mah = cr["cross_mahal_only"]["roc_auc_mean"]
                    cross_comb = cr["cross_combined"]["roc_auc_mean"]
                    native_mah = native_data.get("layers", {}).get(layer, {}).get(
                        "mahalanobis_only", {}
                    ).get("roc_auc_mean", None)
                    transfer_pct = (
                        f"{100 * cross_mah / native_mah:.0f}%"
                        if native_mah and native_mah > 0
                        else "—"
                    )
                    clf_mah = cr.get("clf_transfer_mahal_only", {}).get("roc_auc", None)
                    clf_comb = cr.get("clf_transfer_combined", {}).get("roc_auc", None)
                    lines.append(
                        f"| {model} | {base_ds} | L{layer} "
                        f"| {fmt(native_mah)} | {fmt(cross_mah)} | {transfer_pct} "
                        f"| {fmt(cross_comb)} | {fmt(clf_mah)} | {fmt(clf_comb)} |"
                    )
        lines.append("")

    # --- Difficulty stratification ---
    has_diff = any(
        "difficulty_stratification" in probe_results[model].get(ds, {})
        for model in probe_results
        for ds in probe_results[model]
    )
    if has_diff:
        lines.append("## Difficulty stratification (MATH-500)")
        lines.append("")
        for model in sorted(probe_results):
            for ds in sorted(probe_results[model]):
                if ds.endswith("_cross"):
                    continue
                d = probe_results[model][ds]
                diff = d.get("difficulty_stratification")
                if not diff:
                    continue
                lines.append(f"### {model} / {ds}")
                lines.append("")
                lines.append("| Bucket | N | Correct | Incorrect | Ent AUC | Comb AUC | Δ | Combined 95% CI | Entropy 95% CI |")
                lines.append("|---|---|---|---|---|---|---|---|---|")
                for bucket in ["level_1", "level_2", "level_3", "level_4", "level_5",
                               "easy_1-2", "medium_3", "hard_4-5"]:
                    dr = diff.get(bucket)
                    if not dr or dr.get("skipped"):
                        continue
                    n = dr["n_total"]
                    nc = dr["n_correct"]
                    ni = dr["n_incorrect"]
                    ea = dr["entropy_only"]["roc_auc_mean"]
                    ca = dr["combined"]["roc_auc_mean"]
                    delta = dr["delta"]
                    ci = dr.get("bootstrap_ci", {})
                    cci = ci.get("combined_ci95", (None, None))
                    eci = ci.get("entropy_ci95", (None, None))
                    cci_str = f"[{cci[0]:.3f}, {cci[1]:.3f}]" if cci[0] is not None else "—"
                    eci_str = f"[{eci[0]:.3f}, {eci[1]:.3f}]" if eci[0] is not None else "—"
                    lines.append(
                        f"| {bucket} | {n} | {nc} | {ni} | {fmt(ea)} | {fmt(ca)} "
                        f"| {delta:+.3f} | {cci_str} | {eci_str} |"
                    )
                lines.append("")

    # --- Subject stratification ---
    has_subj = any(
        "subject_stratification" in probe_results[model].get(ds, {})
        for model in probe_results
        for ds in probe_results[model]
    )
    if has_subj:
        lines.append("## Subject stratification (MATH-500)")
        lines.append("")
        for model in sorted(probe_results):
            for ds in sorted(probe_results[model]):
                if ds.endswith("_cross"):
                    continue
                d = probe_results[model][ds]
                subj = d.get("subject_stratification")
                if not subj:
                    continue
                lines.append(f"### {model} / {ds}")
                lines.append("")
                lines.append("| Subject | N | Correct | Incorrect | Ent AUC | Mahal AUC | Comb AUC | Δ |")
                lines.append("|---|---|---|---|---|---|---|---|")
                for subject in sorted(subj):
                    sr = subj[subject]
                    if sr.get("skipped"):
                        continue
                    n = sr["n_total"]
                    nc = sr["n_correct"]
                    ni = sr["n_incorrect"]
                    ea = sr["entropy_only"]["roc_auc_mean"]
                    ma = sr["mahalanobis_only"]["roc_auc_mean"]
                    ca = sr["combined"]["roc_auc_mean"]
                    delta = sr["delta"]
                    lines.append(
                        f"| {subject} | {n} | {nc} | {ni} | {fmt(ea)} | {fmt(ma)} | {fmt(ca)} | {delta:+.3f} |"
                    )
                lines.append("")

    # --- Prefix filtering ---
    has_prefix_filter = any(prefix_filter_results.get(model) for model in prefix_filter_results)
    if has_prefix_filter:
        lines.append("## Prefix filtering (abort/restart simulation)")
        lines.append("")
        lines.append(
            "Full sweep over all settings and threshold quantiles. "
            "All operating points are shown to avoid selection bias across "
            "score kinds, layers, prefix lengths, and quantile thresholds."
        )
        lines.append("")
        lines.append("| Model | Dataset | Setting | Quantile | Pass@1 | Baseline Pass@1 | ΔPass | Token Savings | False Abort Rate | Avg Aborts / Problem | Max Restarts |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

        for model in sorted(prefix_filter_results):
            for ds in sorted(prefix_filter_results[model]):
                data = prefix_filter_results[model][ds]
                max_restarts = data.get("max_restarts", "—")
                dataset_label = data.get("dataset", ds)
                has_any = False

                for setting in data.get("settings", {}).values():
                    if setting.get("skipped"):
                        continue
                    layer = setting.get("layer")
                    layer_label = "-" if layer is None else f"L{layer}"
                    setting_label = (
                        f"{setting['score_kind']} {layer_label} k={setting['prefix_len']}"
                    )
                    baseline = setting["baseline"]["pass_at_1_mean"]

                    for quantile_key, payload in sorted(
                        setting.get("thresholds", {}).items(), key=lambda x: float(x[0])
                    ):
                        if payload.get("skipped"):
                            continue
                        has_any = True
                        lines.append(
                            f"| {model} | {dataset_label} | {setting_label} "
                            f"| {payload['quantile']:.2f} | {payload['pass_at_1_mean']:.3f} "
                            f"| {baseline:.3f} | {payload['pass_delta_vs_baseline']:+.3f} "
                            f"| {payload['token_savings_mean']:+.3f} | {payload['false_abort_rate']:.3f} "
                            f"| {payload['avg_aborts_per_problem']:.3f} | {max_restarts} |"
                        )

                if not has_any:
                    lines.append(
                        f"| {model} | {dataset_label} | — | — | — | — | — | — | — | — | {max_restarts} |"
                    )
        lines.append("")

    if bestofn_results:
        render_bestofn_section(
            bestofn_results, concordance_results or {}, lines
        )

    if prompt_decomposition_results:
        render_prompt_decomposition_section(prompt_decomposition_results, lines)

    if prompt_selection_results:
        render_prompt_selection_section(prompt_selection_results, lines)

    if application_alignment_result:
        render_application_alignment_section(application_alignment_result, lines)

    if one_class_sweep_results:
        render_one_class_sweep_section(one_class_sweep_results, lines)

    if selective_results:
        render_selective_prediction_section(selective_results, lines)

    md = "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(md)
    print(f"Summary written to {output_path}")


def _selector_mean(payload: dict | None) -> float | None:
    if not payload or payload.get("skipped"):
        return None
    return payload.get("pass_at_1_mean")


def _best_layer_selector(n_result: dict, selector: str) -> tuple[float, str] | None:
    candidates = []
    for layer, layer_result in n_result.get("layers", {}).items():
        value = _selector_mean(layer_result.get(selector))
        if value is not None:
            candidates.append((float(value), str(layer)))
    return max(candidates) if candidates else None


def _fmt_layer_selector(candidate: tuple[float, str] | None) -> str:
    if candidate is None:
        return "—"
    value, layer = candidate
    return f"{value:.3f} (L{layer})"


def render_bestofn_section(
    bestofn_results: dict,
    concordance_results: dict,
    lines: list[str],
) -> None:
    """Append Best-of-N selection and legacy concordance diagnostics."""
    lines.extend(
        [
            "## Best-of-N Selection",
            "",
            (
                "Leakage-safe prompt-grouped CV. Each row reports the largest available "
                "N for that artifact; layer-based columns select the best layer for that "
                "selector within the reported run."
            ),
            "",
            (
                "| Model | Scale | Dataset | N | Random | Majority | Entropy | "
                "Best Mahal | Best combined | Best RMD | Best RMD+entropy | Oracle |"
            ),
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: |",
        ]
    )

    for model in sorted(bestofn_results):
        for scale in sorted(bestofn_results[model]):
            for ds in sorted(bestofn_results[model][scale]):
                data = bestofn_results[model][scale][ds]
                valid_n = [
                    int(key)
                    for key, value in data.get("n_values", {}).items()
                    if not value.get("skipped")
                ]
                if not valid_n:
                    continue
                n = max(valid_n)
                n_result = data["n_values"][str(n)]
                selectors = n_result.get("selectors", {})
                lines.append(
                    f"| {model} | {scale} | {ds} | {n} "
                    f"| {fmt(_selector_mean(selectors.get('random')))} "
                    f"| {fmt(_selector_mean(selectors.get('majority_vote')))} "
                    f"| {fmt(_selector_mean(selectors.get('entropy_only')))} "
                    f"| {_fmt_layer_selector(_best_layer_selector(n_result, 'mahalanobis_only'))} "
                    f"| {_fmt_layer_selector(_best_layer_selector(n_result, 'combined'))} "
                    f"| {_fmt_layer_selector(_best_layer_selector(n_result, 'rmd_only'))} "
                    f"| {_fmt_layer_selector(_best_layer_selector(n_result, 'rmd_combined'))} "
                    f"| {fmt(_selector_mean(selectors.get('oracle_pass_at_n')))} |"
                )
    lines.append("")

    if not concordance_results:
        return

    lines.extend(
        [
            "### Within-prompt concordance diagnostic",
            "",
            (
                "Legacy diagnostic using raw Mahalanobis with an in-sample reference "
                "fit on all correct pilot traces. Treat this as descriptive only; the "
                "OOF raw/RMD prompt decomposition is the confirmatory replacement."
            ),
            "",
            "| Model | Scale | Dataset | Layer | Concordance | SD | Mixed prompts |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for model in sorted(concordance_results):
        for scale in sorted(concordance_results[model]):
            for ds in sorted(concordance_results[model][scale]):
                data = concordance_results[model][scale][ds]
                for layer, layer_result in sorted(
                    data.get("layers", {}).items(), key=lambda item: int(item[0])
                ):
                    lines.append(
                        f"| {model} | {scale} | {ds} | L{layer} "
                        f"| {fmt(layer_result.get('mean_concordance'))} "
                        f"| {fmt(layer_result.get('std_concordance'))} "
                        f"| {layer_result.get('n_problems_evaluated', 0)} |"
                    )
    lines.append("")


def render_selective_prediction_section(
    selective_results: dict, lines: list[str]
) -> None:
    """Append selective prediction results to lines in-place."""
    if not selective_results:
        return

    lines.append("## Selective Prediction")
    lines.append("")
    lines.append(
        "Coverage-accuracy evaluation: trust vs. abstain on a single completed trace. "
        "AUSC is normalised over [min_coverage, 1.0]. "
        "Acc@K = accuracy at the most selective operating point with coverage ≥ K."
    )
    lines.append("")

    # Collect all scorer names across all results for a consistent column order
    op_keys: list[str] = []
    for model in sorted(selective_results):
        for ds in sorted(selective_results[model]):
            data = selective_results[model][ds]
            ops = data.get("settings", {}).get("operating_points", [])
            op_keys = [str(o) for o in ops]
            break
        if op_keys:
            break
    if not op_keys:
        op_keys = ["0.6", "0.7", "0.8", "0.9"]

    op_header = " | ".join(f"Acc@{k}" for k in op_keys)
    lines.append(f"| Model | Dataset | Scorer | n_eval | AUSC | {op_header} |")
    lines.append("| --- | --- | --- | --- | --- | " + " | ".join(["---"] * len(op_keys)) + " |")

    for model in sorted(selective_results):
        for ds in sorted(selective_results[model]):
            data = selective_results[model][ds]
            scorers = data.get("scorers", {})
            base_acc = data["n_correct"] / data["n_traces"]
            for scorer_name, s in scorers.items():
                n_eval = s.get("n_eval", 0)
                ausc_val = s.get("ausc")
                ausc_str = fmt(ausc_val, 4) if ausc_val is not None else "—"
                op_dict = s.get("operating_points", {})
                op_cells = " | ".join(
                    fmt(op_dict.get(k), 3) for k in op_keys
                )
                lines.append(
                    f"| {model} | {ds} | {scorer_name} | {n_eval} | {ausc_str} | {op_cells} |"
                )
        lines.append("")


def render_prompt_decomposition_section(results: dict, lines: list[str]) -> None:
    lines.extend(
        [
            "## Prompt Decomposition",
            "",
            (
                "Prompt-grouped out-of-fold scores. Higher scores predict correctness; "
                "all reference-dependent quantities are fit on training prompts only."
            ),
            "",
        ]
    )
    legacy = any(
        set(layer_result.get("methods", {})) <= {"raw", "rmd"}
        for model_results in results.values()
        for data in model_results.values()
        for layer_result in data.get("layers", {}).values()
    )
    if legacy:
        lines.extend(
            [
                (
                    "**Schema note:** at least one artifact has the raw/RMD-only legacy "
                    "schema. Rerun prompt decomposition to add entropy, log-probability, "
                    "length, activation-norm, and centroid controls."
                ),
                "",
            ]
        )
    lines.extend(
        [
            (
                "| Model | Dataset | Layer | Method | Pooled AUC | Centered AUC | "
                "Within AUC | ICC | Prompt/pass Spearman |"
            ),
            "|:---|:---|:---|:---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in sorted(results):
        for dataset in sorted(results[model]):
            data = results[model][dataset]
            for layer, layer_result in sorted(
                data.get("layers", {}).items(), key=lambda item: int(item[0])
            ):
                for method, method_result in layer_result.get("methods", {}).items():
                    metrics = method_result.get("metrics", {})
                    lines.append(
                        f"| {model} | {dataset} | L{layer} | {method} "
                        f"| {fmt(metrics.get('pooled_auc'))} "
                        f"| {fmt(metrics.get('prompt_centered_auc'))} "
                        f"| {fmt(metrics.get('within_prompt_pair_weighted'))} "
                        f"| {fmt(metrics.get('score_icc'))} "
                        f"| {fmt(metrics.get('prompt_score_pass_rate_spearman'))} |"
                    )
    lines.append("")

    contrastive_entries = [
        (model, dataset, data)
        for model in sorted(results)
        for dataset in sorted(results[model])
        for data in [results[model][dataset]]
        if data.get("contrastive", {}).get("regions")
    ]
    if contrastive_entries:
        lines.extend(
            [
                "### Prompt-contrastive correctness (parseable-only primary)",
                "",
                (
                    "These supervised direction scores are reported separately from "
                    "the unsupervised decomposition table. Values below are restricted "
                    "to traces with parseable final answers."
                ),
                "",
                "| Model | Dataset | Layer | Region | Centered AUC | Within macro | Mixed prompts |",
                "|:---|:---|---:|:---|---:|---:|---:|",
            ]
        )
        for model, dataset, data in contrastive_entries:
            for layer, layer_result in sorted(
                data.get("layers", {}).items(), key=lambda item: int(item[0])
            ):
                parseable = layer_result.get("parseable_only", {}).get("methods", {})
                for method in (
                    "contrast_full",
                    "contrast_high_entropy_q20",
                    "contrast_tail_q20",
                ):
                    metrics = parseable.get(method)
                    if not metrics:
                        continue
                    lines.append(
                        f"| {model} | {dataset} | L{layer} | {method} | "
                        f"{fmt(metrics.get('prompt_centered_auc'))} | "
                        f"{fmt(metrics.get('within_prompt_macro'))} | "
                        f"{metrics.get('n_mixed_prompts', 0)} |"
                    )
        lines.extend(
            [
                "",
                "| Model | Dataset | Layer | Region | Observed alignment | Null mean | Null p |",
                "|:---|:---|---:|:---|---:|---:|---:|",
            ]
        )
        for model, dataset, data in contrastive_entries:
            for diagnostic in data.get("contrastive", {}).get(
                "alignment_diagnostics", []
            ):
                null = diagnostic.get("null", {})
                lines.append(
                    f"| {model} | {dataset} | L{diagnostic['layer']} | "
                    f"{diagnostic['region']} | "
                    f"{fmt(diagnostic.get('observed_alignment'))} | "
                    f"{fmt(null.get('mean'))} | {fmt(null.get('p_value'))} |"
                )
        lines.append("")


def render_prompt_selection_section(results: dict, lines: list[str]) -> None:
    lines.extend(["## OOF Prompt Selection", ""])
    strict_invalid_policy = any(
        data.get("settings", {}).get("invalid_answer_policy")
        == "count as failure"
        for model_results in results.values()
        for data in model_results.values()
    )
    if strict_invalid_policy:
        lines.extend(
            [
                "Unparsed answers are counted as failures. They are not "
                "silently removed from voting.",
                "",
            ]
        )

    parsing_rows = []
    for model in sorted(results):
        for dataset in sorted(results[model]):
            data = results[model][dataset]
            for layer, layer_result in sorted(
                data.get("layers", {}).items(), key=lambda item: int(item[0])
            ):
                parsing = layer_result.get("answer_parsing")
                if parsing:
                    parsing_rows.append((model, dataset, layer, parsing))
    if parsing_rows:
        lines.extend(
            [
                "### Answer Parsing",
                "",
                (
                    "| Model | Dataset | Layer | Parsed traces | Parse rate | "
                    "Correct parse rate | Incorrect parse rate | "
                    "Prompts with no parsed answer |"
                ),
                "|:---|:---|:---|---:|---:|---:|---:|---:|",
            ]
        )
        for model, dataset, layer, parsing in parsing_rows:
            lines.append(
                f"| {model} | {dataset} | L{layer} "
                f"| {parsing.get('n_parsed', 0)}/{parsing.get('n_traces', 0)} "
                f"| {fmt(parsing.get('parse_rate'))} "
                f"| {fmt(parsing.get('correct_parse_rate'))} "
                f"| {fmt(parsing.get('incorrect_parse_rate'))} "
                f"| {parsing.get('n_prompts_without_parsed_answer', 0)} |"
            )
        lines.append("")

    lines.extend(
        [
            "### Selection Results",
            "",
            "| Model | Dataset | Layer | Selector | Pass@1 |",
            "|:---|:---|:---|:---|---:|",
        ]
    )
    for model in sorted(results):
        for dataset in sorted(results[model]):
            data = results[model][dataset]
            for layer, layer_result in sorted(
                data.get("layers", {}).items(), key=lambda item: int(item[0])
            ):
                for selector, payload in layer_result.get("selectors", {}).items():
                    lines.append(
                        f"| {model} | {dataset} | L{layer} | {selector} "
                        f"| {fmt(payload.get('pass_at_1'))} |"
                    )
    lines.append("")


def render_application_alignment_section(result: dict, lines: list[str]) -> None:
    lines.extend(
        [
            "## Application Alignment",
            "",
            result.get("warning", "Exploratory descriptive comparison."),
            "",
            (
                "| Model | Layer | Method | Within AUC | ICC | Top-1 gain | "
                "Selective AUSC gain |"
            ),
            "|:---|---:|:---|---:|---:|---:|---:|",
        ]
    )
    for row in result.get("conditions", []):
        lines.append(
            f"| {row.get('model')} | {row.get('layer')} | {row.get('method')} "
            f"| {fmt(row.get('within_prompt_pair_weighted'))} "
            f"| {fmt(row.get('score_icc'))} "
            f"| {fmt(row.get('top1_gain_over_random'))} "
            f"| {fmt(row.get('selective_ausc_gain_over_entropy'))} |"
        )
    lines.append("")


def render_one_class_sweep_section(results: dict, lines: list[str]) -> None:
    lines.extend(
        [
            "## One-Class Mechanism Sweep",
            "",
            "All configured dimensions are shown; no post-hoc best dimension is selected.",
            "",
            (
                "| Model | Dataset | Layer | Dimension | Method | Pooled ROC | "
                "Fold ROC | Pooled PR | N |"
            ),
            "|:---|:---|:---|---:|:---|---:|:---|---:|---:|",
        ]
    )
    for model in sorted(results):
        for dataset in sorted(results[model]):
            data = results[model][dataset]
            for layer, layer_result in sorted(
                data.get("layers", {}).items(), key=lambda item: int(item[0])
            ):
                for dimension, dimension_result in sorted(
                    layer_result.get("dimensions", {}).items(),
                    key=lambda item: int(item[0]),
                ):
                    for method, metrics in dimension_result.get(
                        "methods", {}
                    ).items():
                        fold_mean = metrics.get("fold_roc_auc_mean")
                        fold_std = metrics.get("fold_roc_auc_std")
                        fold_text = (
                            "—"
                            if fold_mean is None or fold_std is None
                            else f"{fold_mean:.3f} ± {fold_std:.3f}"
                        )
                        lines.append(
                            f"| {model} | {dataset} | L{layer} | {dimension} "
                            f"| {method} | {fmt(metrics.get('pooled_roc_auc'))} "
                            f"| {fold_text} | {fmt(metrics.get('pooled_pr_auc'))} "
                            f"| {metrics.get('n_eval', 0)} |"
                        )
    lines.append("")


def generate_combined_json(
    results: dict,
    output_path: str,
    supplemental: dict | None = None,
):
    """Write a combined JSON with all results."""
    payload = dict(results)
    if supplemental:
        payload["_supplemental"] = supplemental
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Combined JSON written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate results into summary tables")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root results directory")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Where to write summary files")
    parser.add_argument(
        "--profile",
        choices=("all", "current"),
        default="all",
        help="Result profile to render; current excludes retired/contaminated runs.",
    )
    args = parser.parse_args()

    results = load_results(args.results_dir)
    pca_ablation_results = load_pca_ablation_results(args.results_dir)
    selective_results = load_selective_results(args.results_dir)
    bestofn_results, concordance_results = load_bestofn_results(args.results_dir)
    prompt_decomposition_results = load_prompt_decomposition_results(
        args.results_dir
    )
    prompt_selection_results = load_prompt_selection_results(args.results_dir)
    application_alignment_result = load_application_alignment_result(
        args.results_dir
    )
    one_class_sweep_results = load_one_class_sweep_results(args.results_dir)

    if args.profile == "current":
        # The results directory intentionally retains historical artifacts.  Do
        # not let those artifacts silently re-enter the current summary merely
        # because the loader can recognize their schema.
        results = {
            model: {
                dataset: value
                for dataset, value in data.items()
                if "_cross" not in dataset
            }
            for model, data in results.items()
            if model in {"qwen", "qwen_dense"}
        }
        pca_ablation_results = {
            model: data
            for model, data in pca_ablation_results.items()
            if model == "qwen"
        }
        selective_results = {}
        bestofn_results = {
            model: data
            for model, data in bestofn_results.items()
            if model == "qwen"
        }
        concordance_results = {}
        prompt_decomposition_results = {
            model: data
            for model, data in prompt_decomposition_results.items()
            if model == "qwen"
        }
        prompt_selection_results = {
            model: data
            for model, data in prompt_selection_results.items()
            if model == "qwen"
        }
        application_alignment_result = None
        one_class_sweep_results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    md_path = os.path.join(args.output_dir, "SUMMARY.md")
    json_path = os.path.join(args.output_dir, "all_results.json")

    generate_markdown(
        results,
        md_path,
        pca_ablation_results=pca_ablation_results,
        selective_results=selective_results,
        bestofn_results=bestofn_results,
        concordance_results=concordance_results,
        prompt_decomposition_results=prompt_decomposition_results,
        prompt_selection_results=prompt_selection_results,
        application_alignment_result=application_alignment_result,
        one_class_sweep_results=one_class_sweep_results,
    )
    generate_combined_json(
        results,
        json_path,
        supplemental={
            "pca_ablation": pca_ablation_results,
            "selective_prediction": selective_results,
            "best_of_n": bestofn_results,
            "bestofn_concordance": concordance_results,
            "prompt_decomposition": prompt_decomposition_results,
            "prompt_selection": prompt_selection_results,
            "application_alignment": application_alignment_result,
            "one_class_sweep": one_class_sweep_results,
        },
    )


if __name__ == "__main__":
    main()
