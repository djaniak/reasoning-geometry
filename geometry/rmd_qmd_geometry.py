"""A 2-D slice through the real RMD and QMD score fields.

The 2026-08-08 ladder splits the label-efficiency gap into a supervision rung
(`rmd_tail_q20` vs `qmd_tail_q20`) and a decision-function-form rung
(`qmd_tail_q20` vs `probe_token_tail_q20`).  This module draws the geometry
behind that split.  Nothing in the figure is invented: the PCA basis, all three
Gaussians and the token probe come from ``label_efficiency.fit_budget`` at one
of the sweep's own budget/replicate splits, and the contours are the *actual*
score functions restricted to a plane -- not a cartoon fitted to look like them.

The plane is chosen to make the two rungs visible at once.  ``e1`` is the
class-contrast direction ``mu_incorrect - mu_correct``, the only direction a
linear probe can use.  ``e2`` is the direction of extremal variance *ratio*
between the two class covariances, searched inside the orthogonal complement of
``e1`` so that it carries no mean shift at all: pure shape, which a hyperplane
cannot read and a quadratic can.

This is an explanatory figure, not evidence.  It is one model, one layer, one
label draw and one plane; panels A-D are scored in-sample.  The pooled 30-draw
numbers in ``EXPERIMENT_LOG.md`` remain the authority for every claim about the
size of either rung.  No DVC stage, CPU only, cached data only.

The run behind the 2026-08-08 log entry::

    uv run python rmd_qmd_geometry.py \\
      --data_dir data/qwen_bestofn_full/math500 \\
      --oof results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv \\
      --output_dir results/rmd_qmd_geometry \\
      --layer 21 --max_new_tokens 1024 --budget 100 --replicate 0 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from scipy.linalg import eigh as generalized_eigh  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import geometry.label_efficiency as le  # noqa: E402
from analysis.analyze import load_all_traces, set_compute_dtype, set_max_reference_tokens  # noqa: E402
from applications.incremental_abstention import (  # noqa: E402
    _read_oof,
    aggregate_prompt_features,
    select_layer_rows,
)

DEFAULT_PCA_COMPONENTS = 128
DEFAULT_MAX_TOKENS_PER_TRACE = 256
DEFAULT_TOKENS_PER_TRACE_PLOTTED = 40
DEFAULT_GRID = 240


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def mahal(points: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    """The distance ``analyze.compute_mahal_distances`` computes, in PCA space.

    Note the square root.  Both features are differences of *distances*, not of
    squared distances, so neither is literally a log-likelihood ratio -- but
    they share the convention, which is what keeps the comparison matched.
    """
    diff = np.asarray(points, dtype=float) - mu
    return np.sqrt(np.maximum(np.sum((diff @ cov_inv) * diff, axis=1), 0.0))


def choose_plane(
    mu_correct: np.ndarray,
    mu_incorrect: np.ndarray,
    sigma_correct: np.ndarray,
    sigma_incorrect: np.ndarray,
) -> np.ndarray:
    """Return the orthonormal 2-D basis ``[e1, e2]`` the figure is drawn in.

    ``e1`` is the class-mean contrast.  ``e2`` maximizes ``|log(variance
    ratio)|`` between the two class covariances *within the complement of e1*,
    so it is orthogonal to the mean shift by construction rather than by a
    projection applied afterwards -- projecting an unconstrained extremal
    direction onto the complement degrades the ratio, sometimes to nothing.

    The ratio is extremal for the generalized symmetric-definite problem
    ``Sigma_cor v = lambda Sigma_inc v``.  It must be solved as such:
    ``Sigma_inc^-1 Sigma_cor`` is not symmetric, and handing it to a symmetric
    eigensolver returns a direction with no particular variance ratio at all.
    """
    e1 = np.asarray(mu_incorrect, dtype=float) - np.asarray(mu_correct, dtype=float)
    norm = np.linalg.norm(e1)
    if norm == 0:
        raise ValueError("the two class means coincide; there is no contrast direction")
    e1 = e1 / norm

    complement = np.linalg.svd(e1[None], full_matrices=True)[2][1:]
    eigvals, eigvecs = generalized_eigh(
        complement @ sigma_correct @ complement.T,
        complement @ sigma_incorrect @ complement.T,
    )
    e2 = complement.T @ eigvecs[:, int(np.argmax(np.abs(np.log(eigvals))))]
    return np.stack([e1, e2 / np.linalg.norm(e2)])


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def load_split(args: argparse.Namespace) -> tuple[dict, list, list, int]:
    """Reproduce one sweep split: prompt views, train ids, eval ids, layer."""
    rows, layer = select_layer_rows(_read_oof(args.oof), args.layer, context=args.oof)
    base = aggregate_prompt_features(
        rows,
        max_new_tokens=args.max_new_tokens,
        data_dir=args.data_dir,
        expected_traces=args.expected_traces,
    )
    eval_pool = sorted(
        pid
        for pid, entry in base.items()
        if entry["valid_plurality"] and entry["cap_count"] == 0
    )

    print(f"loading layer {layer} from {args.data_dir}", flush=True)
    traces = load_all_traces(
        args.data_dir,
        [layer],
        max_workers=args.max_workers,
        show_progress=False,
        include_auxiliary=True,
        auxiliary_fields={"entropies"},
        hidden_dtype=np.float16,
    )
    views = le.prepare_trace_views(
        traces, layer, max_tokens_per_trace=args.max_tokens_per_trace, seed=args.seed
    )
    del traces
    by_prompt = le.group_views_by_prompt(views)
    del views

    prompt_ids = sorted(set(by_prompt) & set(base))
    pool = [pid for pid in eval_pool if pid in by_prompt]
    permutation, eval_ids = le.replicate_split(
        prompt_ids, pool, max_budget=args.budget, seed=args.seed, replicate=args.replicate
    )
    return by_prompt, permutation[: args.budget], eval_ids, layer


def main() -> None:
    args = parse_args()
    set_compute_dtype(np.dtype("float32"))
    set_max_reference_tokens(args.max_reference_tokens)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_prompt, train_ids, eval_ids, layer = load_split(args)
    print(f"{len(train_ids)} train prompts, {len(eval_ids)} eval prompts", flush=True)

    fit = le.fit_budget(
        by_prompt,
        train_ids,
        layer,
        args.pca_components,
        max_tokens_per_trace=args.max_tokens_per_trace,
        seed=args.seed,
    )
    pca, mu_c, prec_c, mu_bg, prec_bg = fit["rmd_reference"]
    _, _, _, mu_i, prec_i = fit["qmd_reference"]

    sigma_c = np.linalg.inv(prec_c)
    sigma_i = np.linalg.inv(prec_i)
    basis = choose_plane(mu_c, mu_i, sigma_c, sigma_i)
    e1, e2 = basis

    def to_plane(points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=float) - mu_c) @ basis.T

    def from_plane(coords: np.ndarray) -> np.ndarray:
        return mu_c + coords @ basis

    # --- token clouds ----------------------------------------------------
    # Kept in both 128-D (for honest scoring) and 2-D (for plotting).
    rng = np.random.default_rng(args.seed)
    train_views = [v for pid in train_ids for v in by_prompt[int(pid)]]
    full_clouds, clouds = {}, {}
    for name, keep in (
        ("correct", lambda v: v["is_correct"]),
        ("incorrect", lambda v: not v["is_correct"]),
    ):
        blocks = []
        for view in train_views:
            if not keep(view):
                continue
            projected = pca.transform(np.asarray(view["tail"], dtype=np.float32))
            size = min(args.tokens_per_trace_plotted, len(projected))
            blocks.append(projected[rng.choice(len(projected), size=size, replace=False)])
        full_clouds[name] = np.concatenate(blocks, axis=0)
        clouds[name] = to_plane(full_clouds[name])

    # --- score fields on the plane --------------------------------------
    span_x, span_y = (
        np.percentile(
            np.abs(np.concatenate([clouds["correct"][:, k], clouds["incorrect"][:, k]])), 99
        )
        for k in (0, 1)
    )
    gx = np.linspace(-1.25 * span_x, 1.25 * span_x, args.grid)
    gy = np.linspace(-1.25 * span_y, 1.25 * span_y, args.grid)
    GX, GY = np.meshgrid(gx, gy)
    grid = from_plane(np.stack([GX.ravel(), GY.ravel()], axis=1))

    d_c = mahal(grid, mu_c, prec_c)
    # Negated so that higher = more correct-like, matching label_efficiency's
    # scoring convention.
    rmd = (-(d_c - mahal(grid, mu_bg, prec_bg))).reshape(GX.shape)
    qmd = (-(d_c - mahal(grid, mu_i, prec_i))).reshape(GX.shape)

    probe = fit["token_probe"]
    lda = probe["classifier"].decision_function(
        probe["scaler"].transform(grid)
    ).reshape(GX.shape)

    token_pts = np.concatenate([full_clouds["correct"], full_clouds["incorrect"]])
    token_plane = np.concatenate([clouds["correct"], clouds["incorrect"]])
    token_labels = np.concatenate([
        np.ones(len(full_clouds["correct"]), dtype=int),
        np.zeros(len(full_clouds["incorrect"]), dtype=int),
    ])
    token_d_c = mahal(token_pts, mu_c, prec_c)
    token_rmd = -(token_d_c - mahal(token_pts, mu_bg, prec_bg))
    token_qmd = -(token_d_c - mahal(token_pts, mu_i, prec_i))

    # --- held-out per-trace scores ---------------------------------------
    eval_views = [v for pid in eval_ids for v in by_prompt[int(pid)]]
    labels, s_rmd, s_qmd, s_lda = [], [], [], []
    for view in eval_views:
        projected = pca.transform(np.asarray(view["tail"], dtype=np.float32))
        d_correct = mahal(projected, mu_c, prec_c)
        labels.append(int(view["is_correct"]))
        s_rmd.append(-float(np.mean(d_correct - mahal(projected, mu_bg, prec_bg))))
        s_qmd.append(-float(np.mean(d_correct - mahal(projected, mu_i, prec_i))))
        s_lda.append(le.score_token_probe(projected, probe))
    labels = np.asarray(labels)
    s_rmd, s_qmd, s_lda = map(np.asarray, (s_rmd, s_qmd, s_lda))

    stats = {
        "model_data_dir": args.data_dir,
        "layer": int(layer),
        "budget": int(args.budget),
        "replicate": int(args.replicate),
        "seed": int(args.seed),
        "n_train_prompts": len(train_ids),
        "n_train_traces": fit["n_train_traces"],
        "n_correct_traces": fit["n_correct_traces"],
        "n_incorrect_traces": fit["n_incorrect_traces"],
        "n_eval_traces": int(len(labels)),
        "eval_accuracy": float(labels.mean()),
        # How far each rival Gaussian's mean sits from the correct one, in the
        # correct class's own metric.  RMD's background is the near one because
        # it *contains* the correct traces.
        "mahal_correct_to_background": float(mahal(mu_bg[None], mu_c, prec_c)[0]),
        "mahal_correct_to_incorrect": float(mahal(mu_i[None], mu_c, prec_c)[0]),
        "trace_auroc_rmd": float(roc_auc_score(labels, s_rmd)),
        "trace_auroc_qmd": float(roc_auc_score(labels, s_qmd)),
        "trace_auroc_token_lda": float(roc_auc_score(labels, s_lda)),
        # In-sample: the reference was fit on these very tokens.
        "token_auroc_rmd": float(roc_auc_score(token_labels, token_rmd)),
        "token_auroc_qmd": float(roc_auc_score(token_labels, token_qmd)),
        # How much of its range each score spends on the token cloud.
        "token_score_range_rmd": float(np.ptp(np.percentile(token_rmd, [1, 99]))),
        "token_score_range_qmd": float(np.ptp(np.percentile(token_qmd, [1, 99]))),
        "log_var_ratio_e2": float(np.log((e2 @ sigma_c @ e2) / (e2 @ sigma_i @ e2))),
        "e1_e2_orthogonality": float(abs(e1 @ e2)),
        "fraction_tokens_in_view": float(
            np.mean(
                (np.abs(token_plane[:, 0]) <= gx[-1])
                & (np.abs(token_plane[:, 1]) <= gy[-1])
            )
        ),
    }
    print(json.dumps(stats, indent=2), flush=True)
    (out_dir / "rmd_qmd_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    # --- figure ----------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    C_OK, C_BAD, C_BG = "#1f77b4", "#d62728", "#7f7f7f"

    def ellipse(ax, mu, prec, colour, label, ls="-"):
        cov = basis @ np.linalg.inv(prec) @ basis.T
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        centre = to_plane(mu[None])[0]
        for k in (1, 2):
            ax.add_patch(Ellipse(
                centre, 2 * k * np.sqrt(vals[0]), 2 * k * np.sqrt(vals[1]),
                angle=angle, fill=False, edgecolor=colour, lw=1.6, ls=ls,
                alpha=1.0 if k == 1 else 0.45,
                label=label if k == 1 else None,
            ))
        ax.plot(*centre, marker="x", color=colour, ms=7, mew=2)

    ax = axes[0, 0]
    ax.scatter(*clouds["correct"].T, s=2, alpha=0.13, color=C_OK, lw=0)
    ax.scatter(*clouds["incorrect"].T, s=2, alpha=0.13, color=C_BAD, lw=0)
    ellipse(ax, mu_c, prec_c, C_OK, "correct (foreground)")
    ellipse(ax, mu_bg, prec_bg, C_BG, "all tokens (RMD background)", ls="--")
    ellipse(ax, mu_i, prec_i, C_BAD, "incorrect (QMD background)")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.set_title(
        "A. The three Gaussians\n"
        "$\\|\\mu_{bg}-\\mu_{cor}\\|_{\\Sigma_{cor}}$ = "
        f"{stats['mahal_correct_to_background']:.2f}   vs   "
        "$\\|\\mu_{inc}-\\mu_{cor}\\|_{\\Sigma_{cor}}$ = "
        f"{stats['mahal_correct_to_incorrect']:.2f}"
    )

    def field(ax, values, title, extra_line=None):
        levels = np.linspace(np.percentile(values, 1), np.percentile(values, 99), 25)
        mesh = ax.contourf(GX, GY, values, levels=levels, cmap="RdBu", extend="both")
        ax.contour(GX, GY, values, levels=levels[::4], colors="k", linewidths=0.3, alpha=0.35)
        ax.contour(GX, GY, values, levels=[0.0], colors="k", linewidths=1.8)
        if extra_line is not None:
            ax.contour(GX, GY, extra_line, levels=[0.0], colors="w",
                       linewidths=1.8, linestyles="--")
        ax.scatter(*clouds["correct"].T, s=1.5, alpha=0.10, color="k", lw=0)
        ax.scatter(*clouds["incorrect"].T, s=1.5, alpha=0.10, color="w", lw=0)
        ax.plot(*to_plane(mu_c[None])[0], "x", color=C_OK, ms=8, mew=2.2)
        ax.plot(*to_plane(mu_i[None])[0], "x", color=C_BAD, ms=8, mew=2.2)
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)

    field(axes[0, 1], rmd,
          "B. RMD field  $-(d_{correct}-d_{all})$\n"
          f"in-sample token AUROC {stats['token_auroc_rmd']:.3f}, "
          f"1-99% spread {stats['token_score_range_rmd']:.2f}")
    field(axes[0, 2], qmd,
          "C. QMD field  $-(d_{correct}-d_{incorrect})$\n"
          f"in-sample token AUROC {stats['token_auroc_qmd']:.3f}, "
          f"spread {stats['token_score_range_qmd']:.2f}; "
          "linear probe = white dashed", extra_line=lda)

    ax = axes[1, 0]
    diff = rmd - qmd
    levels = np.linspace(np.percentile(diff, 1), np.percentile(diff, 99), 25)
    mesh = ax.contourf(GX, GY, diff, levels=levels, cmap="PuOr", extend="both")
    ax.contour(GX, GY, diff, levels=[0.0], colors="k", linewidths=1.5)
    ax.scatter(*clouds["correct"].T, s=1.5, alpha=0.10, color="k", lw=0)
    ax.scatter(*clouds["incorrect"].T, s=1.5, alpha=0.10, color="w", lw=0)
    ax.plot(*to_plane(mu_c[None])[0], "x", color=C_OK, ms=8, mew=2.2)
    ax.plot(*to_plane(mu_i[None])[0], "x", color=C_BAD, ms=8, mew=2.2)
    ax.set_title(
        "D. RMD $-$ QMD $= d_{incorrect}-d_{all}$\n"
        "$d_{correct}$ cancels exactly: this is the whole supervision term"
    )
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)

    def hist(ax, values, name, auroc):
        bins = np.linspace(*np.percentile(values, [0.5, 99.5]), 40)
        ax.hist(values[labels == 1], bins=bins, color=C_OK, alpha=0.6,
                label="correct traces", density=True)
        ax.hist(values[labels == 0], bins=bins, color=C_BAD, alpha=0.6,
                label="incorrect traces", density=True)
        ax.legend(fontsize=7.5)
        ax.set_title(f"{name}\nheld-out trace AUROC = {auroc:.3f}")
        ax.set_xlabel("tail-mean score (higher = more correct-like)")

    hist(axes[1, 1], s_rmd, "E. rmd_tail_q20", stats["trace_auroc_rmd"])
    hist(axes[1, 2], s_qmd, "F. qmd_tail_q20", stats["trace_auroc_qmd"])

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        ax.set_xlim(gx[0], gx[-1])
        ax.set_ylim(gy[0], gy[-1])
        ax.set_xlabel("class-contrast direction  $e_1 \\propto \\mu_{inc}-\\mu_{cor}$")
        ax.set_ylabel(
            "shape direction  $e_2$  ($\\sigma^2_{cor}/\\sigma^2_{inc}$ = "
            f"{np.exp(stats['log_var_ratio_e2']):.1f}$\\times$)"
        )

    fig.suptitle(
        "RMD vs QMD: the same estimator, two different second Gaussians "
        f"(layer {layer}, {args.budget} labelled prompts, 2-D slice of "
        f"PCA({args.pca_components}) through $\\mu_{{correct}}$)",
        fontsize=11.5,
    )
    fig.text(
        0.5, 0.005,
        "Contours are the real 128-D score functions restricted to the plane; "
        f"{100 * stats['fraction_tokens_in_view']:.1f}% of the plotted tail tokens fall "
        "inside the frame. Panels A-D use training tokens (the fits are in-sample "
        "there); E and F score held-out prompts.",
        ha="center", fontsize=8, style="italic",
    )
    fig.tight_layout(rect=(0, 0.022, 1, 0.965))
    fig.savefig(out_dir / "rmd_qmd_geometry.png", bbox_inches="tight")
    print(f"wrote {out_dir / 'rmd_qmd_geometry.png'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--oof", required=True, help="prompt-decomposition OOF csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_traces", type=int, default=8)
    parser.add_argument("--pca_components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument(
        "--max_tokens_per_trace", type=int, default=DEFAULT_MAX_TOKENS_PER_TRACE
    )
    parser.add_argument(
        "--tokens_per_trace_plotted", type=int, default=DEFAULT_TOKENS_PER_TRACE_PLOTTED
    )
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID)
    parser.add_argument("--max_reference_tokens", type=int, default=2_000_000)
    parser.add_argument("--max_workers", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    main()
