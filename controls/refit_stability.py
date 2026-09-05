"""Does anything survive refitting the whole pipeline on a different partition?

Every interval this project quotes resamples prompts with the fit held fixed.
Folds, reference manifolds, PCA bases, probe coefficients and layer choices are
all frozen at ``seed=42`` and only the evaluation set moves.  That is the
standard bootstrap, and it answers a narrow question: how much would this
number move if I had drawn different prompts *to score*?  It says nothing about
how much it would move if I had drawn a different partition *to fit on*.

The 2026-08-21 review makes that the fourth blocker, and is explicit that more
draws cannot substitute for it:

    Do ``B1-B0``, the peer-controlled residual, and the between/within result
    survive refitting the complete pipeline?  If the residual changes sign or
    varies widely, keep the original increment and demote the residual.

So this refits, rather than resamples.  One integer -- the refit seed -- is
threaded through *every* stage that has a fitting step, and the whole chain is
re-run from the trace batches:

1. ``applications.prompt_decomposition`` regenerates the OOF scores.  This is
   the expensive rung and the one that matters most: the seed drives
   ``make_prompt_folds``, so each fold's reference manifold is fitted on a
   different set of correct training traces, and ``rmd_tail_q20`` is a
   different score on every refit rather than a fixed column re-read.
2. ``applications.incremental_abstention`` refits the prompt-level readouts on
   those scores, giving ``B1 - B0``.
3. ``controls.last_token_probe`` refits the published-style probe, including
   its in-fold layer and penalty selection, giving ``pooled - macro``.
4. ``controls.peer_cost_ladder`` refits the peer rungs across all three models
   at that seed, giving the peer-controlled residual.

Stage 4 needs all three models at the same seed, which is why the work is
ordered seed-major: a run stopped early leaves *complete* refits behind rather
than three half-finished models.  Every stage writes a marker on success and is
skipped when re-entered, so the loop is resumable -- it is expected to be
started in a terminal multiplexer and left for several hours.

What is deliberately not varied: ``region_seed`` (it selects the random_q20
control region, which nothing here reads) and ``alignment_seed`` (the alignment
shuffles are disabled, since a permutation null is not what is being measured).
Varying those would broaden the question from "does a different partition
change the answer" to "does any randomness change the answer", and the review
asks the first one.

Bootstrap draws inside each refit are lowered by default.  They only set the
per-refit interval, and the quantity of interest here is the *spread of point
estimates across refits*, which no within-refit interval can see.  Comparing a
point estimate across refits is the measurement; the intervals are context.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 42 is the frozen partition.  It is kept first and always run, because a refit
#: sweep that cannot reproduce the committed numbers is measuring its own bugs
#: rather than the pipeline's stability.
REGISTERED_SEEDS = (42, 101, 202, 303)
DEFAULT_SEEDS = REGISTERED_SEEDS

#: Lowered from the pipeline default of 1000; see the module docstring.  The
#: point estimates -- the thing compared across refits -- do not depend on it.
DEFAULT_N_BOOTSTRAP = 200

#: The population the paper leads with.  ``cap_free_valid_plurality``
#: conditions on a difficulty-related outcome, so it is secondary here too.
HEADLINE_POPULATION = "full_population"

#: The strongest peer control on the ladder: both peers, eight samples each.
#: If the geometry increment has a residual over *that*, it has one over every
#: cheaper rung as well.  ``graded`` consults the gold answer, so it bounds the
#: peer family; ``agree`` does not, so it is the rung a reviewer could actually
#: deploy.  Both are tracked because the review treats the peer block as a
#: competing baseline, and demoting the residual on the wrong one of the two
#: would be answering a different objection.
PEER_RESIDUAL_CONTRAST = "B1_minus_B0_graded_both_m8"
PEER_DEPLOYABLE_CONTRAST = "B1_minus_B0_agree_both_m8"


@dataclass(frozen=True)
class ModelSpec:
    """One target model, and the layers each stage reads.

    ``rmd_layer`` is a single layer rather than the collected sweep: the refit
    question is about the headline layer, and scoring three costs three times
    the reference fits for two layers nothing downstream reads.  ``probe_layers``
    stays a sweep because the last-token probe *chooses* among them in fold, and
    removing the choice would remove part of the fitting path being tested.
    """

    label: str
    data_dir: str
    rmd_layer: int
    probe_layers: tuple[int, ...]
    max_new_tokens: int

    @property
    def probe_layer_arg(self) -> str:
        return ",".join(str(layer) for layer in self.probe_layers)


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("qwen", "data/qwen_bestofn_full/math500", 21, (7, 14, 21), 1024),
    ModelSpec("deepseek", "data/deepseek_bestofn_full/math500", 21, (7, 14, 21), 8192),
    ModelSpec(
        "deepseek_llama",
        "data/deepseek_llama_bestofn_full/math500",
        24,
        (8, 16, 24),
        12288,
    ),
)


@dataclass
class Step:
    """One resumable unit of work."""

    name: str
    cmd: list[str]
    marker: Path
    produces: Path
    log: Path
    #: Peak resident memory this step is expected to want, in GB, for the
    #: scheduling note in the plan.  Rough by design; it exists so a reader can
    #: see why the models run one at a time.
    peak_gb: int = 0

    @property
    def done(self) -> bool:
        if not self.marker.exists() or not self.produces.exists():
            return False
        try:
            marker = json.loads(self.marker.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        return marker.get("cmd") == self.cmd


def _python() -> str:
    return sys.executable


def decomposition_step(
    spec: ModelSpec, seed: int, work_dir: Path, *, n_bootstrap: int, load_workers: int
) -> Step:
    """Regenerate the OOF scores under this seed's partition.

    Mirrors the frozen ``evaluate_prompt_decomposition`` stage, minus the parts
    nothing downstream reads: two of the three layers, three of the four
    contrastive regions, and two of the three hidden-probe regions.  What is
    kept is exactly what ``B1`` and the probe join consume.
    """
    out_dir = work_dir / "decomposition"
    cmd = [
        _python(), "-u", "-m", "applications.prompt_decomposition",
        "--data_dir", spec.data_dir,
        "--output_dir", str(out_dir),
        "--dataset_label", "math500",
        "--model_label", spec.label,
        "--layers", str(spec.rmd_layer),
        "--pca_dim", "128",
        "--n", "8",
        "--expected_prompts", "500",
        "--n_splits", "5",
        "--n_bootstrap", str(n_bootstrap),
        "--seed", str(seed),
        "--load_workers", str(load_workers),
        "--contrastive_regions", "tail_q20",
        "--hidden_probe_regions", "tail_q20",
        "--localized_rmd_regions",
        "tail_q10,tail_q20,tail_q50,high_entropy_q20,random_q20",
        "--region_seed", "42",
        "--hidden_dtype", "float16",
        "--compute_dtype", "float32",
        "--max_reference_tokens", "2000000",
        "--max_new_tokens", str(spec.max_new_tokens),
        "--no_progress",
    ]
    return Step(
        name=f"decomposition/{spec.label}",
        cmd=cmd,
        marker=work_dir / ".done_decomposition",
        produces=out_dir / "math500_prompt_decomposition_oof.csv",
        log=work_dir / "decomposition.log",
        peak_gb=140,
    )


def abstention_step(spec: ModelSpec, seed: int, work_dir: Path, *, n_bootstrap: int) -> Step:
    """Refit the prompt-level readouts -- this is where ``B1 - B0`` comes from."""
    oof = work_dir / "decomposition" / "math500_prompt_decomposition_oof.csv"
    out_dir = work_dir / "abstention"
    cmd = [
        _python(), "-u", "-m", "applications.incremental_abstention",
        "--oof_csv", str(oof),
        "--output_dir", str(out_dir),
        "--model_label", spec.label,
        "--dataset_label", "math500",
        "--layer", str(spec.rmd_layer),
        "--data_dir", spec.data_dir,
        "--max_new_tokens", str(spec.max_new_tokens),
        "--expected_traces", "8",
        "--n_bootstrap", str(n_bootstrap),
        "--seed", str(seed),
    ]
    return Step(
        name=f"abstention/{spec.label}",
        cmd=cmd,
        marker=work_dir / ".done_abstention",
        produces=out_dir / "math500_incremental_abstention_results.json",
        log=work_dir / "abstention.log",
        peak_gb=4,
    )


def probe_step(spec: ModelSpec, seed: int, work_dir: Path, *, n_bootstrap: int) -> Step:
    """Refit the published-style probe, folds and in-fold selection included.

    The ``--oof`` join points at *this* seed's decomposition, not the frozen
    one, so ``rmd_tail_q20`` and ``probe_hidden_tail_q20`` move with the refit
    in this table too.  The extraction cache is shared across seeds: it holds
    raw hidden states, which no partition can change.
    """
    oof = work_dir / "decomposition" / "math500_prompt_decomposition_oof.csv"
    out_dir = work_dir / "probe"
    cmd = [
        _python(), "-u", "-m", "controls.last_token_probe",
        "--model", f"{spec.label}:{spec.data_dir}:{spec.probe_layer_arg}",
        "--oof", f"{spec.label}:{oof}:{spec.rmd_layer}",
        "--n_bootstrap", str(n_bootstrap),
        "--seed", str(seed),
        "--output_dir", str(out_dir),
    ]
    return Step(
        name=f"probe/{spec.label}",
        cmd=cmd,
        marker=work_dir / ".done_probe",
        produces=out_dir / "last_token_probe_results.json",
        log=work_dir / "probe.log",
        peak_gb=8,
    )


def peer_step(
    specs: Sequence[ModelSpec], seed: int, seed_dir: Path, *, n_bootstrap: int
) -> Step:
    """Refit the peer ladder across every model at this seed.

    A peer rung is only a control if the target and its peers were fitted on the
    same partition, so this runs once per seed after all models are through.
    """
    out_dir = seed_dir / "peer"
    cmd = [_python(), "-u", "-m", "controls.peer_cost_ladder"]
    for spec in specs:
        oof = seed_dir / spec.label / "decomposition" / "math500_prompt_decomposition_oof.csv"
        cmd += ["--model", f"{spec.label}:{oof}:{spec.data_dir}"]
    cmd += [
        "--population", HEADLINE_POPULATION,
        "--n_bootstrap", str(n_bootstrap),
        "--seed", str(seed),
        "--output_dir", str(out_dir),
    ]
    return Step(
        name="peer",
        cmd=cmd,
        marker=seed_dir / ".done_peer",
        produces=out_dir / "peer_cost_ladder_results.json",
        log=seed_dir / "peer.log",
        peak_gb=8,
    )


def plan_seed(
    seed: int,
    specs: Sequence[ModelSpec],
    work_root: Path,
    *,
    n_bootstrap: int,
    load_workers: int,
    skip_peer: bool = False,
) -> list[Step]:
    """Every step for one refit, in the order it must run."""
    seed_dir = work_root / f"seed_{seed}"
    steps: list[Step] = []
    for spec in specs:
        model_dir = seed_dir / spec.label
        steps.append(
            decomposition_step(
                spec, seed, model_dir, n_bootstrap=n_bootstrap, load_workers=load_workers
            )
        )
        steps.append(abstention_step(spec, seed, model_dir, n_bootstrap=n_bootstrap))
        steps.append(probe_step(spec, seed, model_dir, n_bootstrap=n_bootstrap))
    if not skip_peer and len(specs) >= 2:
        steps.append(peer_step(specs, seed, seed_dir, n_bootstrap=n_bootstrap))
    return steps


def run_step(step: Step, *, dry_run: bool = False) -> bool:
    """Run one step unless it is already done.  Returns True if it ran."""
    if step.done:
        print(f"  [skip] {step.name} (marker present)", flush=True)
        return False
    if dry_run:
        print(f"  [plan] {step.name}\n         {' '.join(step.cmd)}", flush=True)
        return False

    step.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"  [run ] {step.name} -> {step.log}", flush=True)
    with step.log.open("w") as handle:
        completed = subprocess.run(
            step.cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{step.name} exited {completed.returncode} after {elapsed:.0f}s; "
            f"see {step.log}"
        )
    if not step.produces.exists():
        raise RuntimeError(
            f"{step.name} exited 0 but did not write {step.produces}; see {step.log}"
        )
    # The marker carries the command so a later run cannot silently reuse work
    # produced by different arguments.
    step.marker.write_text(
        json.dumps(
            {"cmd": step.cmd, "elapsed_seconds": round(elapsed, 1), "finished": time.time()},
            indent=2,
        )
    )
    print(f"  [done] {step.name} in {elapsed:.0f}s", flush=True)
    return True


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _point(entry: Mapping | None) -> float | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("point_estimate")
    return None if value is None else float(value)


def collect_seed(
    seed: int, specs: Sequence[ModelSpec], work_root: Path
) -> dict:
    """Pull the three tracked quantities out of one refit's artifacts.

    Missing stages come back as ``None`` rather than raising: a sweep stopped
    part way through is still worth summarising, and the report says how many
    refits each row rests on.
    """
    seed_dir = work_root / f"seed_{seed}"
    record: dict = {"seed": seed, "models": {}, "peer": {}}

    for spec in specs:
        model_dir = seed_dir / spec.label
        entry: dict = {}

        abstention = _load(model_dir / "abstention" / "math500_incremental_abstention_results.json")
        if abstention:
            population = abstention["populations"].get(HEADLINE_POPULATION, {})
            deltas = population.get("paired_deltas", {})
            entry["b1_minus_b0_aurc"] = _point(deltas.get("B1_minus_B0_aurc"))
            entry["b1_minus_b0_auacc"] = _point(deltas.get("B1_minus_B0_auacc"))
            entry["n_prompts"] = population.get("n_prompts")

        probe = _load(model_dir / "probe" / "last_token_probe_results.json")
        if probe:
            body = probe["models"][spec.label]["populations"]["parseable"]
            scores = body["scores"]
            probe_row = scores["last_token_probe"]["point"]
            entry["probe_pooled"] = probe_row["pooled_auroc"]
            entry["probe_macro"] = probe_row["macro_prompt_auroc"]
            entry["probe_pooled_minus_macro"] = probe_row["pooled_minus_macro"]
            rmd_row = scores.get("rmd_tail_q20", {}).get("point")
            if rmd_row:
                entry["rmd_pooled"] = rmd_row["pooled_auroc"]
                entry["rmd_pooled_minus_macro"] = rmd_row["pooled_minus_macro"]
            entry["n_mixed_prompts"] = probe_row["n_mixed_prompts"]
            entry["selected_layers"] = sorted(set(body["selected_layers"]))

        record["models"][spec.label] = entry

    peer = _load(seed_dir / "peer" / "peer_cost_ladder_results.json")
    if peer:
        for model in peer["models"]:
            population = model["populations"].get(HEADLINE_POPULATION, {})
            contrasts = population.get("contrasts", {})
            record["peer"][model["label"]] = {
                "residual_aurc": _point(contrasts.get(PEER_RESIDUAL_CONTRAST, {}).get("aurc")),
                "residual_deployable_aurc": _point(
                    contrasts.get(PEER_DEPLOYABLE_CONTRAST, {}).get("aurc")
                ),
                "b1_minus_b0_aurc": _point(contrasts.get("B1_minus_B0", {}).get("aurc")),
            }

    return record


def record_complete(
    record: Mapping, specs: Sequence[ModelSpec], *, require_peer: bool
) -> bool:
    """Whether one seed finished every quantity in the registered sweep."""
    model_quantities = (
        "b1_minus_b0_aurc",
        "probe_pooled_minus_macro",
        "rmd_pooled_minus_macro",
    )
    peer_quantities = ("residual_aurc", "residual_deployable_aurc")
    for spec in specs:
        model = record["models"].get(spec.label, {})
        if any(model.get(name) is None for name in model_quantities):
            return False
        if require_peer:
            peer = record["peer"].get(spec.label, {})
            if any(peer.get(name) is None for name in peer_quantities):
                return False
    return True


def _values(records: Iterable[Mapping], getter) -> list[float]:
    out = []
    for record in records:
        value = getter(record)
        if value is not None:
            out.append(float(value))
    return out


def summarize_quantity(values: Sequence[float], *, frozen: float | None = None) -> dict:
    """Spread and sign agreement across refits.

    Sign agreement is the review's stated decision rule -- "if the residual
    changes sign ... demote it" -- so it is computed rather than left to the
    reader.  ``frozen`` is the seed-42 value when it is in the sweep; the drift
    from it is what a reader of the committed tables actually experiences.
    """
    if not values:
        return {"n": 0}
    signs = {value > 0 for value in values if value != 0}
    summary = {
        "n": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "sign_stable": len(signs) <= 1,
        "n_negative": sum(1 for value in values if value < 0),
        "n_positive": sum(1 for value in values if value > 0),
    }
    if frozen is not None:
        summary["frozen"] = frozen
        summary["max_abs_drift_from_frozen"] = max(abs(value - frozen) for value in values)
    return summary


def summarize(records: Sequence[Mapping], specs: Sequence[ModelSpec]) -> dict:
    """One summary row per (model, quantity)."""
    frozen_record = next((r for r in records if r["seed"] == 42), None)

    def frozen_value(getter):
        return getter(frozen_record) if frozen_record else None

    quantities = {
        "b1_minus_b0_aurc": lambda r, label: r["models"].get(label, {}).get("b1_minus_b0_aurc"),
        "peer_residual_aurc": lambda r, label: r["peer"].get(label, {}).get("residual_aurc"),
        "peer_residual_deployable_aurc": lambda r, label: r["peer"]
        .get(label, {})
        .get("residual_deployable_aurc"),
        "probe_pooled_minus_macro": lambda r, label: r["models"].get(label, {}).get(
            "probe_pooled_minus_macro"
        ),
        "probe_pooled": lambda r, label: r["models"].get(label, {}).get("probe_pooled"),
        "probe_macro": lambda r, label: r["models"].get(label, {}).get("probe_macro"),
        "rmd_pooled_minus_macro": lambda r, label: r["models"].get(label, {}).get(
            "rmd_pooled_minus_macro"
        ),
    }

    out: dict = {}
    for spec in specs:
        model_summary = {}
        for name, getter in quantities.items():
            values = _values(records, lambda r, g=getter, l=spec.label: g(r, l))
            frozen = frozen_value(lambda r, g=getter, l=spec.label: g(r, l))
            model_summary[name] = summarize_quantity(values, frozen=frozen)
        out[spec.label] = model_summary
    return out


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def write_report(body: Mapping, path: Path) -> None:
    records = body["records"]
    summary = body["summary"]
    seeds = [record["seed"] for record in records]

    lines = [
        "# Full-refit stability",
        "",
        "Built by `controls/refit_stability.py`. Each refit re-runs the pipeline "
        "end to end on a different prompt partition: the OOF scores are "
        "regenerated, the prompt-level readouts refitted, the last-token probe "
        "refitted including its in-fold layer and penalty choice, and the peer "
        "ladder refitted across all models at that seed.",
        "",
        f"Complete refits collected: {len(records)} "
        f"(seeds {', '.join(str(s) for s in seeds) or 'none'}). "
        f"Incomplete seeds: {', '.join(str(s) for s in body['incomplete_seeds']) or 'none'}.",
        "Seed 42 is the frozen-partition reproduction check when it appears among "
        "the complete refits.",
        "",
        "The quantity is the **spread of point estimates across refits**. The "
        "bootstrap intervals inside any single refit cannot see it, which is why "
        "the review says more draws are not a substitute.",
        "",
        "## 1. Per-refit values",
        "",
        "| Seed | Model | `B1 - B0` AURC | Peer residual AURC | Probe pooled | Probe macro | Probe pooled - macro | Layers chosen |",
        "|---:|:--|---:|---:|---:|---:|---:|:--|",
    ]
    for record in records:
        for label, entry in record["models"].items():
            peer = record["peer"].get(label, {})
            layers = entry.get("selected_layers") or []
            lines.append(
                f"| {record['seed']} | {label} | "
                f"{_fmt(entry.get('b1_minus_b0_aurc'))} | "
                f"{_fmt(peer.get('residual_aurc'))} | "
                f"{_fmt(entry.get('probe_pooled'))} | "
                f"{_fmt(entry.get('probe_macro'))} | "
                f"{_fmt(entry.get('probe_pooled_minus_macro'))} | "
                f"{', '.join(str(layer) for layer in layers) or '--'} |"
            )

    lines += [
        "",
        "## 2. Stability across refits",
        "",
        "`sign stable` is the review's decision rule: a quantity that changes "
        "sign across refits is demoted regardless of how tight its within-refit "
        "interval is.",
        "",
        "| Model | Quantity | n | Mean | Min | Max | Spread | Sign stable | Max drift from frozen |",
        "|:--|:--|---:|---:|---:|---:|---:|:--|---:|",
    ]
    for label, quantities in summary.items():
        for name, stats in quantities.items():
            if not stats.get("n"):
                lines.append(f"| {label} | `{name}` | 0 | -- | -- | -- | -- | -- | -- |")
                continue
            lines.append(
                f"| {label} | `{name}` | {stats['n']} | "
                f"{_fmt(stats['mean'])} | {_fmt(stats['min'])} | {_fmt(stats['max'])} | "
                f"{_fmt(stats['spread'])} | "
                f"{'yes' if stats['sign_stable'] else '**NO**'} | "
                f"{_fmt(stats.get('max_abs_drift_from_frozen'))} |"
            )

    lines += [
        "",
        "## What this establishes",
        "",
        "A quantity whose spread across refits is comparable to or larger than "
        "its bootstrap interval was being reported with the wrong uncertainty. "
        "A quantity that changes sign across refits does not survive, and the "
        "review's instruction for that case is to keep the original increment "
        "and demote the residual rather than to average the refits.",
        "",
        "The refits share one thing that is not resampled: the collected traces. "
        "This measures stability of the fitting path, not of the data "
        "collection.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=None,
        help=f"repeatable refit seed; default {list(DEFAULT_SEEDS)}",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="repeatable model label to include; default all three",
    )
    parser.add_argument("--work_dir", default="results/refit_stability/work")
    parser.add_argument("--output_dir", default="results/refit_stability")
    parser.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--load_workers", type=int, default=8)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print the full plan and the commands without running anything",
    )
    parser.add_argument(
        "--collect_only",
        action="store_true",
        help="skip execution; summarise whatever artifacts already exist",
    )
    parser.add_argument(
        "--skip_peer",
        action="store_true",
        help="omit the peer ladder (it needs every model at the same seed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    wanted = set(args.model) if args.model else {spec.label for spec in MODEL_SPECS}
    specs = tuple(spec for spec in MODEL_SPECS if spec.label in wanted)
    if not specs:
        raise SystemExit(f"no models matched {sorted(wanted)}")

    work_root = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    if not args.collect_only:
        for seed in seeds:
            print(f"[seed {seed}]", flush=True)
            steps = plan_seed(
                seed,
                specs,
                work_root,
                n_bootstrap=args.n_bootstrap,
                load_workers=args.load_workers,
                skip_peer=args.skip_peer,
            )
            for step in steps:
                run_step(step, dry_run=args.dry_run)

    if args.dry_run:
        print(
            "\nDry run: nothing executed. Models run one at a time by design -- "
            "the decomposition step wants ~140 GB resident on the long-trace "
            "models, and two at once would not fit."
        )
        return

    records = [collect_seed(seed, specs, work_root) for seed in seeds]
    require_peer = not args.skip_peer and len(specs) >= 2
    complete_records = [
        record for record in records
        if record_complete(record, specs, require_peer=require_peer)
    ]
    complete_seeds = [record["seed"] for record in complete_records]
    incomplete_seeds = [record["seed"] for record in records if record not in complete_records]
    protocol_complete = (
        tuple(seeds) == REGISTERED_SEEDS
        and tuple(specs) == MODEL_SPECS
        and require_peer
        and not incomplete_seeds
    )
    body = {
        "seeds": list(seeds),
        "registered_seeds": list(REGISTERED_SEEDS),
        "complete": protocol_complete,
        "complete_seeds": complete_seeds,
        "incomplete_seeds": incomplete_seeds,
        "models": [spec.label for spec in specs],
        "headline_population": HEADLINE_POPULATION,
        "peer_residual_contrast": PEER_RESIDUAL_CONTRAST,
        "n_bootstrap": args.n_bootstrap,
        "records": complete_records,
        "partial_records": [record for record in records if record not in complete_records],
        "summary": summarize(complete_records, specs),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "refit_stability" if protocol_complete else "refit_stability_partial"
    (output_dir / f"{stem}_results.json").write_text(json.dumps(body, indent=2))
    write_report(body, output_dir / f"{stem}_report.md")
    print(f"wrote {output_dir}/{stem}_report.md")

    for label, quantities in body["summary"].items():
        for name, stats in quantities.items():
            if stats.get("n") and not stats["sign_stable"]:
                print(f"  SIGN FLIP: {label} {name} across {stats['n']} refits")


if __name__ == "__main__":
    main()
