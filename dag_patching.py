"""Five-item residual-stream patching feasibility check on ground-truth DAGs.

Prototype 1 of the causal-DAG fidelity direction. For each item we run the clean
trace, run each donor trace, then re-run the clean trace with the donor's
residual state written in at the edited token positions. We read the model's
ten-way digit distribution at the position that predicts the target's result.

The question this script answers is *not* "does the model have a causal graph".
It is the prior question: **is this intervention selective enough to support that
experiment at all**. It therefore reports three separate verdicts — positive,
scientific negative, and invalid test — and never reports "no causal graph" when
the intervention itself failed.

Layer indexing note: unlike ``collect_data.py``, which reads
``outputs.hidden_states[i]``, this script both reads and writes at the output of
``model.model.layers[j]``. Using one mechanism for both directions makes the
off-by-one between the two conventions impossible rather than merely tested.
``hidden_states[n_layers]`` is also post-final-norm, so it is not a valid patch
site at all.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from dag_tasks import CHECKPOINTS, DONOR_CONDITIONS, generate_items

LAYER_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
DIGITS = tuple(str(d) for d in range(10))

# Surface-control policies. See ``_surface_gate`` for why there are two.
GATE_POLICIES = ("v1_two_sided", "v2_one_sided")
ACTIVE_GATE_POLICY = "v2_one_sided"


def layer_bins(n_layers: int, fractions=LAYER_FRACTIONS) -> list[int]:
    """Four relative-depth decoder-layer indices, fixed once per checkpoint.

    Frozen before the first run and reused for every checkpoint and prototype,
    per the pre-registered measurement conditions.
    """
    return sorted({max(0, round(fraction * n_layers) - 1) for fraction in fractions})


def decoder_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError(f"{type(model).__name__} has no model.layers to hook")
    return layers


def _residual(output):
    return output[0] if isinstance(output, tuple) else output


def _rewrap(output, residual):
    if isinstance(output, tuple):
        return (residual,) + tuple(output[1:])
    return residual


def capture_states(model, token_ids, bins, positions):
    """Residual states at ``positions``, one tensor per layer bin, plus logits."""
    import torch

    store = {}
    handles = []
    for layer in bins:
        def hook(module, args, output, layer=layer):
            store[layer] = _residual(output)[:, positions, :].detach().clone()
        handles.append(decoder_layers(model)[layer].register_forward_hook(hook))
    try:
        with torch.no_grad():
            input_ids = torch.as_tensor([list(token_ids)], device=model.device)
            logits = model(input_ids, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    return store, logits


def run_patched(model, token_ids, layer, positions, donor):
    """Run ``token_ids`` with ``donor`` written into layer ``layer`` at ``positions``."""
    import torch

    def hook(module, args, output):
        residual = _residual(output).clone()
        residual[:, positions, :] = donor.to(residual.device, residual.dtype)
        return _rewrap(output, residual)

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            input_ids = torch.as_tensor([list(token_ids)], device=model.device)
            return model(input_ids, use_cache=False).logits
    finally:
        handle.remove()


def digit_readout(logits, read_position, digit_ids):
    """(probabilities over the ten digits, log-odds, total digit mass)."""
    import torch

    row = logits[0, read_position].detach().float()
    selected = row[list(digit_ids)]
    full = torch.softmax(row, dim=-1)
    return (
        torch.softmax(selected, dim=-1).cpu().numpy(),
        torch.log_softmax(selected, dim=-1).cpu().numpy(),
        float(full[list(digit_ids)].sum()),
    )


def digit_token_ids(tokenizer) -> list[int]:
    ids = []
    for digit in DIGITS:
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"digit {digit!r} is {len(encoded)} tokens, not 1")
        ids.append(encoded[0])
    if len(set(ids)) != 10:
        raise ValueError("digit token ids are not distinct")
    return ids


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def measure_item(model, item, bins, digit_ids) -> list[dict]:
    """One row per (layer bin, edit)."""
    _, clean_logits = capture_states(model, item.token_ids, bins, [item.read_position])
    probs, logodds, mass = digit_readout(clean_logits, item.read_position, digit_ids)

    rows = []
    for edit in item.edits:
        if max(edit.positions) >= item.read_position:
            raise ValueError("edit is not upstream of the read position")
        if len(edit.token_ids) != len(item.token_ids):
            raise ValueError("donor and clean traces differ in length")
        donor_states, _ = capture_states(
            model, edit.token_ids, bins, list(edit.positions)
        )
        for layer in bins:
            patched_logits = run_patched(
                model, item.token_ids, layer, list(edit.positions), donor_states[layer]
            )
            p_probs, p_logodds, p_mass = digit_readout(
                patched_logits, item.read_position, digit_ids
            )
            rows.append({
                "kind": edit.kind,
                "node": edit.node,
                "layer": layer,
                "distance_to_read": edit.distance_to_read,
                "tv": float(0.5 * abs(p_probs - probs).sum()),
                "delta_toward": float(
                    p_logodds[edit.implied_target_value]
                    - logodds[edit.implied_target_value]
                ),
                "delta_away": float(
                    p_logodds[item.target_value] - logodds[item.target_value]
                ),
                "digit_mass_clean": mass,
                "digit_mass_patched": p_mass,
            })
    return rows, {
        "target_value": item.target_value,
        "clean_top_digit": int(probs.argmax()),
        "clean_target_logodds": float(logodds[item.target_value]),
        "clean_digit_mass": mass,
    }


def scoring_layers(bins: list[int], n_layers: int | None) -> list[int]:
    """The layer bins a gate may be decided on.

    Patching the last decoder layer at a position upstream of the read position
    cannot reach that position: there is no later layer to carry it. Every TV
    there is 0 by construction, so a containment gate passes trivially
    (``0 <= 0 <= 0``) and an ``any(layer)`` rule would let the inert bin rescue a
    gate that fails at every informative layer. The prose summary already
    excluded it; this makes the scorer agree.

    ``n_layers`` is what identifies the model's final layer. Without it the
    caller has not said which bin that is, so every bin scores.
    """
    if n_layers is None:
        return list(bins)
    return [layer for layer in bins if layer != n_layers - 1]


def _surface_gate(passes_by_layer, bins, policy, n_by_layer, scored) -> dict:
    """Aggregate one surface policy with the same 4/5 rule as the other gates.

    ``v1_two_sided`` is the rule as originally registered: the surface
    perturbation must fall *inside* the range spanned by the null perturbations.
    That is a distributional-matching test, and the surface control was intended
    to test one-sided non-interference -- a computationally irrelevant edit must
    not move the readout *more* than an irrelevant value edit does.

    ``v2_one_sided`` states that role directly and is the active policy. It is a
    post-hoc amendment, so both policies stay runnable and both are reported.
    Passing v2 establishes only that the tag edit is quiet; it does not establish
    selectivity. The tag edit is a floor check, not a matched control -- the
    cross-item donor control is what would test selectivity.
    """
    per_layer = {
        layer: {
            "surface_items": sum(ok for lay, ok in passes_by_layer if lay == layer),
            "n_items": n_by_layer[layer],
        }
        for layer in bins
    }
    return {
        "policy": policy,
        "per_layer": per_layer,
        "failure_reason": (
            "surface_above_null" if policy == "v2_one_sided"
            else "surface_outside_null_spread"
        ),
        "passes": any(
            per_layer[layer]["surface_items"] >= max(1, per_layer[layer]["n_items"] - 1)
            for layer in scored
        ),
    }


def evaluate_gates(per_item: list[list[dict]], bins: list[int],
                   policy: str = ACTIVE_GATE_POLICY,
                   n_layers: int | None = None) -> dict:
    """The four continue-rules from the prototype specification.

    ``policy`` selects which surface rule the verdict is bound to. Both are
    always computed and reported; only the active one gates the verdict.
    """
    if policy not in GATE_POLICIES:
        raise ValueError(f"unknown gate policy {policy!r}, expected one of "
                         f"{GATE_POLICIES}")
    scored = scoring_layers(bins, n_layers)
    if not scored:
        raise ValueError(f"no scoring layers left in {bins} for {n_layers} layers")
    gates = {"gate_policy_version": policy, "scoring_layers": scored}

    # 1. Directional positive control. An ancestor patch must move the target
    #    toward the value implied by the donor, not merely away from the clean
    #    answer -- a patch that only corrupts the model also does the latter.
    directional = {}
    for layer in bins:
        toward = [
            row["delta_toward"]
            for rows in per_item
            for row in rows
            if row["kind"] == "ancestor" and row["layer"] == layer
        ]
        directional[layer] = {
            "n_positive": sum(value > 0 for value in toward),
            "n_items": len(toward),
            "median_delta_toward": statistics.median(toward) if toward else 0.0,
        }
    gates["directional_control"] = {
        "per_layer": directional,
        "passes": any(
            directional[layer]["n_positive"] >= max(1, directional[layer]["n_items"] - 1)
            and directional[layer]["median_delta_toward"] > 0
            for layer in scored
        ),
    }

    # 2. Fluency. The answer distribution must stay on digits at all.
    ratios = [
        row["digit_mass_patched"] / row["digit_mass_clean"]
        for rows in per_item
        for row in rows
        if row["kind"] == "ancestor" and row["digit_mass_clean"] > 0
    ]
    gates["fluency"] = {
        "min_digit_mass_ratio": min(ratios) if ratios else 0.0,
        "passes": bool(ratios) and min(ratios) >= 0.5,
    }

    # 3. The ancestor-minus-non-ancestor gap must exceed the per-item null spread.
    # 4. The surface edit must not interfere; see ``_surface_gate`` for the two
    #    policies and why v2 is active.
    gap_pass, detail = [], []
    surface_pass = {name: [] for name in GATE_POLICIES}
    for rows in per_item:
        for layer in bins:
            at_layer = [row for row in rows if row["layer"] == layer]
            nulls = [row["tv"] for row in at_layer if row["kind"] == "null"]
            ancestor = next(r["tv"] for r in at_layer if r["kind"] == "ancestor")
            non_ancestor = next(r["tv"] for r in at_layer if r["kind"] == "non_ancestor")
            surface = next(r["tv"] for r in at_layer if r["kind"] == "surface_null")
            spread = max(nulls) - min(nulls)
            gap_pass.append((layer, (ancestor - non_ancestor) > spread))
            surface_pass["v1_two_sided"].append(
                (layer, min(nulls) <= surface <= max(nulls)))
            surface_pass["v2_one_sided"].append((layer, surface <= max(nulls)))
            detail.append({
                "layer": layer,
                "tv_ancestor": ancestor,
                "tv_non_ancestor": non_ancestor,
                "tv_surface_null": surface,
                "tv_null_min": min(nulls),
                "tv_null_max": max(nulls),
                "ancestor_above_all_nulls": ancestor > max(nulls),
                # Which side a v1 failure fell on. Every archived v1 failure but
                # one is "below", which is the direction the control wants.
                "surface_side": (
                    "below" if surface < min(nulls)
                    else "above" if surface > max(nulls)
                    else "in"
                ),
            })
    n_by_layer = {layer: sum(lay == layer for lay, _ in gap_pass) for layer in bins}
    gates["ancestor_gap"] = {
        "per_layer": {
            layer: {
                "gap_items": sum(ok for lay, ok in gap_pass if lay == layer),
                "n_items": n_by_layer[layer],
            }
            for layer in bins
        },
        "passes": any(
            sum(ok for lay, ok in gap_pass if lay == layer)
            >= max(1, n_by_layer[layer] - 1)
            for layer in scored
        ),
    }
    for name in GATE_POLICIES:
        gates[f"surface_{name}"] = _surface_gate(
            surface_pass[name], bins, name, n_by_layer, scored
        )
    gates["surface_active"] = gates[f"surface_{policy}"]
    gates["detail"] = detail
    return gates


def invalid_reasons(gates: dict) -> list[str]:
    """Why the intervention itself failed, if it did. Empty means it held.

    Directional control, fluency, and the active surface gate are validity
    requirements: they say whether the patch measured anything. Ancestor
    separation is only meaningful once all three hold, so it is not consulted
    here.
    """
    reasons = []
    if not gates["directional_control"]["passes"]:
        reasons.append("directional_control_failed")
    if not gates["fluency"]["passes"]:
        reasons.append("digit_mass_collapsed")
    if not gates["surface_active"]["passes"]:
        reasons.append(gates["surface_active"]["failure_reason"])
    return reasons


def verdict(gates: dict) -> str:
    """Positive, scientific negative, or invalid test -- kept strictly separate.

    An invalid test is not evidence about the model. Reporting one as a null is
    the specific failure the feasibility stage exists to catch. A loud surface
    edit belongs in that bucket: it means any two-token perturbation moves the
    readout, so the ancestor gap is not attributable to the edge.
    """
    if invalid_reasons(gates):
        return "invalid test"
    if gates["ancestor_gap"]["passes"]:
        return "positive"
    return "scientific negative"


# --------------------------------------------------------------------------
# offline rescoring
#
# The rows are the measurement; the gates and the verdict are a policy over
# them. Keeping the two separable is what lets a gate revision be re-run on an
# archived report instead of costing a GPU replay.
# --------------------------------------------------------------------------


def unflatten_rows(report: dict) -> list[list[dict]]:
    """Recover per-item row blocks from a stored report's flat ``rows``.

    ``measure_item`` emits ``for edit: for layer:``, so one item is a run of
    edit groups, each group one row per layer bin in ``layer_bins`` order. This
    validates that layout rather than assuming it: a silently regrouped file
    would mix two items into one block and produce a plausible-looking verdict.
    """
    rows = report["rows"]
    n_items = report["n_items"]
    bins = list(report["layer_bins"])
    if n_items <= 0 or len(rows) % n_items:
        raise ValueError(
            f"row count {len(rows)} does not divide into {n_items} items"
        )
    per_item = len(rows) // n_items
    if not bins or per_item % len(bins):
        raise ValueError(
            f"row count {per_item} per item is not a multiple of "
            f"{len(bins)} layer bins"
        )
    blocks = [rows[i * per_item:(i + 1) * per_item] for i in range(n_items)]
    for index, block in enumerate(blocks):
        for start in range(0, per_item, len(bins)):
            group = block[start:start + len(bins)]
            if [row["layer"] for row in group] != bins:
                raise ValueError(
                    f"item {index}: unexpected layer order "
                    f"{[row['layer'] for row in group]}, expected {bins}"
                )
            if len({row["kind"] for row in group}) != 1:
                raise ValueError(
                    f"item {index}: edit block does not have a single kind, got "
                    f"{sorted({row['kind'] for row in group})}"
                )
    return blocks


def rescore_report(report: dict, *, active: str = ACTIVE_GATE_POLICY) -> dict:
    """Re-run every gate policy over a stored report's rows.

    Returns a new report; the input and its rows are left untouched. Both
    policies' gates and verdicts are recorded so the legacy numbers stay
    auditable against the same measurement.
    """
    per_item = unflatten_rows(report)
    bins = list(report["layer_bins"])
    scoring = {}
    for policy in GATE_POLICIES:
        gates = evaluate_gates(per_item, bins, policy=policy,
                               n_layers=report.get("n_layers"))
        scoring[policy] = {
            "gates": gates,
            "verdict": verdict(gates),
            "invalid_reasons": invalid_reasons(gates),
        }
    rescored = dict(report)
    if "verdict" in report:
        rescored["original_verdict"] = report["verdict"]
    rescored["gate_policy_version"] = active
    rescored["gates"] = scoring[active]["gates"]
    rescored["verdict"] = scoring[active]["verdict"]
    rescored["scoring"] = scoring
    return rescored


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def identity_patch_check(model, token_ids, bins, positions) -> dict:
    """Patch the clean run's own state back into itself; logits must not move.

    This is the one cheap check that pins the read and write sites to the same
    place. A silent mismatch there produces a plausible-looking null.
    """
    import torch

    states, clean_logits = capture_states(model, token_ids, bins, positions)
    worst = 0.0
    for layer in bins:
        patched = run_patched(model, token_ids, layer, positions, states[layer])
        worst = max(worst, float((patched - clean_logits).abs().max()))
    return {"max_abs_logit_change": worst, "passes": worst < 1e-3}


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def run(*, model_name: str, n_items: int, n_decoys: int, seed: int,
        output_path: str | None, self_test_only: bool,
        condition: str = "both", depth: int = 1, gap: int | None = None) -> dict:
    from transformers import AutoTokenizer

    from collect_data import load_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    items = generate_items(
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        n_items=n_items,
        n_decoys=n_decoys,
        seed=seed,
        condition=condition,
        depth=depth,
        gap=gap,
    )
    digit_ids = digit_token_ids(tokenizer)

    model, _ = load_model(False, model_name=model_name)
    bins = layer_bins(model.config.num_hidden_layers)

    identity = identity_patch_check(
        model, items[0].token_ids, bins, list(items[0].edits[0].positions)
    )
    report = {
        "model": model_name,
        "condition": condition,
        "depth": depth,
        "gap": [item.gap for item in items],
        "n_items": len(items),
        "seed": seed,
        "layer_bins": bins,
        "n_layers": model.config.num_hidden_layers,
        "n_tokens": [len(item.token_ids) for item in items],
        # Depth arms are matched against gap arms on this, not on the knobs.
        "ancestor_distance": [
            next(e.distance_to_read for e in item.edits if e.kind == "ancestor")
            for item in items
        ],
        "identity_patch": identity,
    }
    if not identity["passes"]:
        report["verdict"] = "invalid test"
        report["reason"] = "identity patch changed the logits; read and write sites differ"
        return report
    if self_test_only:
        report["verdict"] = "self test only"
        return report

    per_item, summaries = [], []
    for item in items:
        rows, summary = measure_item(model, item, bins, digit_ids)
        per_item.append(rows)
        summaries.append(summary)

    report["items"] = summaries
    report["rows"] = [row for rows in per_item for row in rows]
    # Score through the same path an archived report takes, so a fresh run and a
    # rescored one cannot disagree, and the row layout is validated on the way.
    report = rescore_report(report)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
    return report


def print_gate_table(report: dict) -> None:
    gates = report.get("gates")
    print(f"model          {report['model']}")
    # depth / gap / ancestor_distance postdate the first three artifacts. Show
    # them as unrecorded rather than inventing a default; `dag_evidence.py`
    # derives the true values by regenerating the items.
    print(f"condition      {report['condition']}  "
          f"depth {report.get('depth', '(v0: unrecorded)')}  "
          f"gap {report.get('gap', '(v0: unrecorded)')}")
    print(f"ancestor dist  {report.get('ancestor_distance', '(v0: unrecorded)')} "
          f"tokens to the read position")
    print(f"layer bins     {report['layer_bins']} of {report['n_layers']} layers")
    print(f"identity patch max |dlogit| = "
          f"{report['identity_patch']['max_abs_logit_change']:.2e} "
          f"({'pass' if report['identity_patch']['passes'] else 'FAIL'})")
    if not gates:
        print(f"verdict        {report['verdict']}")
        return
    print()
    print(f"gate policy    {report.get('gate_policy_version', ACTIVE_GATE_POLICY)}")
    print(f"{'gate':<32}{'passes':<10}detail")
    for name in ("directional_control", "fluency", "ancestor_gap",
                 *(f"surface_{policy}" for policy in GATE_POLICIES)):
        gate = gates[name]
        detail = {k: v for k, v in gate.items() if k != "passes"}
        print(f"{name:<32}{str(gate['passes']):<10}{json.dumps(detail)[:110]}")
    print()
    for policy, scored in report.get("scoring", {}).items():
        mark = " (active)" if policy == report.get("gate_policy_version") else ""
        reasons = ", ".join(scored["invalid_reasons"]) or "-"
        print(f"verdict[{policy}]{mark:<9} {scored['verdict']:<20} {reasons}")
    if "scoring" not in report:
        print(f"verdict        {report['verdict']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=CHECKPOINTS[2])
    parser.add_argument("--n_items", type=int, default=5)
    parser.add_argument("--n_decoys", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", choices=DONOR_CONDITIONS, default="both",
                        help="which part of the edited line the donor rewrites")
    parser.add_argument("--depth", type=int, default=1,
                        help="steps from the edited ancestor to the target")
    parser.add_argument("--gap", type=int, default=None,
                        help="decoy lines between the chain and the target; the "
                             "distance control for --depth")
    parser.add_argument("--output", default=None)
    parser.add_argument("--self_test", action="store_true",
                        help="run the identity patch and stop, no science")
    parser.add_argument("--rescore", default=None, metavar="REPORT_JSON",
                        help="re-run every gate policy over a stored report's "
                             "rows and stop; no model is loaded")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rescore:
        report = rescore_report(json.loads(Path(args.rescore).read_text()))
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2))
        print_gate_table(report)
        return
    report = run(
        model_name=args.model_name,
        n_items=args.n_items,
        n_decoys=args.n_decoys,
        seed=args.seed,
        condition=args.condition,
        depth=args.depth,
        gap=args.gap,
        output_path=args.output,
        self_test_only=args.self_test,
    )
    print_gate_table(report)


if __name__ == "__main__":
    main()
