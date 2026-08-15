"""Stage B of E2: patch the pairs stage A chose, and read them the registered way.

Stage A decided which items are comparable across depth, from clean forward
passes alone, before any patch existed. This module patches those items and
nothing else. The two halves are separate modules on purpose: `dag_screening`
holds no code that could produce a patched number, and this one holds no code
that could change which items are compared.

Three things here are registered in `EXPERIMENT_LOG.md`, 2026-08-15, and are not
open for adjustment now that stage A has run:

- **Layer 13**, inherited from the `v3_distinct` discovery table and not
  re-searched. Stage B is confirmatory for the depth *contrast*, not the layer.
- **One primary outcome**: the implied digit *uniquely* on top under the ancestor
  patch, depth 1 minus depth 2, with a 1,000-replicate cluster bootstrap. The
  four-way level split is reported beside it and gates nothing; `delta_toward` is
  a log-odds gain from a digit-dependent baseline and is not an outcome at all.
- **A validity gate at 20% null flips.** An arm whose background moves as much as
  its ancestor edit does is an *invalid test*, not a negative one. Every other
  gate in this project is relative, which is how `depth2_chain` passed all of
  them with nulls flipping 23/40.

The one thing stage B cannot do is the fifth row kind. See `ROW_KINDS`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import median

from dag.dag_pooling import _tops, outcomes
from dag.dag_tasks import DEFAULT_GENERATOR, generate_items

# Inherited from the discovery table, not searched here. `dag_pooling.POOLED_LAYER`
# is the same number for the same reason; the test asserts they agree so that a
# change to one cannot silently leave stage B reading a different depth of the
# residual stream than the archived comparison did.
LAYER = 13

# The registered row kinds, minus the one a matched design cannot reach. A
# cross-item batch is *selected* for mutual donatability -- `generate_items`
# oversamples and keeps a subset -- so it is a different batch from a plain run
# at the same seed, and the items stage A screened and matched do not occur in
# one. Running cross-item donors here would mean either matching a fresh batch
# after seeing stage A, or building a donor permutation over the 24 selected
# items; both are design decisions taken after the registration, and the
# cross-item claim is exploratory in it anyway. So it is left out and said so.
ROW_KINDS = ("ancestor", "non_ancestor", "null", "surface_null")
UNREACHABLE_ROW_KINDS = ("cross_item",)
CONTROL_KINDS = ("null", "surface_null", "non_ancestor")

# 20% or more, as registered: the boundary case is inside the gate.
NULL_FLIP_LIMIT = 0.20

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 0

# The clean readout is a rerun of a forward pass stage A already did, on the same
# weights, tokens and precision, so it should return bit-for-bit. The tolerance
# is here to name a drift rather than to allow one.
CLEAN_TOLERANCE = 1e-6


# --------------------------------------------------------------------------
# Which items, and the proof that they are the screened ones
# --------------------------------------------------------------------------


def batches(selection: dict) -> list[dict]:
    """The selected items, grouped by the generation batch they come out of.

    Grouping is what makes the run affordable -- one regeneration per
    `(depth, seed, gap)` rather than one per item -- and it is also where the
    pairing would be lost, since one batch can hold items matched into several
    different pairs. So the pair ordinal is carried on each item rather than
    left implicit in the order.
    """
    grouped: dict[tuple, list] = {}
    for pair, sides in enumerate(selection["selection"]["pairs"]):
        for record in sides:
            key = (record["depth"], record["seed"], record["gap"],
                   record.get("generator", DEFAULT_GENERATOR))
            grouped.setdefault(key, []).append(
                {"index": record["index"], "pair": pair, "record": record})
    return [{"depth": depth, "seed": seed, "gap": gap, "generator": generator,
             "items": sorted(entries, key=lambda entry: entry["index"])}
            for (depth, seed, gap, generator), entries in sorted(grouped.items())]


def regenerated_facts(item) -> dict:
    """What a regenerated `DagItem` claims about itself, as stage A recorded it."""
    return {
        "gap": item.gap,
        "target_value": item.target_value,
        "ancestor_distance": next(edit.distance_to_read for edit in item.edits
                                  if edit.kind == "ancestor"),
    }


def check_regenerated(record: dict, facts: dict) -> None:
    """Refuse an item that is not the one stage A screened.

    Generation is deterministic in `(seed, depth, gap, n_decoys, generator)`,
    and the screening file records the first three but not `n_decoys` -- which
    changes the trace. Rather than trust the flags on the command line, compare
    against what stage A measured. This is the same rule the v0 metadata
    regeneration was accepted under: a regenerated input counts only where it
    reproduces the archived measurement.
    """
    for field in ("gap", "ancestor_distance", "target_value"):
        want, got = record.get(field), facts.get(field)
        if want is not None and want != got:
            raise ValueError(
                f"regenerated item disagrees with the screening record on "
                f"{field}: screened {want}, regenerated {got}. The batch is "
                "not the screened batch -- check n_decoys and the generator.")


def check_clean(record: dict, summary: dict, *,
                tolerance: float = CLEAN_TOLERANCE) -> None:
    """Refuse an item whose clean readout does not reproduce stage A's."""
    target = record.get("target_value", summary.get("target_value"))
    want = record.get("clean_target_share")
    if want is None or target is None:
        return
    got = summary["clean_probs"][target]
    if abs(want - got) > tolerance:
        raise ValueError(
            f"clean readout did not reproduce: screened p(target)={want!r}, "
            f"re-measured {got!r}. Same weights and tokens should give the "
            "same number, so this is a different item.")


# --------------------------------------------------------------------------
# Reading one arm
# --------------------------------------------------------------------------


def _rows_at(report: dict, layer: int) -> list[list[dict]]:
    from dag.dag_patching import unflatten_rows

    if layer not in report.get("layer_bins", []):
        raise ValueError(f"this arm did not measure layer {layer}; it has "
                         f"{report.get('layer_bins')}")
    return [[row for row in block if row["layer"] == layer]
            for block in unflatten_rows(report)]


def flip_rates(report: dict, *, layer: int = LAYER) -> dict[str, dict]:
    """Per row kind, how often the patch takes the clean answer off the top.

    The same statistic `_control_specificity_gate` reports: the clean digit no
    longer holding the maximum alone. Read at one layer, because pooling the
    bins would divide the rate by the number of bins that cannot reach the read
    position at all.
    """
    counts: dict[str, dict] = {}
    for block in _rows_at(report, layer):
        for row in block:
            probs, clean = row.get("probs_patched"), row.get("clean_value")
            if probs is None or clean is None:
                continue
            entry = counts.setdefault(row["kind"], {"flipped": 0, "n": 0})
            entry["n"] += 1
            entry["flipped"] += int(probs[clean] < max(probs))
    for entry in counts.values():
        entry["rate"] = entry["flipped"] / entry["n"] if entry["n"] else None
    return counts


def validity(report: dict, *, layer: int = LAYER,
             limit: float = NULL_FLIP_LIMIT) -> dict:
    """Is this arm a test at all?

    A null edit rewrites a line the target does not depend on. If those move the
    answer as often as the real edit does, the arm has no quiet background to
    read the real edit against, and its comparison is not made. That keeps the
    verdict space three-valued: invalid test, positive, scientific negative.
    """
    rates_by_kind = flip_rates(report, layer=layer)
    null = rates_by_kind.get("null", {"flipped": 0, "n": 0, "rate": None})
    return {
        "rule": f"null edits flipping the answer on {limit:.0%} or more of the "
                "arm's null rows make the arm an invalid test",
        "limit": limit,
        "layer": layer,
        "null": null,
        "controls": {kind: rates_by_kind[kind] for kind in CONTROL_KINDS
                     if kind in rates_by_kind},
        "ancestor": rates_by_kind.get("ancestor"),
        "invalid_test": null["rate"] is not None and null["rate"] >= limit,
    }


def rates(report: dict, *, layer: int = LAYER) -> dict:
    """The primary outcome for one arm: implied uniquely on top, where eligible.

    Eligible is the clean target uniquely on top, which is the denominator the
    corrected pooled table uses. Ties are counted apart rather than resolved by
    digit order.
    """
    _rows_at(report, layer)  # raises if the arm never measured this layer
    tagged = report.get("selected") or []
    items = []
    for record in outcomes(report, layer=layer):
        if record["kind"] != "ancestor" or not record["measured"]:
            continue
        if not record["clean_correct_unique"]:
            continue
        tag = tagged[record["index"]] if record["index"] < len(tagged) else {}
        items.append({
            "pair": tag.get("pair"),
            "seed": tag.get("seed"),
            "index": tag.get("index"),
            "gap": tag.get("gap"),
            "clean_target_share": record["clean_target_share"],
            "implied_top_unique": bool(record["implied_top_unique"]),
            "implied_top_tied": bool(record["implied_top_tied"]),
            "raw_top_unique": bool(record["raw_top_unique"]),
            "clean_top_unique": bool(record["clean_top_unique"]),
        })
    hits = sum(entry["implied_top_unique"] for entry in items)
    return {
        "depth": report.get("depth"),
        "layer": layer,
        "n_measured": sum(1 for record in outcomes(report, layer=layer)
                          if record["kind"] == "ancestor"),
        "n": len(items),
        "hits": hits,
        "rate": hits / len(items) if items else None,
        "tied_implied": sum(entry["implied_top_tied"] for entry in items),
        "items": items,
    }


def level_split(report: dict, *, layer: int = LAYER) -> dict:
    """Where the mass went: median clean and patched share of each named digit.

    Reported, never gated. It exists because "the implied digit wins" is a
    statement about an argmax, and the argmax hides that the transplanted state
    also promotes the donor's *literal* digit -- 0.0005 to 0.373 under a foreign
    donor at depth 1. A reader asking what actually happened to the
    distribution should not have to take the argmax's word for it.
    """
    blocks = _rows_at(report, layer)
    named = {"implied": [], "raw": [], "target": [], "other": []}
    tied = 0
    for block, summary in zip(blocks, report["items"]):
        clean = summary.get("clean_probs")
        target = summary.get("target_value")
        if clean is None or target is None or _tops(clean) != [target]:
            continue
        row = next((row for row in block if row["kind"] == "ancestor"), None)
        patched = row.get("probs_patched") if row else None
        if patched is None:
            continue
        implied, raw = row.get("implied_value"), row.get("raw_value")
        tops = _tops(patched)
        tied += int(tops is not None and len(tops) > 1)
        digits = {"implied": implied, "raw": raw, "target": target}
        for name, digit in digits.items():
            named[name].append((clean[digit], patched[digit])
                               if digit is not None else (None, None))
        claimed = {digit for digit in digits.values() if digit is not None}
        named["other"].append((1.0 - sum(clean[d] for d in claimed),
                               1.0 - sum(patched[d] for d in claimed)))
    split = {"layer": layer, "n": len(named["target"]), "patched_tied": tied}
    for name, values in named.items():
        pairs = [value for value in values if value[0] is not None]
        split[name] = {
            "clean": median(value[0] for value in pairs) if pairs else None,
            "patched": median(value[1] for value in pairs) if pairs else None,
        }
    return split


# --------------------------------------------------------------------------
# The one test
# --------------------------------------------------------------------------


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def bootstrap(by_pair: dict, *, depths: tuple, replicates: int, seed: int) -> dict:
    """Resample the matched pairs, not the items.

    The registered unit is the spine cluster, and stage A's one-item-per-spine
    rule makes each selected item its own spine -- so the cluster and the item
    coincide here, and the only remaining choice is whether the two depths are
    resampled together or apart. Together: the design is matched, and a
    replicate holding one side of a pair without the other is an unmatched
    comparison inside a procedure whose whole point is the matching.
    """
    keys = sorted(pair for pair, sides in by_pair.items()
                  if all(depth in sides for depth in depths))
    if not keys:
        return {"replicates": 0, "interval": None, "n_pairs": 0}
    rng = random.Random(seed)
    differences = []
    for _ in range(replicates):
        draw = [keys[rng.randrange(len(keys))] for _ in keys]
        first, second = (
            [by_pair[pair][depth] for pair in draw] for depth in depths)
        differences.append(sum(first) / len(first) - sum(second) / len(second))
    return {
        "replicates": len(differences),
        "n_pairs": len(keys),
        "interval": [_percentile(differences, 0.025),
                     _percentile(differences, 0.975)],
    }


def primary(reports: dict, *, layer: int = LAYER,
            replicates: int = BOOTSTRAP_REPLICATES,
            seed: int = BOOTSTRAP_SEED, depths: tuple = (1, 2)) -> dict:
    """The registered comparison, and the only confirmatory test in this run."""
    per_depth = {depth: rates(reports[depth], layer=layer) for depth in depths
                 if depth in reports}
    by_pair: dict = {}
    for depth, arm in per_depth.items():
        for entry in arm["items"]:
            if entry["pair"] is None:
                continue
            by_pair.setdefault(entry["pair"], {})[depth] = int(
                entry["implied_top_unique"])
    resampled = bootstrap(by_pair, depths=depths, replicates=replicates,
                          seed=seed)
    rate = {depth: arm["rate"] for depth, arm in per_depth.items()}
    first, second = depths
    difference = (None if rate.get(first) is None or rate.get(second) is None
                  else rate[first] - rate[second])
    return {
        "outcome": "the implied digit uniquely on top under the ancestor patch",
        "layer": layer,
        "depths": list(depths),
        "n": {depth: arm["n"] for depth, arm in per_depth.items()},
        "hits": {depth: arm["hits"] for depth, arm in per_depth.items()},
        "rate": rate,
        "difference": difference,
        "bootstrap_seed": seed,
        "replicates": resampled["replicates"],
        "n_pairs": resampled["n_pairs"],
        "interval": resampled["interval"],
        "per_depth": per_depth,
    }


# --------------------------------------------------------------------------
# The patched run, which is the only part that needs a GPU
# --------------------------------------------------------------------------


def measure(*, model, tokenizer, selection: dict, n_decoys: int,
            model_name: str, condition: str = "both") -> dict:
    """Patch the selected items, one arm per depth, in the archived row schema.

    Written to the same shape `dag_patching.run` writes so that `unflatten_rows`,
    `rescore_report` and `dag_pooling.outcomes` read it unchanged -- the stage-B
    numbers and the archived ones then come off the same code, which is the only
    way the comparison between them means anything.
    """
    from dag.dag_patching import (
        digit_token_ids,
        identity_patch_check,
        layer_bins,
        measure_item,
        readout_dtype,
        rescore_report,
    )

    digit_ids = digit_token_ids(tokenizer)
    bins = layer_bins(model.config.num_hidden_layers)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)

    arms: dict[int, dict] = {}
    identity = None
    for batch in batches(selection):
        wanted = batch["items"]
        items = generate_items(
            encode, n_items=max(entry["index"] for entry in wanted) + 1,
            n_decoys=n_decoys, seed=batch["seed"], condition=condition,
            depth=batch["depth"], gap=batch["gap"],
            generator=batch["generator"], omit="none",
        )
        for entry in wanted:
            item = items[entry["index"]]
            check_regenerated(entry["record"], regenerated_facts(item))
            if identity is None:
                identity = identity_patch_check(
                    model, item.token_ids, bins, list(item.edits[0].positions))
                if not identity["passes"]:
                    raise ValueError(
                        "identity patch changed the logits; read and write "
                        "sites differ")
            rows, summary = measure_item(model, item, bins, digit_ids)
            check_clean(entry["record"], summary)
            arm = arms.setdefault(batch["depth"], {
                "model": model_name, "generator": batch["generator"],
                "readout_dtype": readout_dtype(model), "n_decoys": n_decoys,
                "cross_item": False, "donor_map": None, "condition": condition,
                "depth": batch["depth"], "omit": "none", "seed": None,
                "layer_bins": bins, "n_layers": model.config.num_hidden_layers,
                # The pairs are the unit of this run, so the identities are on
                # the report rather than reconstructed from the seed later.
                "selected": [], "seeds": [], "gap": [], "omitted_nodes": [],
                "n_tokens": [], "ancestor_distance": [],
                "items": [], "rows": [],
                "stage_a": {"row_kinds": list(ROW_KINDS),
                            "unreachable_row_kinds": list(UNREACHABLE_ROW_KINDS)},
            })
            arm["selected"].append({
                "depth": batch["depth"], "seed": batch["seed"],
                "index": entry["index"], "gap": batch["gap"],
                "pair": entry["pair"],
                "clean_target_share": entry["record"].get("clean_target_share"),
                "ancestor_distance": entry["record"].get("ancestor_distance"),
            })
            arm["seeds"].append(batch["seed"])
            arm["gap"].append(item.gap)
            arm["omitted_nodes"].append(list(item.omit))
            arm["n_tokens"].append(len(item.token_ids))
            arm["ancestor_distance"].append(
                regenerated_facts(item)["ancestor_distance"])
            arm["items"].append(summary)
            arm["rows"].extend(rows)

    scored = {}
    for depth, arm in sorted(arms.items()):
        arm["n_items"] = len(arm["items"])
        arm["identity_patch"] = identity
        scored[depth] = rescore_report(arm)
    return scored


def analyse(arms: dict, *, layer: int = LAYER,
            replicates: int = BOOTSTRAP_REPLICATES,
            seed: int = BOOTSTRAP_SEED) -> dict:
    """Every registered reading of the arms, in one payload."""
    depths = tuple(sorted(arms))
    return {
        "layer": layer,
        "row_kinds": list(ROW_KINDS),
        "unreachable_row_kinds": list(UNREACHABLE_ROW_KINDS),
        "validity": {depth: validity(arm, layer=layer)
                     for depth, arm in arms.items()},
        "primary": primary(arms, layer=layer, replicates=replicates,
                           seed=seed, depths=depths),
        "level_split": {depth: level_split(arm, layer=layer)
                        for depth, arm in arms.items()},
        "verdict": {depth: arm.get("verdict") for depth, arm in arms.items()},
        "control_specificity": {
            depth: arm.get("gates", {}).get("control_specificity")
            for depth, arm in arms.items()},
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_name",
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--selection", required=True,
                        help="A stage-A SELECTION.json. Stage B measures the "
                             "pairs in it and nothing else.")
    parser.add_argument("--n_decoys", type=int, default=6,
                        help="Must be what stage A screened with; every item "
                             "is checked against its screened measurement and "
                             "the run refuses a batch that disagrees.")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap_seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def print_summary(analysis: dict) -> None:
    primary_outcome = analysis["primary"]
    print(f"layer          {analysis['layer']}")
    for depth, gate in sorted(analysis["validity"].items()):
        null = gate["null"]
        state = "INVALID TEST" if gate["invalid_test"] else "valid"
        print(f"depth {depth} nulls  {null['flipped']}/{null['n']} flipped "
              f"-> {state}")
    for depth in primary_outcome["depths"]:
        rate = primary_outcome["rate"].get(depth)
        print(f"depth {depth}        implied uniquely top "
              f"{primary_outcome['hits'].get(depth)}/{primary_outcome['n'].get(depth)}"
              f"{'' if rate is None else f'  ({rate:.2f})'}")
    interval = primary_outcome["interval"]
    print(f"difference     {primary_outcome['difference']}"
          f"{'' if interval is None else f'  95% CI [{interval[0]:.3f}, {interval[1]:.3f}]'}"
          f"  over {primary_outcome['n_pairs']} pairs")
    for depth, split in sorted(analysis["level_split"].items()):
        parts = ", ".join(
            f"p({name}) {split[name]['clean']:.4f} -> {split[name]['patched']:.4f}"
            for name in ("implied", "raw", "target", "other")
            if split[name]["clean"] is not None)
        print(f"depth {depth} levels {parts}")


def main() -> None:
    args = parse_args()
    selection = json.loads(Path(args.selection).read_text())
    if not selection["selection"]["proceed"]:
        raise SystemExit(
            "stage A did not reach the registered floor; the registered "
            "outcome is to stop and close the depth claim, not to patch anyway")

    from dag.dag_screening import load_once

    model, tokenizer = load_once(args.model_name)
    arms = measure(model=model, tokenizer=tokenizer, selection=selection,
                   n_decoys=args.n_decoys, model_name=args.model_name)
    analysis = analyse(arms, layer=args.layer, replicates=args.replicates,
                       seed=args.bootstrap_seed)

    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for depth, arm in sorted(arms.items()):
        path = directory / f"depth{depth}.json"
        path.write_text(json.dumps(arm, indent=2))
        print(f"wrote {path}")
    path = directory / "ANALYSIS.json"
    path.write_text(json.dumps({
        "selection": args.selection,
        "n_decoys": args.n_decoys,
        "analysis": analysis,
    }, indent=2))
    print(f"wrote {path}")
    print_summary(analysis)


if __name__ == "__main__":
    main()
