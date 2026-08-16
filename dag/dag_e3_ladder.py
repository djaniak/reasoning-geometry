"""Read the E3 ladder: the depth contrast at the registered N, and the chain arm.

Two questions, one campaign of arms in `results/dag_patching/e3_ladder/`.

* **E3.** Does the depth-versus-token-distance separation survive 48 items and
  three seeds? The gap arms are the distance-matched controls: the depth-2
  ancestor sits 23-36 tokens from the read position and the depth-1 gap-1
  ancestor sits 24-37, so a depth effect that is really a distance effect dies
  here.
* **The chain arm.** At depth 2 and 3 the written intermediates are patched too
  (`dag_patching --chain_edits`). The ancestor edit and the chain edit are the
  same intervention in the same trace against the same clean readout, differing
  only in how many written lines stand between the patched value and the target.
  That makes the contrast paired *within* an item rather than across arms, which
  is the one thing the depth ladder cannot do.

This module holds no torch and loads no model. It reads arm files a patch run
already wrote, the way `dag_pooling` does, so every number here is re-derivable
from the committed artifacts without a GPU.

**The reading is inherited, not chosen here.** `LAYER` and the primary outcome --
the donor-implied digit *alone* on top, among items whose clean answer was alone
on top to begin with -- are stage B's, registered in `EXPERIMENT_LOG.md` on
2026-08-15. Reusing them is what makes these numbers comparable to that run's
instead of merely adjacent to it.

**No gate reads a chain row.** The verdict function is frozen at
`v2_gap_and_floor`; binding a verdict to a reading introduced alongside the arm
that produces it would let the arm decide how it is scored. Arm verdicts are
reported here beside the rates, and `arm_verdicts` exists to say why they are not
the reading: the quorum they use does not scale to this N. See its docstring.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from math import comb
from pathlib import Path

from dag.dag_patching import unflatten_rows
from dag.dag_pooling import _tops

LADDER_DIR = Path("results/dag_patching/e3_ladder")
ANALYSIS_NAME = "ANALYSIS.json"

# Stage B's layer, inherited rather than re-searched, and asserted equal to the
# other two copies in the test suite so that a change to one cannot leave this
# module reading a different depth of the residual stream than the comparison it
# is meant to be comparable to.
LAYER = 13

# The two kinds that sit on the dependency path to the target, and are therefore
# the only ones with a step count. Everything else is a control and is counted by
# the gates, not here.
PATH_KINDS = ("ancestor", "chain")

# Half-open on the right except for the last, which is closed. The split points
# are conventional. What they are for is showing whether sites agree within a
# band of token distance -- which is what a distance explanation predicts -- or
# split by step count inside it, which is what a step explanation predicts.
DISTANCE_BANDS = ((0, 15), (16, 30), (31, 45), (46, 60))


def sign_test(n_a: int, n_b: int) -> float:
    """Two-sided exact p over the discordant pairs of a paired binary contrast.

    The concordant pairs carry no information about the direction of a within-item
    difference, so they are not in the denominator. With no discordant pairs at
    all there is nothing to test and the answer is 1.0 rather than an error: that
    is the depth-3 ancestor-versus-chain-``m`` case, where both sites are dead and
    the honest reading is that nothing separates them.
    """
    n = n_a + n_b
    if not n:
        return 1.0
    k = min(n_a, n_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def records(report: dict, name: str, *, layer: int = LAYER) -> list[dict]:
    """One record per (item, patch site) at ``layer``, for the sites on the path.

    ``eligible`` is the gate every rate below is taken over: the clean answer has
    to have been the model's own answer, alone, or "the patch moved it off the
    clean answer" is not a counterfactual flip. It is a property of the item, so
    it is the same for every site in it -- which is what lets the within-item
    contrast compare two sites without the eligibility differing between them.
    """
    out = []
    for index, (item, block) in enumerate(zip(report["items"],
                                              unflatten_rows(report))):
        clean_tops = _tops(item.get("clean_probs"))
        target = item["target_value"]
        for row in block:
            if row["layer"] != layer or row["kind"] not in PATH_KINDS:
                continue
            tops = _tops(row.get("probs_patched"))
            toward, raw = row.get("delta_toward"), row.get("delta_toward_raw")
            out.append({
                "arm": name,
                "depth": report["depth"],
                "gap": report["gap"][index],
                "seed": report["seed"],
                "item": index,
                "kind": row["kind"],
                "node": row["node"],
                "steps": row.get("steps_to_target"),
                "distance": row["distance_to_read"],
                "eligible": clean_tops == [target],
                "measured": tops is not None,
                "implied_top": tops == [row.get("implied_value")],
                "raw_top": tops == [row.get("raw_value")],
                "clean_top": tops == [target],
                "tv": row["tv"],
                # Which of the two competing movements went further. None where
                # the row stored no raw baseline, which is every surface edit and
                # every report predating the field.
                "toward_over_raw": None if toward is None or raw is None
                else toward > raw,
            })
    return out


def load_records(directory=LADDER_DIR, *, layer: int = LAYER) -> list[dict]:
    paths = sorted(p for p in Path(directory).glob("*.json")
                   if p.name != ANALYSIS_NAME)
    if not paths:
        raise FileNotFoundError(f"no arm files in {directory}")
    return [record for path in paths
            for record in records(json.loads(path.read_text()), path.stem,
                                  layer=layer)]


def _site(record: dict) -> tuple:
    """What makes two records the same patch site: where it lands, not which arm.

    The node name is part of it because depth 3 has two chain lines, and they are
    two steps and one step from the target respectively -- collapsing them would
    average the cell that answers the question with the cell that does not.
    """
    return (record["depth"], record["gap"], record["kind"], record["steps"],
            record["node"] if record["kind"] == "chain" else None)


def _label(site: tuple) -> str:
    depth, gap, kind, _, node = site
    return (f"depth {depth} gap {gap} "
            f"{'ancestor' if kind == 'ancestor' else 'chain ' + str(node)}")


def by_site(records: list[dict]) -> list[dict]:
    """Rates per patch site, over eligible items, pooled across seeds."""
    groups: dict[tuple, list] = defaultdict(list)
    for record in records:
        if record["eligible"]:
            groups[_site(record)].append(record)
    out = []
    for site in sorted(groups, key=lambda s: (s[0], s[1], s[3])):
        rows = groups[site]
        seeds = sorted({row["seed"] for row in rows})
        out.append({
            "label": _label(site),
            "depth": site[0], "gap": site[1], "kind": site[2],
            "steps": site[3], "node": site[4],
            "n": len(rows),
            "distance_min": min(row["distance"] for row in rows),
            "distance_max": max(row["distance"] for row in rows),
            "implied_top_unique": sum(row["implied_top"] for row in rows),
            "raw_top_unique": sum(row["raw_top"] for row in rows),
            "clean_top_unique": sum(row["clean_top"] for row in rows),
            "toward_over_raw": sum(bool(row["toward_over_raw"]) for row in rows),
            "n_toward_comparable": sum(row["toward_over_raw"] is not None
                                       for row in rows),
            "median_tv": statistics.median(row["tv"] for row in rows),
            "per_seed": [{"seed": seed,
                          "n": sum(row["seed"] == seed for row in rows),
                          "implied_top_unique": sum(
                              row["implied_top"] for row in rows
                              if row["seed"] == seed)}
                         for seed in seeds],
        })
    return out


def by_distance_band(records: list[dict],
                     bands=DISTANCE_BANDS) -> list[dict]:
    """The same rates banded by token distance, then split by step count.

    This is the table that separates the two explanations of the depth collapse.
    A band holding both a one-step and a multi-step site is a matched comparison:
    if token distance is what matters its rows agree, and if steps are what
    matters they split.
    """
    eligible = [record for record in records if record["eligible"]]
    out = []
    for low, high in bands:
        inside = [row for row in eligible if low <= row["distance"] <= high]
        for steps in sorted({row["steps"] for row in inside}):
            rows = [row for row in inside if row["steps"] == steps]
            out.append({
                "band": [low, high],
                "steps": steps,
                "n": len(rows),
                "implied_top_unique": sum(row["implied_top"] for row in rows),
                "sites": sorted({_label(_site(row)) for row in rows}),
            })
    return out


def within_item(records: list[dict]) -> list[dict]:
    """The ancestor against each chain line of the same item.

    Paired, so the clean readout, the token count, the null spread and the
    surface control are all held fixed and cannot explain a difference. The
    concordant pairs are reported beside the test rather than folded into it.
    """
    by_item: dict[tuple, dict] = defaultdict(dict)
    for record in records:
        if record["eligible"]:
            by_item[(record["arm"], record["item"])][
                (record["kind"], record["steps"])] = record
    out = []
    depths = sorted({record["depth"] for record in records if record["depth"] > 1})
    for depth in depths:
        for steps in range(1, depth):
            pairs = [(sites[("ancestor", depth)], sites[("chain", steps)])
                     for sites in by_item.values()
                     if ("ancestor", depth) in sites and ("chain", steps) in sites]
            if not pairs:
                continue
            chain_only = sum(c["implied_top"] and not a["implied_top"]
                             for a, c in pairs)
            ancestor_only = sum(a["implied_top"] and not c["implied_top"]
                                for a, c in pairs)
            out.append({
                "depth": depth,
                "ancestor_steps": depth,
                "chain_steps": steps,
                "n": len(pairs),
                "chain_only": chain_only,
                "ancestor_only": ancestor_only,
                "both": sum(a["implied_top"] and c["implied_top"]
                            for a, c in pairs),
                "neither": sum(not a["implied_top"] and not c["implied_top"]
                               for a, c in pairs),
                "p": sign_test(chain_only, ancestor_only),
            })
    return out


def arm_verdicts(directory=LADDER_DIR) -> list[dict]:
    """Each arm's own verdict, and the quorum that decided it.

    Reported, and deliberately not the reading. `dag_patching._quorum` is
    ``max(1, n - 1)`` -- "all but one" -- which asks for 80% of items at the n=5
    the ladder was designed at and 47/48, or 97.9%, here. The surface control's
    real pass rate is 85-100% in every arm of this campaign, so which side of the
    line an arm falls on turns on one or two items: `depth1_gap0` is an invalid
    test at all three seeds while its ancestor gap is 48/48.

    The rule is not changed. Rewriting a quorum after seeing which arms it fails
    is a retroactive policy move, and it would be made on evidence produced by the
    run being scored. What is owed instead is a pre-registered decision on the
    quorum as a function of N, written before it is applied to anything -- see
    `EXPERIMENT_LOG.md`, 2026-08-16. Until then the per-layer counts below are
    what a reader should use, not the verdict beside them.
    """
    out = []
    for path in sorted(p for p in Path(directory).glob("*.json")
                       if p.name != ANALYSIS_NAME):
        report = json.loads(path.read_text())
        active = report["scoring"][report["gate_policy_version"]]
        scored = report["gates"]["scoring_layers"]
        per_layer = report["gates"]["surface_v2_one_sided"]["per_layer"]
        counts = [per_layer[str(layer)]["surface_items"] for layer in scored]
        out.append({
            "arm": path.stem,
            "n_items": report["n_items"],
            "verdict": report["verdict"],
            "invalid_reasons": active["invalid_reasons"],
            # The rule the verdict was decided by, recorded beside the counts so
            # the 80%-versus-97.9% point is checkable and not merely asserted.
            "quorum": max(1, report["n_items"] - 1),
            "surface_by_layer": dict(zip(map(str, scored), counts)),
            "surface_best": max(counts),
        })
    return out


def build(directory=LADDER_DIR, *, layer: int = LAYER) -> dict:
    everything = load_records(directory, layer=layer)
    eligible = [record for record in everything if record["eligible"]]
    return {
        "layer": layer,
        "reading": "the donor-implied digit alone on top, among items whose "
                   "clean answer was alone on top",
        "path_kinds": list(PATH_KINDS),
        "chain_rows_gate_nothing": True,
        "n_sites": len(everything),
        "n_eligible": len(eligible),
        "n_items": len({(record["arm"], record["item"])
                        for record in everything}),
        "by_site": by_site(everything),
        "by_distance_band": by_distance_band(everything),
        "within_item": within_item(everything),
        "arm_verdicts": arm_verdicts(directory),
        "records": everything,
    }


def _rate(numerator: int, denominator: int) -> str:
    return (f"{numerator:4d}/{denominator:<4d} ({100 * numerator / denominator:5.1f}%)"
            if denominator else "-")


def print_summary(built: dict) -> None:
    print(f"{built['n_sites']} patch sites over {built['n_items']} items; "
          f"{built['n_eligible']} on items whose clean answer was alone on top "
          f"({100 * built['n_eligible'] / built['n_sites']:.0f}%)")
    print(f"layer {built['layer']}: {built['reading']}")

    print(f"\n{'arm':24s} {'verdict':20s} {'surface / scoring layer':28s}"
          f"{'need':>5s}  why invalid")
    print("-" * 108)
    for arm in built["arm_verdicts"]:
        counts = "  ".join(f"L{layer}={n}" for layer, n
                           in arm["surface_by_layer"].items())
        counts = f"{counts} of {arm['n_items']}"
        print(f"{arm['arm']:24s} {arm['verdict']:20s} {counts:28s}"
              f"{arm['quorum']:>5d}  {', '.join(arm['invalid_reasons']) or '-'}")
    print("  the quorum is `max(1, n-1)`: 80% of items at n=5, 97.9% at n=48. "
          "Reported, not applied here.")

    print(f"\n{'site':30s} {'steps':>5s} {'tokens':>9s} {'implied':>18s} "
          f"{'clean held':>18s} {'medTV':>6s}   per seed")
    print("-" * 118)
    for row in built["by_site"]:
        seeds = "  ".join(f"s{s['seed']}:{s['implied_top_unique']}/{s['n']}"
                          for s in row["per_seed"])
        print(f"{row['label']:30s} {row['steps']:>5d} "
              f"{row['distance_min']:>4d}-{row['distance_max']:<4d} "
              f"{_rate(row['implied_top_unique'], row['n']):>18s} "
              f"{_rate(row['clean_top_unique'], row['n']):>18s} "
              f"{row['median_tv']:>6.3f}   {seeds}")

    print("\nSteps or tokens? If distance is what matters the rows of a band "
          "agree; if steps are,\nthey split along `steps` instead.\n")
    print(f"{'distance band':16s} {'steps':>5s} {'n':>5s} {'implied':>18s}   sites")
    print("-" * 96)
    for row in built["by_distance_band"]:
        low, high = row["band"]
        print(f"{f'{low}-{high} tokens':16s} {row['steps']:>5d} {row['n']:>5d} "
              f"{_rate(row['implied_top_unique'], row['n']):>18s}   "
              f"{', '.join(row['sites'])}")

    print("\nWithin-item: the ancestor and a chain line, same trace, same clean "
          "readout.\nSign test over the discordant pairs.\n")
    for row in built["within_item"]:
        print(f"depth {row['depth']}: ancestor ({row['ancestor_steps']} steps) "
              f"vs chain ({row['chain_steps']} "
              f"step{'s' if row['chain_steps'] > 1 else ''}), n={row['n']}")
        print(f"    chain only {row['chain_only']:4d} | ancestor only "
              f"{row['ancestor_only']:4d} | both {row['both']:4d} | "
              f"neither {row['neither']:4d} | p={row['p']:.2e}")

    print("\nCarried through the remaining written steps, or copied off the "
          "patched position?\n")
    for row in built["by_site"]:
        if not row["n_toward_comparable"]:
            continue
        print(f"{row['label']:30s} implied "
              f"{_rate(row['implied_top_unique'], row['n'])}  raw "
              f"{_rate(row['raw_top_unique'], row['n'])}  toward>raw "
              f"{row['toward_over_raw']:4d}/{row['n_toward_comparable']:<4d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(LADDER_DIR))
    parser.add_argument("--layer", type=int, default=LAYER,
                        help="inherited from stage B; changing it makes these "
                             "numbers incomparable to that run's")
    parser.add_argument("--output", default=None,
                        help=f"where to write the analysis (default: "
                             f"<dir>/{ANALYSIS_NAME}); '-' writes nothing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    built = build(args.dir, layer=args.layer)
    print_summary(built)
    if args.output != "-":
        path = Path(args.output or Path(args.dir) / ANALYSIS_NAME)
        path.write_text(json.dumps(built, indent=2) + "\n")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
