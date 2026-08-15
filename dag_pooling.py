"""Pool the per-item patching outcomes across the DAG arms.

Every arm holds five items, and every arm README reports its own five. The
strongest count written down anywhere in this repository is therefore ``5/5``,
which makes the depth-1 result look like a five-item observation when the
committed runs already hold thirty-three of them, over four seeds and three gap
placements. This module pools them and reports the one outcome a control edit
has no reason to imitate -- whether the argmax of the patched digit readout is
the digit the donor's value *implies* -- split by the confidence the model had
in its clean answer before the patch.

Three things it deliberately does not do.

It does not rescore. Nothing here feeds a gate, changes a verdict, or touches a
stored report; ``dag_patching`` owns the verdict and this is a reading of the
measurements underneath it.

It does not count files. The same item is stored in up to four arms -- the
written-versus-omitted depth-1 pair is one measurement recorded twice, and the
cross-item runs re-store the paired rows alongside the transplanted ones -- so
measurements are identified by their contents and pooled once. Deduplication
that mattered is visible: every record carries the arms it was found in.

And it does not make the result held-out. Layer 13 was selected from these same
runs. Pooling buys precision on an effect already seen, not independent
confirmation of it, and a held-out family is still the experiment to run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dag_patching import unflatten_rows

ARTIFACT_DIR = Path("results/dag_patching")

# The layer the v3_distinct table settled on. Fixed here rather than searched:
# picking the best layer per arm is what made the earlier cross-item count look
# twice as good as it was.
POOLED_LAYER = 13

# The archived eight predate `probs_patched` and come from a family whose three
# competing digits could coincide, which is the whole point of v3_distinct. They
# are provenance, not evidence about which digit won.
POOLED_GENERATOR = "v3_distinct"

# Two donors, two claims. `ancestor` transplants the item's own paired state and
# asks whether one downstream operation reads it. `cross_item` transplants a
# state from a different item entirely and asks whether the value is portable.
DONOR_KINDS = ("ancestor", "cross_item")

# Half-open, lower bound inclusive, with the top band closed at 1.0. The split
# points are conventional; what they are for is showing whether the implied-hit
# rate decays as the model grows confident, which is the confound that a
# floor-based gate cannot see.
CONFIDENCE_BANDS = ((0.0, 0.5), (0.5, 0.8), (0.8, 1.0))


def _measurement_id(item: dict, kind: str, rows: list[dict]) -> tuple:
    """What makes two records the same measurement.

    Not the arm, the seed, or the file: which intervention ran, and the numbers
    it produced. Two records whose clean readout and whose patched rows agree to
    float precision came out of the same forward passes, and counting them twice
    would inflate the only result the project still stands on. Fifty-odd float32
    values agreeing by coincidence is not a case worth designing around.

    The donor kind is part of the identity rather than part of the numbers. Two
    different interventions on one item are two measurements even where they
    happen to land in the same place, which is exactly the case a reader would
    want counted twice.
    """
    return (
        kind,
        tuple(item.get("clean_probs") or ()),
        item.get("target_value"),
        tuple((row["layer"], row["tv"], row.get("delta_toward"),
               row.get("implied_value")) for row in rows),
    )


def measurement_id(report: dict, index: int, kind: str) -> tuple:
    """The identity of one item's rows of one donor kind, as stored."""
    rows = [row for row in unflatten_rows(report)[index] if row["kind"] == kind]
    return _measurement_id(report["items"][index], kind, rows)


# Omission is a grouping dimension, not a detail. The chain-omitted arms hit the
# pre-registered stop condition -- clean accuracy collapsed alongside the
# manipulation -- so they are a clean-behaviour ablation and not a valid patching
# test. Pooling them into a depth's implied-hit rate would launder that back into
# the one number this module exists to report. Depth-1 omission is a no-op and
# deduplicates against the written arm on its own.
def _arm_group(record: dict) -> tuple:
    return record["kind"], record["depth"], record["omit"]


def _group_order(entry):
    (kind, depth, omit), _ = entry
    return DONOR_KINDS.index(kind), depth or 0, omit


def _unique_top(probs: list[float] | None) -> bool | None:
    if not probs:
        return None
    ordered = sorted(probs)
    return ordered[-1] > ordered[-2]


def outcomes(report: dict, *, layer: int = POOLED_LAYER) -> list[dict]:
    """One record per (item, donor kind) at ``layer``.

    ``measured`` is False where the arm stored no distribution to take an argmax
    of. Those items keep a record -- they were run, and the count of what could
    not be read is part of the reading -- but they answer nothing, so every
    outcome field is None rather than False.
    """
    blocks = unflatten_rows(report)
    records = []
    for index, (item, block) in enumerate(zip(report["items"], blocks)):
        for kind in DONOR_KINDS:
            rows = [row for row in block if row["kind"] == kind]
            if not rows:
                continue
            at_layer = [row for row in rows if row["layer"] == layer]
            if not at_layer:
                continue
            row = at_layer[0]
            probs = row.get("probs_patched")
            top = probs.index(max(probs)) if probs else None
            clean = item.get("clean_probs")
            target = item.get("target_value")
            toward, toward_raw = (row.get("delta_toward"),
                                  row.get("delta_toward_raw"))
            records.append({
                "id": _measurement_id(item, kind, rows),
                "kind": kind,
                "index": index,
                "depth": report.get("depth"),
                "omit": report.get("omit", "none"),
                "seed": report.get("seed"),
                "layer": layer,
                "target_value": target,
                "implied_value": row.get("implied_value"),
                "raw_value": row.get("raw_value"),
                "patched_top": top,
                "tv": row.get("tv"),
                "clean_correct": item.get("clean_top_digit") == target,
                "clean_unique": _unique_top(clean),
                "clean_target_share": (clean[target] if clean and
                                       target is not None else None),
                "measured": probs is not None,
                "on_implied": None if top is None
                else top == row.get("implied_value"),
                "on_raw": None if top is None else top == row.get("raw_value"),
                "on_clean": None if top is None else top == target,
                "toward_implied_over_raw": (
                    None if toward is None or toward_raw is None
                    else toward > toward_raw),
            })
    return records


def pool(reports: dict[str, dict], *, layer: int = POOLED_LAYER,
         generator: str | None = POOLED_GENERATOR) -> list[dict]:
    """Deduplicate the outcomes of many arms into one list of measurements.

    Arms are filtered to a single generator because the families are not
    comparable: ``v1_unpaired`` allowed the implied, raw and clean digits to
    coincide, so "landed on the implied digit" does not mean there what it means
    here. Raises if no surviving arm measured ``layer`` -- an empty pool because
    of a mistyped layer should not read as an empty result.
    """
    kept = {name: report for name, report in reports.items()
            if generator is None or report.get("generator") == generator}
    if kept and not any(layer in report.get("layer_bins", [])
                        for report in kept.values()):
        raise ValueError(f"no arm measured layer {layer}")
    merged: dict[tuple, dict] = {}
    for name, report in sorted(kept.items()):
        for record in outcomes(report, layer=layer):
            found = merged.setdefault(record["id"], {**record, "arms": []})
            found["arms"].append(name)
    for record in merged.values():
        record["n_arms"] = len(record["arms"])
        del record["id"]
    return list(merged.values())


def summarize(pooled: list[dict]) -> list[dict]:
    """Rates per donor kind and depth, with the clustering beside them.

    Clean-correct items are split out rather than filtered: on an item whose
    clean answer was already wrong there is no clean answer for the patch to
    move off, so the implied-hit rate there measures something else. ``seeds``
    travels with the counts because thirty-three measurements over four seeds
    are not thirty-three independent draws.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in pooled:
        groups[_arm_group(record)].append(record)
    rows = []
    for (kind, depth, omit), records in sorted(groups.items(), key=_group_order):
        correct = [r for r in records if r["clean_correct"] and r["measured"]]
        rows.append({
            "kind": kind,
            "depth": depth,
            "omit": omit,
            "n_items": len(records),
            "n_measured": sum(r["measured"] for r in records),
            "n_clean_correct": sum(bool(r["clean_correct"]) for r in records),
            "n_on_implied": sum(bool(r["on_implied"]) for r in records),
            "n_clean_correct_measured": len(correct),
            "n_on_implied_clean_correct": sum(bool(r["on_implied"])
                                              for r in correct),
            "n_on_raw_clean_correct": sum(bool(r["on_raw"]) for r in correct),
            "n_on_clean_clean_correct": sum(bool(r["on_clean"])
                                            for r in correct),
            "n_toward_implied_over_raw_clean_correct": sum(
                bool(r["toward_implied_over_raw"]) for r in correct),
            "seeds": sorted({r["seed"] for r in records if r["seed"] is not None}),
            "n_arms_max": max((r["n_arms"] for r in records), default=0),
        })
    return rows


def band_table(pooled: list[dict], bands=CONFIDENCE_BANDS) -> list[dict]:
    """The implied-hit rate against clean confidence, on clean-correct items.

    This is the table the pooling was for. If the depth-1 effect were an
    artefact of the model being unsure, the rate would fall as the share rises.

    Banded *within* each donor kind and depth, never across them. Depth and
    clean confidence are collinear in this family -- depth-1 items top out at
    0.961 and every depth-2 written item starts above it -- so a pooled top band
    is almost entirely depth-2 and depth-3 misses, and reads as the decay the
    table exists to look for.
    """
    eligible = [record for record in pooled
                if record["measured"] and record["clean_correct"]
                and record["clean_target_share"] is not None]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in pooled:
        groups.setdefault(_arm_group(record), [])
    for record in eligible:
        groups[_arm_group(record)].append(record)
    rows = []
    for (kind, depth, omit), records in sorted(groups.items(), key=_group_order):
        for index, (lower, upper) in enumerate(bands):
            last = index == len(bands) - 1
            inside = [record for record in records
                      if lower <= record["clean_target_share"] < upper
                      or (last and record["clean_target_share"] == upper)]
            rows.append({
                "kind": kind,
                "depth": depth,
                "omit": omit,
                "band": (lower, upper),
                "n_items": len(inside),
                "n_on_implied": sum(bool(r["on_implied"]) for r in inside),
                "n_on_raw": sum(bool(r["on_raw"]) for r in inside),
            })
    return rows


def load_reports(directory=ARTIFACT_DIR) -> dict[str, dict]:
    """Every arm under ``directory``, keyed by its path below it."""
    directory = Path(directory)
    reports = {}
    for path in sorted(directory.rglob("*.json")):
        if path.name == "MANIFEST.json" or path.name == POOLED_NAME:
            continue
        report = json.loads(path.read_text())
        if "rows" in report and "items" in report:
            reports[str(path.relative_to(directory).with_suffix(""))] = report
    return reports


POOLED_NAME = "POOLED.json"


def build(directory=ARTIFACT_DIR, *, layer=POOLED_LAYER,
          generator=POOLED_GENERATOR) -> dict:
    reports = load_reports(directory)
    pooled = pool(reports, layer=layer, generator=generator)
    return {
        "layer": layer,
        "generator": generator,
        "n_arms": len(reports),
        "n_measurements": len(pooled),
        "donor_kinds": list(DONOR_KINDS),
        "confidence_bands": [list(band) for band in CONFIDENCE_BANDS],
        "by_kind_and_depth": summarize(pooled),
        "by_confidence_band": [
            {**row, "band": list(row["band"])} for row in band_table(pooled)],
        "measurements": sorted(
            pooled, key=lambda r: (r["kind"], r["depth"] or 0, r["arms"][0],
                                   r["index"])),
    }


def _rate(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}" if denominator else "-"


def print_summary(built: dict) -> None:
    print(f"pooled at layer {built['layer']}, generator {built['generator']}: "
          f"{built['n_measurements']} measurements from {built['n_arms']} arms")
    print()
    header = (f"{'donor':>10} {'depth':>5} {'omit':>6} {'n':>3} {'seeds':>8} "
              f"{'clean ok':>9} | {'implied':>9} {'raw':>7} {'clean':>7} "
              f"{'toward>raw':>10}")
    print(header)
    print("-" * len(header))
    for row in built["by_kind_and_depth"]:
        n = row["n_clean_correct_measured"]
        print(f"{row['kind']:>10} {row['depth']:>5} {row['omit']:>6} "
              f"{row['n_items']:>3} {','.join(map(str, row['seeds'])):>8} "
              f"{_rate(row['n_clean_correct'], row['n_items']):>9} | "
              f"{_rate(row['n_on_implied_clean_correct'], n):>9} "
              f"{_rate(row['n_on_raw_clean_correct'], n):>7} "
              f"{_rate(row['n_on_clean_clean_correct'], n):>7} "
              f"{_rate(row['n_toward_implied_over_raw_clean_correct'], n):>10}")
    print()
    print("implied-digit rate against clean confidence, clean-correct items:")
    for row in built["by_confidence_band"]:
        if not row["n_items"]:
            continue
        low, high = row["band"]
        print(f"  {row['kind']:>10} depth {row['depth']} {row['omit']:>6}  "
              f"p(target) [{low:.2f}, {high:.2f})  "
              f"n={row['n_items']:>3}  implied "
              f"{_rate(row['n_on_implied'], row['n_items']):>7}  raw "
              f"{_rate(row['n_on_raw'], row['n_items']):>7}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--layer", type=int, default=POOLED_LAYER)
    parser.add_argument("--generator", default=POOLED_GENERATOR)
    parser.add_argument("--output", default=None,
                        help=f"write the pooled record (default: "
                             f"<dir>/{POOLED_NAME}); '-' writes nothing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    built = build(args.dir, layer=args.layer, generator=args.generator)
    print_summary(built)
    if args.output != "-":
        path = Path(args.output or Path(args.dir) / POOLED_NAME)
        path.write_text(json.dumps(built, indent=2) + "\n")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
