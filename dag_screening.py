"""Stage A of E2: screen items on clean behaviour, then choose comparable pairs.

The depth ladder collapses after depth 1, and depth is confounded with two
quantities that move with it. Eligible clean p(target) is 0.666 / 0.725 / 0.961
at depth 1 and 0.966 / 0.975 / 0.996 / 0.997 / 0.999 at depth 2 -- supports that
do not touch, so every depth-1 success is on an item the model was unsure of and
every depth-2 failure on one it was sure of. `ancestor_distance` at gap 0 is
{11, 24} against {23, 36}. A patch that fails to move a near-saturated readout
twelve tokens further away is not evidence about graph depth.

So the comparison has to be made on items matched for both, and this module
decides which those are. Two things about it matter more than the arithmetic:

- **It reads clean measurements only.** Screening is a forward pass per item and
  nothing else; no patch has been run when `select` is called, so the rule
  cannot be tuned toward a result. `tests/test_dag_screening.py` pins that by
  feeding it fabricated patched fields and asserting the answer does not move.
- **It is allowed to say no.** If the two supports do not overlap, or the
  overlap holds fewer than `MIN_PAIRS` matched items, the registered outcome is
  to stop and close the depth claim rather than run the comparison small.

The rule and the floor are registered in `EXPERIMENT_LOG.md`, 2026-08-15, before
any E2 item existed. Changing either after stage A has run is a different
experiment and should be logged as one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dag_tasks import DEFAULT_GENERATOR, generate_items

# Tokens. Adjacent depths differ by one token at gap 0 ({11, 24} against
# {23, 36}); this leaves room for that without letting a depth-1 item match a
# depth-2 item a whole line away.
DISTANCE_TOLERANCE = 2

# Fisher's exact at 16 per arm has 0.93 power against a fall from 0.9 to 0.3 and
# 0.58 against a fall to 0.5. So this detects a collapse and not an attenuation,
# and the floor is set where the first of those is worth running. Above 24 the
# marginal pair buys little and the screening cost is already paid.
MIN_PAIRS = 16
MAX_PAIRS = 24


def eligible(records: list[dict]) -> list[dict]:
    """Items with a clean answer the model committed to, uniquely.

    A tied clean readout has no answer for the patch to move off, and an item
    with no recorded share cannot be matched on one.
    """
    return [record for record in records
            if record.get("clean_correct_unique")
            and record.get("clean_target_share") is not None]


def confidence_window(records: list[dict], *, depths=(1, 2)) -> tuple | None:
    """The overlap of the per-depth eligible p(target) supports, or ``None``.

    Empirical rather than fixed in advance, which is safe because it is computed
    from clean measurements alone -- but it is genuinely allowed to be empty,
    and on the archived runs it would be: depth 1 reaches 0.961 and depth 2
    starts at 0.966. Empty is the answer, not a failure to find one.
    """
    shares = {}
    for depth in depths:
        at_depth = [record["clean_target_share"]
                    for record in eligible(records) if record["depth"] == depth]
        if not at_depth:
            return None
        shares[depth] = (min(at_depth), max(at_depth))
    low = max(bounds[0] for bounds in shares.values())
    high = min(bounds[1] for bounds in shares.values())
    return None if low > high else (low, high)


def _spine(record: dict) -> tuple:
    """What counts as one item's worth of evidence.

    Not the row and not the file: `depth1_gap{0,1,2}` hold the same five spines
    at three ancestor distances, which is how 33 pooled depth-1 observations
    came from 17 spines.
    """
    return record["seed"], record["index"]


def _order(record: dict) -> tuple:
    """A total order that does not depend on the order the records arrived in."""
    return (record["depth"], record["seed"], record["index"], record["gap"])


def match(records: list[dict], *, depths=(1, 2), window: tuple | None,
          distance_tolerance: int = DISTANCE_TOLERANCE,
          max_pairs: int = MAX_PAIRS) -> list[tuple[dict, dict]]:
    """One-to-one greedy pairing across the two depths, closest confidence first.

    Greedy rather than optimal on purpose: it is short enough to read, and the
    quantity being matched is a nuisance parameter, not the outcome. Ties in
    ``|delta share|`` fall to the closer distance and then to a fixed record
    order, so the result does not depend on how the arms were loaded.
    """
    if len(depths) != 2:
        raise ValueError(
            f"matching is between exactly two depths, got {list(depths)}")
    if window is None:
        return []
    low, high = window
    inside = {depth: sorted((record for record in eligible(records)
                             if record["depth"] == depth
                             and low <= record["clean_target_share"] <= high),
                            key=_order)
              for depth in depths}
    first, second = depths
    candidates = []
    for left in inside[first]:
        for right in inside[second]:
            distance = abs(left["ancestor_distance"] - right["ancestor_distance"])
            if distance > distance_tolerance:
                continue
            gap = abs(left["clean_target_share"] - right["clean_target_share"])
            candidates.append((gap, distance, _order(left), _order(right),
                               left, right))
    pairs, used = [], set()
    for _, _, _, _, left, right in sorted(candidates, key=lambda c: c[:4]):
        keys = ((first, _spine(left)), (second, _spine(right)))
        if any(key in used for key in keys):
            continue
        used.update(keys)
        pairs.append((left, right))
        if len(pairs) == max_pairs:
            break
    return pairs


def select(records: list[dict], *, depths=(1, 2),
           distance_tolerance: int = DISTANCE_TOLERANCE,
           min_pairs: int = MIN_PAIRS, max_pairs: int = MAX_PAIRS) -> dict:
    """The registered stage-A decision: which pairs, and whether to proceed.

    ``proceed`` false is a result. It says the depth contrast cannot be
    unconfounded within this item family, which closes the depth claim and
    leaves the one-step result standing on its own.
    """
    window = confidence_window(records, depths=depths)
    pairs = match(records, depths=depths, window=window,
                  distance_tolerance=distance_tolerance, max_pairs=max_pairs)
    return {
        "depths": list(depths),
        "window": window,
        "distance_tolerance": distance_tolerance,
        "min_pairs": min_pairs,
        "max_pairs": max_pairs,
        "n_eligible": {depth: sum(1 for record in eligible(records)
                                  if record["depth"] == depth)
                       for depth in depths},
        "n_pairs": len(pairs),
        "proceed": len(pairs) >= min_pairs,
        "pairs": pairs,
    }


# --------------------------------------------------------------------------
# The forward pass, which is the only part that needs a GPU
# --------------------------------------------------------------------------


def load_once(model_name: str):
    """The model and tokenizer, loaded one time for a whole screening session.

    Screening is one forward pass per item, so a per-batch load would dominate
    the run: the first attempt reloaded 339 weight shards for each of ten seeds.
    """
    import torch
    from transformers import AutoTokenizer

    from collect_data import load_model
    from dag_patching import READOUT_DTYPE

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model, _ = load_model(False, model_name=model_name,
                          dtype=getattr(torch, READOUT_DTYPE))
    return model, tokenizer


def screen(*, model, tokenizer, depth: int, n_items: int, n_decoys: int,
           seed: int, gap: int | None = None,
           generator: str = DEFAULT_GENERATOR) -> list[dict]:
    """Clean readouts for one (depth, gap, seed) batch. No patching, no edits.

    Deliberately a different entry point from `dag_patching.run`: stage A must
    not be able to produce a patched number, so the code path that would has
    been left out rather than switched off.
    """
    from dag_patching import (
        capture_states,
        digit_readout,
        digit_token_ids,
        layer_bins,
        readout_dtype,
    )
    from dag_pooling import _tops

    items = generate_items(
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        n_items=n_items, n_decoys=n_decoys, seed=seed, condition="both",
        depth=depth, gap=gap, generator=generator, omit="none",
    )
    digit_ids = digit_token_ids(tokenizer)
    bins = layer_bins(model.config.num_hidden_layers)

    records = []
    for index, item in enumerate(items):
        _, logits = capture_states(model, item.token_ids, bins,
                                   [item.read_position])
        probs, _, _ = digit_readout(logits, item.read_position, digit_ids)
        shares = [float(value) for value in probs]
        tops = _tops(shares)
        records.append({
            "depth": depth,
            "gap": item.gap,
            "seed": seed,
            "index": index,
            "generator": generator,
            "readout_dtype": readout_dtype(model),
            "ancestor_distance": next(
                edit.distance_to_read for edit in item.edits
                if edit.kind == "ancestor"),
            "target_value": item.target_value,
            "clean_probs": shares,
            "clean_target_share": shares[item.target_value],
            "clean_correct_unique": tops == [item.target_value],
            "clean_tied": len(tops) > 1,
        })
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_name",
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--gaps", type=int, nargs="+", default=[0],
                        help="Extra ancestor-distance placements. A distance "
                             "sampler only: one placement per spine survives "
                             "selection.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Disjoint from 0-3, which the archived runs used. "
                             "Required unless --screened is given.")
    parser.add_argument("--screened", nargs="+", default=None, metavar="JSON",
                        help="Select over already-screened files instead of "
                             "running forwards. Depth 1 needs the gap sweep to "
                             "widen its ancestor distances and depth 2 does "
                             "not, so the two are screened separately and the "
                             "registered rule is applied over both at once -- "
                             "by this code, not by a one-off script.")
    parser.add_argument("--n_items", type=int, default=10)
    parser.add_argument("--n_decoys", type=int, default=6)
    parser.add_argument("--generator", default=DEFAULT_GENERATOR)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_screened(paths) -> list[dict]:
    """Records from earlier screening files, deduplicated by what identifies one.

    Two files can legitimately hold the same item -- rerunning a seed, or
    overlapping seed ranges -- and one item screened twice is still one item.
    """
    seen = {}
    for path in paths:
        for record in json.loads(Path(path).read_text())["screened"]:
            seen[(record["depth"], record["seed"], record["index"],
                  record["gap"], record.get("generator"))] = record
    return [seen[key] for key in sorted(seen)]


def main() -> None:
    args = parse_args()
    if args.screened:
        records = load_screened(args.screened)
    else:
        if args.seeds is None:
            raise SystemExit("--seeds is required unless --screened is given")
        model, tokenizer = load_once(args.model_name)
        records = []
        for depth in args.depths:
            for gap in args.gaps:
                for seed in args.seeds:
                    records.extend(screen(
                        model=model, tokenizer=tokenizer, depth=depth,
                        n_items=args.n_items, n_decoys=args.n_decoys,
                        seed=seed, gap=gap, generator=args.generator))

    # Written before anything is selected. Screening is the expensive half and
    # selection is pure, so a selection that raises should cost a rerun of the
    # rule and not of four hundred forward passes -- which is exactly what the
    # first attempt cost.
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": args.model_name, "screened": records, "selection": None}
    path.write_text(json.dumps(payload, indent=2))
    print(f"screened       {len(records)} items over depths {args.depths}")
    print(f"wrote {path}")

    if len(args.depths) != 2:
        print("selection      not run: it is a comparison between two depths, "
              f"and this file holds {args.depths}. Re-run with --screened over "
              "both.")
        return
    outcome = select(records, depths=tuple(args.depths))
    payload["selection"] = outcome
    path.write_text(json.dumps(payload, indent=2))
    window = outcome["window"]
    print(f"eligible       {outcome['n_eligible']}")
    print(f"window         "
          f"{'none -- the supports do not overlap' if window is None else window}")
    print(f"matched pairs  {outcome['n_pairs']} "
          f"(floor {outcome['min_pairs']}, ceiling {outcome['max_pairs']})")
    print(f"decision       {'proceed to stage B' if outcome['proceed'] else 'STOP'}")


if __name__ == "__main__":
    main()
