"""
merge_results.py - Merge partial analyze.py outputs into one canonical result JSON.

This keeps DVC stages incremental: base, controls, and subspace can run
independently, then this script assembles the final <dataset>_results.json
that existing notebooks and summarize.py already expect.
"""
import argparse
import json
import os
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True, help="Path for merged JSON output")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more partial result JSON files to merge in order",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def merge_dicts(base: dict, incoming: dict) -> dict:
    merged = deepcopy(base)
    for key, value in incoming.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def validate_compatible(anchor: dict, candidate: dict, path: str):
    checks = ["dataset", "n_correct", "n_incorrect"]
    for key in checks:
        if key in anchor and key in candidate and anchor[key] != candidate[key]:
            raise ValueError(
                f"Incompatible partial result for {path}: {key}={candidate[key]!r} "
                f"does not match {anchor[key]!r}"
            )


def main():
    args = parse_args()
    merged = None

    for path in args.inputs:
        current = load_json(path)
        if merged is None:
            merged = current
            continue
        validate_compatible(merged, current, path)
        merged = merge_dicts(merged, current)

    if merged is None:
        raise ValueError("No inputs were provided for merge")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as fh:
        json.dump(merged, fh, indent=2)

    print(f"Merged {len(args.inputs)} partial result files into {args.output_path}")


if __name__ == "__main__":
    main()
