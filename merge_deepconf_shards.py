"""Merge sharded exact-DeepConf outputs into one artifact.

`deepconf_exact.py` shards by striding the prompt list, so the shards partition
the prompts and merging is a concatenation followed by a sort on `prompt_id`.
The reconstruction checks are the reason this is a script and not a `cat`: they
are per-shard maxima and means over different token counts, so they have to be
recombined as a weighted max/mean rather than averaged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def merge(shard_dirs: list[Path], output_dir: Path, *, stem: str) -> dict:
    prompt_ids: list[int] = []
    confidence: list[np.ndarray] = []
    summaries: list[dict] = []
    metas: list[dict] = []

    for directory in shard_dirs:
        npz = np.load(directory / "deepconf_exact_pilot.npz", allow_pickle=True)
        meta = json.loads((directory / "deepconf_exact_pilot.json").read_text())
        prompt_ids.extend(int(value) for value in npz["prompt_ids"])
        confidence.extend(npz["exact_token_confidence"])
        summaries.extend(npz["trace_summaries"])
        metas.append(meta)

    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("shards overlap: the same prompt_id appears twice")

    order = np.argsort(np.asarray(prompt_ids, dtype=np.int64), kind="stable")
    prompt_ids = [prompt_ids[i] for i in order]
    # Traces stay grouped with their prompt, so sort them on prompt_id too.
    trace_order = sorted(range(len(summaries)), key=lambda i: (int(summaries[i]["prompt_id"]),
                                                              int(summaries[i]["trace_id"])))
    confidence = [confidence[i] for i in trace_order]
    summaries = [summaries[i] for i in trace_order]

    checks = [meta["reconstruction_checks"] for meta in metas]
    weights = np.asarray([check["n_error_values"] for check in checks], dtype=float)
    total = float(weights.sum())

    def weighted(key: str) -> float:
        return float(np.sum(weights * [check[key] for check in checks]) / total) if total else 0.0

    merged_meta = dict(metas[0])
    merged_meta.update(
        {
            "sample_size": len(prompt_ids),
            "roundtrip_token_mismatches": int(
                sum(meta["roundtrip_token_mismatches"] for meta in metas)
            ),
            "reconstruction_checks": {
                "max_entropy_abs_error": max(c["max_entropy_abs_error"] for c in checks),
                "max_sampled_logprob_abs_error": max(
                    c["max_sampled_logprob_abs_error"] for c in checks
                ),
                "mean_entropy_abs_error": weighted("mean_entropy_abs_error"),
                "mean_sampled_logprob_abs_error": weighted("mean_sampled_logprob_abs_error"),
                "n_error_values": int(total),
            },
            "shards": [str(directory) for directory in shard_dirs],
        }
    )
    merged_meta.pop("shard_index", None)
    merged_meta.pop("num_shards", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        prompt_ids=np.asarray(prompt_ids, dtype=np.int64),
        exact_token_confidence=np.array(confidence, dtype=object),
        trace_summaries=np.array(summaries, dtype=object),
    )
    (output_dir / f"{stem}.json").write_text(json.dumps(merged_meta, indent=2))
    return merged_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_dir", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stem", default="deepconf_exact_full")
    args = parser.parse_args()
    meta = merge([Path(d) for d in args.shard_dir], Path(args.output_dir), stem=args.stem)
    print(f"{meta['sample_size']} prompts, "
          f"{meta['roundtrip_token_mismatches']} token mismatches, "
          f"mean entropy error {meta['reconstruction_checks']['mean_entropy_abs_error']:.6f}")


if __name__ == "__main__":
    main()
