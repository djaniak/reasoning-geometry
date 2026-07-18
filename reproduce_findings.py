"""Reproduce the corrected (de-confounded) findings from this analysis.

Runs CHEAP on already-collected artifacts only -- no GPU, no model load, no rerun.
It reads:
  - greedy single-trace NPZ *metadata* (data/{model}/math500)  [contamination rates]
  - Best-of-N OOF score CSVs (results/{model}_bestofn_full/math500/...oof.csv)

It regenerates the three things the paper's correction rests on:
  A. Truncation/parseability contamination ledger (greedy + Best-of-N).
  B. Three-tier AUC table: correctness (all vs parseable-only) and termination,
     for RMD vs entropy vs the trace-length baseline, per model and layer.
  C. The within-prompt collapse: within-prompt concordance, all-traces vs parseable-only.

Usage:
    uv run python reproduce_findings.py
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from prompt_decomposition import within_prompt_concordance

GREEDY = {"qwen": ("data/qwen/math500", 1024), "deepseek": ("data/deepseek/math500", 2048)}
OOF = {
    "qwen": "results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv",
    "deepseek": "results/deepseek_bestofn_full/math500/math500_prompt_decomposition_oof.csv",
}
LAYERS = ("7", "14", "21")


def _auc(labels, scores):
    return roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")


def _scan_greedy(data_dir: str, cap: int) -> dict | None:
    files = sorted(glob.glob(os.path.join(data_dir, "batch_*.npz")))
    if not files:
        return None
    md = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:  # only "metadata" is touched -> cheap
            if "metadata" not in z:
                continue
            for m in z["metadata"]:
                md.append(m if isinstance(m, dict) else m.item())
    n = len(md)

    def unparsed(m):
        pred = m.get("predicted", m.get("predicted_answer"))
        return pred is None or str(pred).strip() == ""

    lengths = np.array([m["n_tokens"] for m in md])
    ninc = sum(1 for m in md if not int(m.get("is_correct", 0)))
    nun = sum(unparsed(m) for m in md)
    nun_inc = sum(1 for m in md if unparsed(m) and not int(m.get("is_correct", 0)))
    ncap = int((lengths >= cap).sum())
    return {
        "n": n,
        "unparsed_rate": nun / n,
        "capped_rate": ncap / n,
        "unparsed_share_incorrect": (nun_inc / ninc) if ninc else float("nan"),
        "len_median": float(np.median(lengths)),
    }


def _load_oof(path: str):
    by_layer: dict[str, list[dict]] = {}
    for r in csv.DictReader(open(path)):
        by_layer.setdefault(r["layer"], []).append(
            {
                "is_correct": int(r["is_correct"]),
                "prompt_id": int(r["prompt_id"]),
                "rmd": float(r["rmd_score"]),
                "entropy": float(r["entropy_score"]),
                "length": float(r["length_score"]),
                "rmd_score": float(r["rmd_score"]),        # keys for within_prompt_concordance
                "entropy_score": float(r["entropy_score"]),
                "unparsed": (r["predicted_answer"] or "").strip() == "",
            }
        )
    return by_layer


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> None:
    section("A. Contamination ledger")
    print(f"{'data':<32}{'unparsed':>10}{'capped':>9}{'unp/incorrect':>15}{'len med':>9}")
    for model, (d, cap) in GREEDY.items():
        s = _scan_greedy(d, cap)
        if s:
            print(f"{model+' greedy MATH-500':<32}{s['unparsed_rate']:>9.0%}"
                  f"{s['capped_rate']:>9.0%}{s['unparsed_share_incorrect']:>15.0%}"
                  f"{s['len_median']:>9.0f}")
    for model, path in OOF.items():
        if not os.path.exists(path):
            continue
        rows = _load_oof(path)["7"]
        nun = sum(r["unparsed"] for r in rows)
        ninc = sum(1 for r in rows if not r["is_correct"])
        print(f"{model+' best-of-N MATH-500':<32}{nun/len(rows):>9.0%}{'-':>9}"
              f"{(sum(1 for r in rows if r['unparsed'] and not r['is_correct'])/ninc):>15.0%}{'-':>9}")

    section("B. Three-tier AUC: correctness (all vs parseable) and termination")
    print(f"{'model':<10}{'layer':>6}{'tier':>22}{'rmd':>8}{'entropy':>9}{'length':>8}{'rmd-len':>9}")
    for model, path in OOF.items():
        if not os.path.exists(path):
            continue
        data = _load_oof(path)
        for L in LAYERS:
            rows = data[L]
            par = [r for r in rows if not r["unparsed"]]
            tiers = [
                ("correctness|all", rows, [r["is_correct"] for r in rows]),
                ("correctness|parseable", par, [r["is_correct"] for r in par]),
                ("termination(parsed?)", rows, [0 if r["unparsed"] else 1 for r in rows]),
            ]
            for name, sub, y in tiers:
                a = {k: _auc(y, [r[k] for r in sub]) for k in ("rmd", "entropy", "length")}
                print(f"{model:<10}{L:>6}{name:>22}{a['rmd']:>8.3f}{a['entropy']:>9.3f}"
                      f"{a['length']:>8.3f}{a['rmd']-a['length']:>9.3f}")

    section("C. Within-prompt concordance: all-traces vs parseable-only (the collapse)")
    print(f"{'model':<10}{'layer':>6}{'set':>16}{'mixed':>7}{'rmd':>8}{'entropy':>9}")
    for model, path in OOF.items():
        if not os.path.exists(path):
            continue
        data = _load_oof(path)
        for L in LAYERS:
            rows = data[L]
            par = [r for r in rows if not r["unparsed"]]
            for tag, sub in (("all", rows), ("parseable", par)):
                cr = within_prompt_concordance(sub, "rmd_score")
                ce = within_prompt_concordance(sub, "entropy_score")
                rmd = cr["macro"] if cr["macro"] is not None else float("nan")
                ent = ce["macro"] if ce["macro"] is not None else float("nan")
                print(f"{model:<10}{L:>6}{tag:>16}{cr['n_mixed_prompts']:>7}{rmd:>8.3f}{ent:>9.3f}")


if __name__ == "__main__":
    main()
