"""Loop and cap precursors in Best-of-N traces.

Capped traces are the largest contamination source in every prompt-level result
this project reports, and every other open direction predicts the same binary
correctness label from the same states.  This module targets the caps themselves:
*why* a trace runs to the budget, *when* that becomes detectable, and whether the
geometry says it before a free token-level statistic does.

Scope note: this is DeepSeek-only.  ``data/qwen_bestofn_full`` stores entropies,
token log-probs, and hidden states but no ``tokens_*`` arrays and no
``generated_text``, so no token- or text-level analysis can run on Qwen.  Nothing
here is a cross-model replication and it must not be written up as one.

Stages are separate because their costs differ by orders of magnitude: the token
stages stream small arrays out of every batch, while the precursor stage needs
hidden states and so runs on a sampled subset.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_NGRAM = 8
DEFAULT_WINDOW = 200
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def repetition_flags(tokens: list, n: int = DEFAULT_NGRAM) -> np.ndarray:
    """Flag each n-gram start position whose n-gram already occurred earlier.

    Novelty against the whole prefix, not within a window: a trace that recycles
    material from thousands of tokens back is looping just as surely as one that
    repeats itself locally, and a windowed measure would miss it.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    codes: dict[object, int] = {}
    encoded = [codes.setdefault(token, len(codes)) for token in tokens]
    if len(encoded) < n:
        return np.zeros(0, dtype=bool)
    seen: set[tuple[int, ...]] = set()
    flags = np.empty(len(encoded) - n + 1, dtype=bool)
    for i in range(len(flags)):
        gram = tuple(encoded[i : i + n])
        flags[i] = gram in seen
        seen.add(gram)
    return flags


def tail_periodicity(
    tokens: list, *, tail: int = 500, max_period: int = 200
) -> tuple[int, float]:
    """Best repeating period in the trace's tail, and how well it matches.

    Reading the sampled traces showed that "an n-gram occurred before" does not
    separate a degenerate loop from ordinary mathematical repetition -- symmetric
    case analysis, re-verification, and formula templates all re-tread n-grams
    while the reasoning still advances.  What does separate them is *periodicity*:
    a stuck trace cycles a short fixed block, while coherent reasoning does not.

    Returns ``(period, match_fraction)`` where ``match_fraction`` is the share of
    tail positions equal to the token one period earlier.
    """
    window = [str(token) for token in tokens[-tail:]]
    if len(window) < 4:
        return 0, 0.0
    best_period, best_score = 0, 0.0
    for period in range(1, min(max_period, len(window) // 2) + 1):
        matches = sum(
            window[i] == window[i - period] for i in range(period, len(window))
        )
        score = matches / (len(window) - period)
        if score > best_score:
            best_period, best_score = period, score
    return best_period, float(best_score)


def repetition_onset(
    tokens: list,
    *,
    n: int = DEFAULT_NGRAM,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
) -> int | None:
    """Return the first token index whose trailing window is mostly stale n-grams.

    The returned index is the *end* of the first qualifying window -- the earliest
    point at which the evidence for a loop is actually in hand, which is what an
    online early-stop rule could act on.
    """
    flags = repetition_flags(tokens, n=n)
    if flags.size < window:
        return None
    rolling = np.convolve(flags.astype(float), np.ones(window) / window, mode="valid")
    hits = np.flatnonzero(rolling > threshold)
    if hits.size == 0:
        return None
    return int(hits[0] + window + n - 2)


# ---------------------------------------------------------------------------
# Streaming trace summaries
# ---------------------------------------------------------------------------

def iter_batches(data_dir: str):
    for path in sorted(Path(data_dir).glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            yield path, data


def summarize_traces(
    data_dir: str,
    *,
    cap: int,
    n: int = DEFAULT_NGRAM,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    limit_batches: int | None = None,
) -> list[dict]:
    """One row per trace: cap status, correctness, and detector onset.

    Streams batch by batch and keeps only the summary; the token arrays for a
    full Best-of-N collection do not fit in memory at once.
    """
    rows: list[dict] = []
    for count, (path, data) in enumerate(iter_batches(data_dir)):
        if limit_batches is not None and count >= limit_batches:
            break
        available = set(data.files)
        for meta in data["metadata"]:
            trace_id = int(meta["trace_id"])
            token_key = f"tokens_{trace_id}"
            if token_key not in available:
                continue
            tokens = data[token_key].tolist()
            entropies = data.get(f"entropies_{trace_id}")
            onset = repetition_onset(tokens, n=n, window=window, threshold=threshold)
            period, periodicity = tail_periodicity(tokens)
            rows.append(
                {
                    "batch": path.name,
                    "trace_id": trace_id,
                    "prompt_id": int(meta["idx"]),
                    "sample_id": int(meta["sample_id"]),
                    "is_correct": bool(meta["is_correct"]),
                    "predicted_answer": _clean(meta["predicted"]),
                    "n_tokens": int(meta["n_tokens"]),
                    "capped": bool(int(meta["n_tokens"]) >= cap),
                    "onset": onset,
                    "onset_frac": None if onset is None else onset / cap,
                    "tail_period": period,
                    "tail_periodicity": round(periodicity, 4),
                    "tail_entropy": (
                        float(np.mean(entropies[-window:]))
                        if entropies is not None and len(entropies) >= 1
                        else None
                    ),
                }
            )
    return rows


def _clean(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text in ("", "None", "nan") else text


# ---------------------------------------------------------------------------
# L1 -- calibration and the flagged-but-uncapped question
# ---------------------------------------------------------------------------

def onset_report(rows: list[dict]) -> dict:
    """Detector behaviour split by cap status, and flagged-uncapped by correctness.

    The flagged-but-uncapped traces are the point of this report.  If they are
    disproportionately wrong, the detector is not misfiring on them -- it is
    reading "this trace is going nowhere", which is a larger claim than cap
    prediction and has to be reported as such.
    """
    capped = [row for row in rows if row["capped"]]
    uncapped = [row for row in rows if not row["capped"]]
    flagged_uncapped = [row for row in uncapped if row["onset"] is not None]
    clean_uncapped = [row for row in uncapped if row["onset"] is None]
    detected = [row["onset_frac"] for row in capped if row["onset"] is not None]
    return {
        "n_traces": len(rows),
        "n_capped": len(capped),
        "n_uncapped": len(uncapped),
        "capped_detected": sum(1 for row in capped if row["onset"] is not None),
        "capped_detection_rate": _rate(
            sum(1 for row in capped if row["onset"] is not None), len(capped)
        ),
        "onset_frac_percentiles": (
            {
                str(p): round(float(np.percentile(detected, p)), 4)
                for p in (10, 25, 50, 75, 90)
            }
            if detected
            else None
        ),
        "uncapped_flagged": len(flagged_uncapped),
        "uncapped_flag_rate": _rate(len(flagged_uncapped), len(uncapped)),
        "accuracy_uncapped_flagged": _rate(
            sum(row["is_correct"] for row in flagged_uncapped), len(flagged_uncapped)
        ),
        "accuracy_uncapped_unflagged": _rate(
            sum(row["is_correct"] for row in clean_uncapped), len(clean_uncapped)
        ),
        "accuracy_capped": _rate(sum(row["is_correct"] for row in capped), len(capped)),
        "degenerate": _degenerate_report(capped, uncapped),
    }


# Calibrated on the L0 hand sample: the two traces read as genuine degenerate
# loops scored 1.000 and 0.423, while the other nineteen -- all coherent, merely
# unfinished reasoning -- topped out at 0.188.
DEGENERATE_PERIODICITY = 0.30


def _degenerate_report(capped: list[dict], uncapped: list[dict]) -> dict:
    """How much of the capped population is a stuck loop rather than unfinished work."""
    stuck = [
        row for row in capped if row["tail_periodicity"] >= DEGENERATE_PERIODICITY
    ]
    return {
        "threshold": DEGENERATE_PERIODICITY,
        "n_capped_degenerate": len(stuck),
        "share_of_capped": _rate(len(stuck), len(capped)),
        "n_uncapped_degenerate": sum(
            row["tail_periodicity"] >= DEGENERATE_PERIODICITY for row in uncapped
        ),
        "share_of_uncapped": _rate(
            sum(row["tail_periodicity"] >= DEGENERATE_PERIODICITY for row in uncapped),
            len(uncapped),
        ),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


# ---------------------------------------------------------------------------
# L0 -- hand taxonomy sample
# ---------------------------------------------------------------------------

def taxonomy_sample(
    rows: list[dict],
    data_dir: str,
    *,
    per_stratum: int = 7,
    seed: int = 42,
    tail_chars: int = 1500,
) -> list[dict]:
    """Sample capped traces stratified by detector outcome, with text to read.

    Stratifying by the detector's own verdict is deliberate: the no-onset stratum
    is where the detector would be wrong if the phenomenon is real, and reading it
    is the only way to find out before fitting anything.
    """
    rng = np.random.default_rng(seed)
    capped = [row for row in rows if row["capped"]]
    strata = {
        "early_onset": [
            r for r in capped if r["onset"] is not None and r["onset_frac"] < 0.5
        ],
        "late_onset": [
            r for r in capped if r["onset"] is not None and r["onset_frac"] >= 0.5
        ],
        "no_onset": [r for r in capped if r["onset"] is None],
    }
    picked: list[dict] = []
    for name, members in strata.items():
        if not members:
            continue
        take = min(per_stratum, len(members))
        for index in rng.choice(len(members), size=take, replace=False):
            row = dict(members[int(index)])
            row["stratum"] = name
            row["tail_text"] = _tail_text(data_dir, row, tail_chars)
            picked.append(row)
    return picked


def _tail_text(data_dir: str, row: dict, tail_chars: int) -> str | None:
    with np.load(Path(data_dir) / row["batch"], allow_pickle=True) as data:
        for meta in data["metadata"]:
            if int(meta["trace_id"]) == row["trace_id"]:
                text = meta["generated_text"] if "generated_text" in meta else None
                return None if text is None else str(text)[-tail_chars:]
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--ngram", type=int, default=DEFAULT_NGRAM)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--per_stratum", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = summarize_traces(
        args.data_dir,
        cap=args.max_new_tokens,
        n=args.ngram,
        window=args.window,
        threshold=args.threshold,
        limit_batches=args.limit_batches,
    )
    report = onset_report(rows)
    settings = {
        "data_dir": args.data_dir,
        "max_new_tokens": args.max_new_tokens,
        "ngram": args.ngram,
        "window": args.window,
        "threshold": args.threshold,
        "seed": args.seed,
    }
    (output_dir / "trace_summaries.json").write_text(
        json.dumps({"settings": settings, "rows": rows}, indent=2)
    )
    (output_dir / "onset_report.json").write_text(
        json.dumps({"settings": settings, "report": report}, indent=2)
    )

    sample = taxonomy_sample(
        rows,
        args.data_dir,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )
    (output_dir / "taxonomy_sample.json").write_text(json.dumps(sample, indent=2))

    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
