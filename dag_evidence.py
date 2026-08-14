"""Provenance and offline reconstruction for the DAG patching artifacts.

The eight runs in ``results/dag_patching/`` are the only record of the pilot's
per-item measurements. This module makes them auditable without a GPU:

* the original JSON files are never rewritten -- they are committed byte-for-byte
  and content-addressed by sha256;
* the first three predate the ``depth``/``gap``/``ancestor_distance`` fields, so
  those values are *derived* into a side file rather than backfilled into the
  originals;
* a derivation is accepted only if regenerating the items reproduces the archived
  measurements. Generation is deterministic and tokenizer-only, so a mismatch
  means the generator moved under the artifact and the derived numbers would be
  describing a different run.

Nothing here loads a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from dag_patching import GATE_POLICIES, rescore_report
from dag_tasks import ANCESTOR, CHECKPOINTS, generate_items

ARTIFACT_DIR = Path("results/dag_patching")

# The run order, oldest first. The first three are schema v0.
ARTIFACTS = (
    "feasibility",
    "result_only",
    "operand_only",
    "depth1_gap0",
    "depth2_gap0",
    "depth3_gap0",
    "depth1_gap1",
    "depth1_gap2",
)

# Not stored in any report; it was the CLI default for every run and the log
# records "six decoy nodes". A wrong value changes the token count, which
# `verify_reconstruction` rejects.
DEFAULT_N_DECOYS = 6

# `feasibility.json` predates the donor split, so it has no `condition` either.
# The condition changes only the donor *text*: positions, token count, target
# value, and every distance are identical across conditions, so a reconstruction
# cannot confirm it. It is inferred from the run order and the log, not derived.
DEFAULT_CONDITION = "both"

# Provenance the reports may or may not state. Whether a field is inferred is a
# property of the individual report -- `result_only` and `operand_only` do record
# their condition, and calling it inferred there understates what we have. Only
# `n_decoys` is absent from all eight, v1 included.
PROVENANCE_FIELDS = ("condition", "n_decoys")

SCHEMA_V1_FIELDS = ("depth", "gap", "ancestor_distance")

# Which commit was checked out when each run was written. No report records it.
# Recovered from the artifact mtimes (see `resolve_run_commit`) and frozen here,
# because mtime is checkout time after a clone and the evidence would be lost.
#
# Corroborated by a second, mtime-independent signal: a report carrying a schema
# field cannot predate the commit that added the field, and a report missing it
# cannot postdate that commit. Both signals agree for all eight; the test
# `test_every_frozen_run_commit_agrees_with_the_schema_its_report_carries`
# re-checks the second one against the repository.
RUN_COMMITS = {
    "feasibility": "015a0f40202b50ab4b8fc56ab16089f4d7a6734b",
    "result_only": "e8117b5ad2547bc96e15135fb47958d8a6e5ccb4",
    "operand_only": "e8117b5ad2547bc96e15135fb47958d8a6e5ccb4",
    "depth1_gap0": "60efa8dd2c7bf086b286b4821acf775ca6fa7e90",
    "depth2_gap0": "60efa8dd2c7bf086b286b4821acf775ca6fa7e90",
    "depth3_gap0": "60efa8dd2c7bf086b286b4821acf775ca6fa7e90",
    "depth1_gap1": "60efa8dd2c7bf086b286b4821acf775ca6fa7e90",
    "depth1_gap2": "60efa8dd2c7bf086b286b4821acf775ca6fa7e90",
}

# The commit that introduced each field, for the corroborating check above.
SCHEMA_INTRODUCED = {
    "condition": "e8117b5ad2547bc96e15135fb47958d8a6e5ccb4",
    "depth": "60efa8dd2c7bf086b286b4821acf775ca6fa7e90",
}

# The commit that put the artifacts under git. An mtime at or after it is a
# checkout timestamp, not a run timestamp.
ARCHIVE_COMMIT = "605b676b07c5a41508a3442908790439a4bfd17e"


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def schema_version(report: dict) -> str:
    """``v1`` once depth/gap/distance were recorded, ``v0`` before that."""
    return "v1" if all(field in report for field in SCHEMA_V1_FIELDS) else "v0"


def inferred_fields(report: dict) -> list[str]:
    """Provenance this report does not state, so it comes from the log instead.

    Built from what is absent rather than asserted for a whole schema version:
    two of the three v0 runs do record their condition, and every run -- v1
    included -- leaves `n_decoys` unrecorded.
    """
    return [field for field in PROVENANCE_FIELDS if field not in report]


def commit_timeline() -> list[tuple[str, int]]:
    """``(sha, unix time)`` for every commit, newest first."""
    result = subprocess.run(
        ["git", "log", "--format=%H %ct"], capture_output=True, text=True,
        check=True,
    )
    return [(line.split()[0], int(line.split()[1]))
            for line in result.stdout.splitlines() if line.strip()]


def resolve_run_commit(mtime: float, timeline, archive_time: float) -> str | None:
    """The commit checked out when a file with this mtime was written.

    ``None`` when the mtime cannot be a run time: at or after ``archive_time``
    it is a checkout timestamp (git does not preserve mtimes), and before every
    commit there is nothing to attribute it to. Both cases must say "unknown"
    rather than confidently name the nearest commit.
    """
    if mtime >= archive_time:
        return None
    for sha, when in timeline:  # newest first
        if when <= mtime:
            return sha
    return None


def run_commit_evidence(name: str, path) -> dict:
    """The frozen run commit for ``name``, plus the mtime check behind it."""
    frozen = RUN_COMMITS.get(name)
    timeline = commit_timeline()
    times = dict(timeline)
    archive_time = times.get(ARCHIVE_COMMIT)
    mtime = Path(path).stat().st_mtime
    resolved = (resolve_run_commit(mtime, timeline, archive_time)
                if archive_time else None)
    return {
        "run_commit": frozen,
        "basis": "artifact mtime, bracketed against the commit timeline; "
                 "corroborated by which schema fields the report carries",
        "artifact_mtime_utc": datetime.fromtimestamp(mtime, UTC).isoformat(),
        # False after a fresh clone, where mtime is checkout time. That does not
        # weaken `run_commit` -- it only means this host can no longer re-derive
        # it, which is why the value is frozen in the first place.
        "mtime_still_confirms": resolved == frozen,
    }


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def tokenizer_revision(model_name: str) -> str | None:
    """The resolved snapshot hash in the local Hugging Face cache, if present."""
    slug = "models--" + model_name.replace("/", "--")
    snapshots = Path.home() / ".cache/huggingface/hub" / slug / "snapshots"
    if not snapshots.is_dir():
        return None
    revisions = sorted(path.name for path in snapshots.iterdir() if path.is_dir())
    return revisions[0] if len(revisions) == 1 else None


# --------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------


def reconstruct_items(report: dict, encode):
    """Regenerate a run's items from its recorded settings.

    ``depth``/``gap`` are absent on v0 reports; both defaults reproduce what the
    generator did before those knobs existed -- depth 1 and a random decoy split.
    """
    return generate_items(
        encode,
        n_items=report["n_items"],
        n_decoys=report.get("n_decoys", DEFAULT_N_DECOYS),
        seed=report["seed"],
        condition=report.get("condition", DEFAULT_CONDITION),
        depth=report.get("depth", 1),
        gap=report.get("gap") if isinstance(report.get("gap"), int) else None,
    )


def _edit_groups(report: dict, index: int) -> list[list[dict]]:
    """One group of layer-rows per edit, for item ``index``."""
    n_bins = len(report["layer_bins"])
    per_item = len(report["rows"]) // report["n_items"]
    block = report["rows"][index * per_item:(index + 1) * per_item]
    return [block[start:start + n_bins] for start in range(0, per_item, n_bins)]


def verify_reconstruction(report: dict, items) -> dict:
    """Reject a reconstruction that does not reproduce the archived run.

    Checked: item count, token count per item, target value per item, and the
    kind, node, and ``distance_to_read`` of every edit in recorded order.

    Not checked, because no report stores them: the raw token ids and the prompt
    text. Token count plus every edit's distance to the read position pins the
    line layout tightly, but this is agreement on the recorded fields, not a
    byte-level replay.
    """
    if len(items) != report["n_items"]:
        raise ValueError(
            f"reconstructed {len(items)} items, report records {report['n_items']}"
        )
    for index, item in enumerate(items):
        expected_tokens = report["n_tokens"][index]
        if len(item.token_ids) != expected_tokens:
            raise ValueError(
                f"item {index}: token count {len(item.token_ids)} does not match "
                f"the archived {expected_tokens}"
            )
        archived_target = report["items"][index]["target_value"]
        if item.target_value != archived_target:
            raise ValueError(
                f"item {index}: target value {item.target_value} does not match "
                f"the archived {archived_target}"
            )
        groups = _edit_groups(report, index)
        if len(groups) != len(item.edits):
            raise ValueError(
                f"item {index}: reconstructed {len(item.edits)} edits, report "
                f"records {len(groups)}"
            )
        for position, (edit, group) in enumerate(zip(item.edits, groups)):
            if group[0]["kind"] != edit.kind:
                raise ValueError(
                    f"item {index} edit {position}: edit kind {edit.kind!r} does "
                    f"not match the archived {group[0]['kind']!r}"
                )
            if group[0]["node"] != edit.node:
                raise ValueError(
                    f"item {index} edit {position}: node {edit.node!r} does not "
                    f"match the archived {group[0]['node']!r}"
                )
            if group[0]["distance_to_read"] != edit.distance_to_read:
                raise ValueError(
                    f"item {index} edit {position}: distance_to_read "
                    f"{edit.distance_to_read} does not match the archived "
                    f"{group[0]['distance_to_read']}"
                )
    return {"matches": True, "n_items": len(items),
            "checked": ["n_tokens", "target_value", "edit_kind", "edit_node",
                        "distance_to_read"]}


def derive_v0_fields(report: dict, items) -> dict:
    """The fields the v0 schema did not record. Verification first, always."""
    verification = verify_reconstruction(report, items)
    return {
        "derived": True,
        "verification": verification,
        # Recovered from the archived rows and confirmed by `verification`.
        "depth": items[0].depth,
        "gap": [item.gap for item in items],
        "ancestor_distance": [
            next(edit.distance_to_read for edit in item.edits
                 if edit.kind == "ancestor")
            for item in items
        ],
        "ancestor_node": ANCESTOR,
    }
    # `condition` and `n_decoys` are deliberately absent. They are provenance,
    # not derivation -- nothing here checked them against the measurement -- so
    # they live on the manifest entry under `inferred_fields`, alongside the same
    # flag for the v1 runs that also never recorded `n_decoys`.


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def verdict_table(reports: dict[str, dict]) -> list[dict]:
    """One row per artifact: archived verdict, each policy's verdict, why."""
    table = []
    for name, report in reports.items():
        rescored = rescore_report(report)
        entry = {
            "name": name,
            "condition": report.get("condition", DEFAULT_CONDITION),
            "schema": schema_version(report),
            "original_verdict": report.get("verdict"),
        }
        entry["reasons"] = {}
        for policy in GATE_POLICIES:
            entry[policy] = rescored["scoring"][policy]["verdict"]
            entry["reasons"][policy] = ", ".join(
                rescored["scoring"][policy]["invalid_reasons"]) or "-"
        active = rescored["gate_policy_version"]
        entry["active_policy"] = active
        entry["changed"] = entry["original_verdict"] != rescored["verdict"]
        entry["reason"] = entry["reasons"][active]
        entry["policies_disagree"] = len(
            {entry[policy] for policy in GATE_POLICIES}) > 1
        # Surface pass counts at the scoring layers, which is where the two
        # policies actually differ.
        entry["surface_per_layer"] = {
            policy: {
                str(layer): rescored["scoring"][policy]["gates"][
                    f"surface_{policy}"]["per_layer"][layer]["surface_items"]
                for layer in rescored["gates"]["scoring_layers"]
            }
            for policy in GATE_POLICIES
        }
        table.append(entry)
    return table


def load_artifacts(directory=ARTIFACT_DIR) -> dict[str, dict]:
    reports = {}
    for name in ARTIFACTS:
        path = Path(directory) / f"{name}.json"
        if path.is_file():
            reports[name] = json.loads(path.read_text())
    return reports


def build_manifest(directory=ARTIFACT_DIR) -> dict:
    from transformers import AutoTokenizer

    directory = Path(directory)
    reports = load_artifacts(directory)
    tokenizers: dict[str, object] = {}
    entries = []
    for name, report in reports.items():
        path = directory / f"{name}.json"
        model = report["model"]
        if model not in tokenizers:
            tokenizers[model] = AutoTokenizer.from_pretrained(model)
        tokenizer = tokenizers[model]
        entry = {
            "artifact": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "schema_version": schema_version(report),
            "model": model,
            "tokenizer_revision": tokenizer_revision(model),
            "seed": report["seed"],
            "condition": report.get("condition", DEFAULT_CONDITION),
            "n_items": report["n_items"],
            "n_decoys": report.get("n_decoys", DEFAULT_N_DECOYS),
            "layer_bins": report["layer_bins"],
            # Which of the two fields above the report did not state itself.
            "inferred_fields": inferred_fields(report),
            **run_commit_evidence(name, path),
            # Reconstructed from the recorded settings, not captured at run time:
            # it reproduces the run, it is not a transcript of what was typed.
            "replay_command": _command_for(name, report),
        }
        if entry["schema_version"] == "v0":
            items = reconstruct_items(
                report,
                lambda text: tokenizer.encode(text, add_special_tokens=False),
            )
            entry["derived_fields"] = derive_v0_fields(report, items)
        entries.append(entry)
    return {
        "note": "Original artifacts are immutable and content-addressed. v0 "
                "fields are derived by regenerating items, never backfilled. "
                "Per artifact: `run_commit` is the commit that produced the run, "
                "`replay_command` reproduces it from the recorded settings, and "
                "`inferred_fields` names provenance the report never stated.",
        # The commit this manifest was generated at -- not the commit any run
        # was produced at. That is `run_commit`, per artifact.
        "manifest_generation_commit": git_commit(),
        "artifacts": entries,
    }


def _command_for(name: str, report: dict) -> str:
    condition = report.get("condition", DEFAULT_CONDITION)
    parts = [f"uv run python dag_patching.py --model_name {report['model']}",
             f"--condition {condition}", f"--seed {report['seed']}",
             f"--n_items {report['n_items']}"]
    if schema_version(report) == "v1":
        parts.append(f"--depth {report['depth']}")
        gaps = report.get("gap")
        if isinstance(gaps, list) and len(set(gaps)) == 1:
            parts.append(f"--gap {gaps[0]}")
    parts.append(f"--output results/dag_patching/{name}.json")
    return " ".join(parts)


def print_verdict_table(table: list[dict]) -> None:
    columns = [
        ("artifact", lambda e: e["name"]),
        ("schema", lambda e: e["schema"]),
        ("archived", lambda e: str(e["original_verdict"])),
        ("v1 two-sided", lambda e: e["v1_two_sided"]),
        ("v2 one-sided *", lambda e: e["v2_one_sided"]),
        ("changed?", lambda e: "yes" if e["changed"] else "no"),
        ("active reason", lambda e: e["reason"]),
    ]
    widths = [
        max(len(title), *(len(get(entry)) for entry in table)) + 2
        for title, get in columns
    ]
    print("".join(title.ljust(width) for (title, _), width in zip(columns, widths)))
    print("-" * sum(widths))
    for entry in table:
        print("".join(get(entry).ljust(width)
                      for (_, get), width in zip(columns, widths)))
    print("\n* active policy. `changed?` compares the active verdict with the "
          "archived one.")
    print("\nsurface items passing, per scoring layer:")
    for entry in table:
        counts = " | ".join(
            f"{policy}: " + " ".join(
                f"L{layer}={count}" for layer, count in per_layer.items())
            for policy, per_layer in entry["surface_per_layer"].items()
        )
        flag = "  <-- policies disagree" if entry["policies_disagree"] else ""
        print(f"  {entry['name']:<14}{counts}{flag}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--manifest", action="store_true",
                        help="write MANIFEST.json with hashes and derived fields")
    parser.add_argument("--table", action="store_true",
                        help="print the archived / v1 / v2 verdict table")
    parser.add_argument("--tokenizer_report", action="store_true",
                        help="write tokenizer_alignment.json for the checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = Path(args.dir)
    if args.manifest:
        manifest = build_manifest(directory)
        (directory / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        print(f"wrote {directory / 'MANIFEST.json'} "
              f"({len(manifest['artifacts'])} artifacts)")
    if args.tokenizer_report:
        from dag_tasks import check_tokenizers

        report = check_tokenizers(
            n_items=5, n_decoys=DEFAULT_N_DECOYS, seed=0, checkpoints=CHECKPOINTS,
        )
        (directory / "tokenizer_alignment.json").write_text(
            json.dumps(report, indent=2)
        )
        print(f"wrote {directory / 'tokenizer_alignment.json'} "
              f"(aligned={report['aligned']})")
    if args.table:
        print_verdict_table(verdict_table(load_artifacts(directory)))


if __name__ == "__main__":
    main()
