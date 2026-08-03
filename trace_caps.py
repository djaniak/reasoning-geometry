"""Length-cap resolution shared by the trace-level analyses.

Every capped-trace count depends on a cap that matches the generation budget the
traces were actually collected under.  Two mistakes silently produce a count of
zero rather than an error, and both make a truncated population wear a cap-free
label:

* the cap is missing, so a guard of the form ``max_new_tokens and length >= cap``
  short-circuits and excludes nothing;
* the cap is copied from a different model (the per-model budgets in
  ``params.yaml`` differ by 8x), so no trace can reach it.

The second mistake is what produced a reported Qwen "cap-free" population of 498
against a true value of 392: the run was given DeepSeek's 8192 instead of Qwen's
1024.

The budget is *not* recorded alongside the traces -- ``collect_data.py`` writes
per-trace metadata only -- so it is recovered from the pipeline instead.  Two
records are authoritative:

``dvc.lock``
    the resolved ``collect_data.py`` command that produced the directory,
    i.e. what actually ran.  Absent for a collect that has not been committed.
``dvc.yaml`` + ``params.yaml``
    the declared budget for the stage whose outputs are the directory.  Present
    before the collect runs.

Disagreement between the two is an error, and so is a caller-supplied cap that
contradicts either.  Observed trace lengths are *not* evidence of a mismatch: a
budget no trace reaches is the normal shape of a clean collect (DeepSeek-Llama
runs at 12288 and may never hit it).  When neither record covers the directory
the cap cannot be validated at all; :class:`Cap` says so, and the reports carry
that label rather than implying the count was checked.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

_CAP_ARG = re.compile(r"--max_new_tokens[=\s]+(\d+)")
_ITEM_REF = re.compile(r"\$\{item\.([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class Cap:
    """A generation cap and the record it was validated against."""

    value: int
    sources: tuple[str, ...] = ()
    warning: str | None = None

    @property
    def verified(self) -> bool:
        """Whether an authoritative pipeline record confirmed this budget."""
        return bool(self.sources)

    @property
    def provenance(self) -> str:
        if not self.sources:
            return "caller-supplied; unvalidated (heuristic)"
        return "confirmed by " + ", ".join(self.sources)

    def __index__(self) -> int:  # lets a Cap be used wherever the int cap was
        return self.value

    def __int__(self) -> int:
        return self.value


def _repo_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / "dvc.yaml").is_file():
            return candidate
    return None


def _substitute(template: str, item: dict) -> str:
    """Fill ``${item.key}`` references; other ``${...}`` references are left as-is."""
    return _ITEM_REF.sub(
        lambda match: str(item[match.group(1)]) if match.group(1) in item else match.group(0),
        template,
    )


def _record(budgets: dict[Path, int], root: Path, out: str, cmd: str, source: str,
            conflicts: list[str]) -> None:
    cap = _CAP_ARG.search(cmd)
    if cap is None or "${" in out:
        return
    path = (root / out).resolve()
    value = int(cap.group(1))
    previous = budgets.setdefault(path, value)
    if previous != value:
        conflicts.append(f"{source} declares both {previous} and {value} for {out}")


def _lock_budgets(root: Path) -> tuple[dict[Path, int], list[str]]:
    lock = root / "dvc.lock"
    budgets: dict[Path, int] = {}
    conflicts: list[str] = []
    if not lock.is_file():
        return budgets, conflicts
    stages = (yaml.safe_load(lock.read_text()) or {}).get("stages") or {}
    for stage in stages.values():
        cmd = stage.get("cmd") or ""
        for out in stage.get("outs") or []:
            path = out.get("path") if isinstance(out, dict) else out
            if path:
                _record(budgets, root, str(path), cmd, "dvc.lock", conflicts)
    return budgets, conflicts


def _declared_budgets(root: Path) -> tuple[dict[Path, int], list[str]]:
    """Budgets from ``dvc.yaml``, resolving ``foreach`` items out of ``params.yaml``."""
    pipeline = root / "dvc.yaml"
    budgets: dict[Path, int] = {}
    conflicts: list[str] = []
    if not pipeline.is_file():
        return budgets, conflicts
    stages = (yaml.safe_load(pipeline.read_text()) or {}).get("stages") or {}
    params_path = root / "params.yaml"
    params = yaml.safe_load(params_path.read_text()) if params_path.is_file() else {}
    for stage in stages.values():
        if not isinstance(stage, dict):
            continue
        foreach = stage.get("foreach")
        body = stage.get("do", stage)
        if foreach is None:
            items: list[dict] = [{}]
        else:
            reference = _ITEM_REF.sub("", str(foreach)).strip()
            key = reference[2:-1] if reference.startswith("${") else reference
            resolved = (params or {}).get(key)
            if not isinstance(resolved, list):
                continue
            items = [item for item in resolved if isinstance(item, dict)]
        for item in items:
            cmd = _substitute(str(body.get("cmd") or ""), item)
            for out in body.get("outs") or []:
                path = out if isinstance(out, str) else next(iter(out), None)
                if path:
                    _record(budgets, root, _substitute(str(path), item), cmd,
                            "dvc.yaml/params.yaml", conflicts)
    return budgets, conflicts


def collection_budget(data_dir: str | Path) -> dict[str, int]:
    """Return ``{record name: budget}`` for every authoritative record of ``data_dir``.

    An empty mapping means no pipeline record covers the directory -- the usual
    case for data collected outside ``dvc repro`` -- not that the budget is zero.
    """
    directory = Path(data_dir).resolve()
    root = _repo_root(directory) or _repo_root(Path.cwd())
    if root is None:
        return {}
    found: dict[str, int] = {}
    for name, (budgets, conflicts) in (
        ("dvc.lock", _lock_budgets(root)),
        ("dvc.yaml/params.yaml", _declared_budgets(root)),
    ):
        if conflicts:
            raise ValueError(
                f"{name} is self-inconsistent about the generation budget: "
                + "; ".join(conflicts)
            )
        if directory in budgets:
            found[name] = budgets[directory]
    return found


def resolve_cap(
    max_new_tokens: int | None,
    *,
    data_dir: str | Path | None = None,
    lengths: Iterable[float | int | None] = (),
    context: str = "",
) -> Cap:
    """Return the generation cap, or raise if the records disagree about it.

    ``max_new_tokens`` is the caller's budget and ``data_dir`` the directory the
    traces were loaded from.  Either may be omitted, but not both.  When both are
    present they must agree: a caller passing another model's budget is the defect
    this guard exists for.

    ``lengths`` is used only when no authoritative record covers ``data_dir``.  A
    cap no observed trace reaches is then worth a warning -- it is the shape of
    both a clean collect and a wrong budget, and without a record the two cannot
    be told apart.
    """
    where = f" ({context})" if context else ""
    origin = repr(str(data_dir)) if data_dir is not None else "these traces (no data_dir given)"
    records = collection_budget(data_dir) if data_dir is not None else {}
    if len(set(records.values())) > 1:
        detail = ", ".join(f"{name}={value}" for name, value in sorted(records.items()))
        raise ValueError(
            f"the pipeline records disagree about the generation budget for "
            f"{origin}{where}: {detail}. Reconcile them before counting capped "
            "traces; the cap decides which population is called cap-free."
        )
    authoritative = next(iter(records.values()), None)

    if max_new_tokens is None and authoritative is None:
        raise ValueError(
            f"max_new_tokens is required to count capped traces{where}, and no "
            f"pipeline record covers {origin}. Pass the budget these traces "
            "were collected under (see the per-model max_new_tokens entries in "
            "params.yaml); without it no trace is ever counted as capped and a "
            "truncated population is labelled cap-free."
        )
    if max_new_tokens is None:
        return Cap(int(authoritative), tuple(records))

    cap = int(max_new_tokens)
    if authoritative is not None and cap != authoritative:
        raise ValueError(
            f"max_new_tokens={cap} contradicts the budget these traces were "
            f"collected under ({authoritative}, per {', '.join(records)}) for "
            f"{origin}{where}. A cap borrowed from another model counts no "
            "trace as capped and reports a truncated population as cap-free."
        )
    if authoritative is not None:
        return Cap(cap, tuple(records))

    observed = [int(value) for value in lengths if value is not None]
    warning = None
    if observed and max(observed) < cap:
        warning = (
            f"max_new_tokens={cap} exceeds every observed trace length "
            f"(max {max(observed)}){where} and no pipeline record covers "
            f"{origin}, so the cap is unvalidated: a clean collect and a cap "
            "borrowed from another model look identical here. No trace will be "
            "counted as capped."
        )
        warnings.warn(warning, stacklevel=2)
    return Cap(cap, (), warning)
