"""Arithmetic DAG items with a known dependency graph, for causal patching.

Prototype 1 of the ground-truth causal-DAG fidelity direction (see
``outputs/reviews/2026-08-12-consolidated-research-directions.md``, section D).

Every node value is a single digit. That is not cosmetic:

* the pre-registered edit-matching rule (digit length, token count, position
  class) holds by construction instead of by hand-checking;
* clean and donor traces stay token-aligned with equal length, so a residual
  state taken at position ``p`` in the donor belongs at position ``p`` in the
  clean run;
* the read-out is a log-odds over a fixed ten-way answer set.

A value edit rewrites one node's line. Which part of the line it rewrites is the
donor condition, and it decides what the measurement can mean:

* ``both`` -- ``a = 3 + 4 = 7`` becomes ``a = 3 + 5 = 8``. Consistent, but the
  two mechanisms are stuck together.
* ``result_only`` -- ``a = 3 + 4 = 8``. Only a model that reads the stated value
  moves.
* ``operand_only`` -- ``a = 3 + 5 = 7``. Only a model that recomputes from the
  operands moves.

The last two are arithmetically inconsistent on purpose. That is not the old
arithmetic-surprise confound, because all three conditions carry the same implied
target value, and the reported statistic is directional: a signal that only says
"something is wrong here" does not move the answer toward that value.

Every condition patches the same two token positions -- the operand digit and the
result digit of the edited node -- so the conditions differ in what the donor text
says, never in how many residual states are written.

This module holds no torch and no transformers import. It runs, and is tested,
on CPU with a fake character-level encoder.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from dataclasses import dataclass, field

# Node names. Variables and line tags come from disjoint pools so that a tag
# edit can never be read as a variable reference.
TARGET = "c"
ANCESTOR = "a"
NON_ANCESTOR = "b"
MERGE = "e"
DECOY_NAMES = "fghijlmn"
TAG_POOL = "dkopqrstuvwxyz"

CHECKPOINTS = (
    "Qwen/Qwen2.5-Math-1.5B",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
)

# Which part of the edited line the donor rewrites. See the module docstring.
DONOR_CONDITIONS = ("both", "result_only", "operand_only")


class Reject(Exception):
    """A sampled candidate violated an invariant; resample."""


@dataclass(frozen=True)
class Node:
    """One line of the trace: ``name = lhs op rhs = value``."""

    name: str
    lhs: str  # node name or single digit
    op: str  # "+" or "-"
    rhs: str  # single digit, or node name for the merge
    value: int


@dataclass(frozen=True)
class Edit:
    """One donor trace, differing from the clean trace at ``positions`` only.

    ``implied_target_value`` is the value the target takes under this edit. For
    every kind except ``ancestor`` that is the clean target value, because a
    faithful model should not move at all.
    """

    kind: str  # ancestor | non_ancestor | surface_null | null
    node: str | None
    positions: tuple[int, ...]
    token_ids: tuple[int, ...]
    implied_target_value: int
    distance_to_read: int  # tokens from the last patched position to read_position


@dataclass(frozen=True)
class DagItem:
    nodes: dict[str, Node]
    order: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]  # transitive reduction, parent -> child
    target: str
    text: str
    token_ids: tuple[int, ...]
    value_positions: dict[str, int]  # node -> token index of its result digit
    operand_positions: dict[str, int]  # node -> token index of its rhs digit
    read_position: int
    condition: str = "both"
    edits: tuple[Edit, ...] = field(default_factory=tuple)

    @property
    def target_value(self) -> int:
        return self.nodes[self.target].value


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------


def dependency_edges(nodes: dict[str, Node]) -> set[tuple[str, str]]:
    """Direct operand references, parent -> child."""
    edges = set()
    for node in nodes.values():
        for operand in (node.lhs, node.rhs):
            if operand in nodes:
                edges.add((operand, node.name))
    return edges


def reachable(edges: set[tuple[str, str]], source: str) -> set[str]:
    out, stack = set(), [source]
    while stack:
        current = stack.pop()
        for parent, child in edges:
            if parent == current and child not in out:
                out.add(child)
                stack.append(child)
    return out


def transitive_reduction(edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Drop every edge implied by a longer path. The DAG ground truth."""
    keep = set()
    for parent, child in edges:
        others = edges - {(parent, child)}
        if child not in reachable(others, parent):
            keep.add((parent, child))
    return keep


def ancestors(edges: set[tuple[str, str]], node: str) -> set[str]:
    reverse = {(child, parent) for parent, child in edges}
    return reachable(reverse, node)


# --------------------------------------------------------------------------
# rendering and tokenization
# --------------------------------------------------------------------------


def _render(nodes: dict[str, Node], order: tuple[str, ...], tags: dict[str, str]):
    """Build the trace as (text, site) chunks, one line per node.

    ``site`` is ``None`` for structural text, or a key naming a patchable
    single-token position. Chunking is what gives exact position knowledge;
    ``encode_chunks`` verifies it against whole-string tokenization.

    Chunk boundaries are not free. The Qwen tokenizers split a space from a
    following digit but merge it with a following letter, and merge ``"["`` with
    the letter after it. So digit sites are bare with the space left in the
    preceding chunk, and the line tag carries its own leading space and sits at
    the end of the line, where nothing can merge into it.
    """
    chunks: list[tuple[str, str | None]] = []
    for name in order:
        node = nodes[name]
        chunks.append((f"{node.name} = {node.lhs} {node.op} ", None))
        if node.rhs.isdigit():
            chunks.append((node.rhs, f"operand:{name}"))
        else:
            # The merge node's operand is a variable, never edited, so it needs
            # no site of its own -- and a name must keep its leading space.
            chunks[-1] = (chunks[-1][0].rstrip() + f" {node.rhs}", None)
        chunks.append((" = ", None))
        chunks.append((str(node.value), f"result:{name}"))
        chunks.append((" #", None))
        chunks.append((f" {tags[name]}", f"tag:{name}"))
        chunks.append(("\n", None))
    return chunks


def encode_chunks(chunks, encode):
    """Return (token_ids, {site: position}).

    Rejects if a patchable chunk is not exactly one token, or if chunk-wise
    tokenization disagrees with tokenizing the whole string. Both would silently
    misplace every patch.
    """
    # Coalesce runs of structural text. Only boundaries at patchable sites can
    # then disagree with whole-string tokenization, which is the smallest
    # surface the position bookkeeping can be exposed to.
    merged: list[tuple[str, str | None]] = []
    for text, site in chunks:
        if site is None and merged and merged[-1][1] is None:
            merged[-1] = (merged[-1][0] + text, None)
        else:
            merged.append((text, site))

    ids: list[int] = []
    sites: dict[str, int] = {}
    for text, site in merged:
        piece = list(encode(text))
        if site is not None:
            if len(piece) != 1:
                raise Reject(f"site {site!r} is {len(piece)} tokens, not 1")
            sites[site] = len(ids)
        ids.extend(piece)
    whole = list(encode("".join(text for text, _ in merged)))
    if whole != ids:
        raise Reject("chunk-wise tokenization disagrees with whole-string")
    return ids, sites


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def _apply(op: str, left: int, right: int) -> int:
    return left + right if op == "+" else left - right


def _sample_root(rng: random.Random, name: str) -> Node:
    """Sample the value first, then operands that produce it.

    Sampling operands first and rejecting out-of-range results biases heavily
    toward a zero operand, which turns the step into a copy rather than a
    computation. Both operands are therefore forced to be at least 1.
    """
    value = rng.randrange(2, 9)  # 2..8 leaves room for two non-zero operands
    op = rng.choice("+-")
    if op == "+":
        left = rng.randrange(1, value)
        right = value - left
    else:
        right = rng.randrange(1, 10 - value)
        left = value + right
    return Node(name, str(left), op, str(right), value)


def _reroll_root(rng: random.Random, node: Node, condition: str) -> tuple[Node, int]:
    """Donor version of one root line, plus the value the reroll implies.

    The new operand and the value it computes to are sampled identically in every
    condition; the condition only decides which of the two the donor line shows.
    ``result_only`` and ``operand_only`` therefore state something arithmetically
    false, which is the point: each leaves exactly one of the two mechanisms able
    to move the answer.

    The implied value is returned separately because under ``operand_only`` the
    rendered ``node.value`` is still the clean one.
    """
    left = int(node.lhs)
    options = [
        right for right in range(1, 10)
        if 0 <= _apply(node.op, left, right) <= 9
        and _apply(node.op, left, right) != node.value
    ]
    if not options:
        raise Reject("no alternative root value")
    right = rng.choice(options)
    value = _apply(node.op, left, right)
    if condition == "result_only":
        return dataclasses.replace(node, value=value), value
    if condition == "operand_only":
        return dataclasses.replace(node, rhs=str(right)), value
    return dataclasses.replace(node, rhs=str(right), value=value), value


def _build(rng: random.Random, encode, n_decoys: int, condition: str) -> DagItem:
    nodes: dict[str, Node] = {}
    nodes[ANCESTOR] = _sample_root(rng, ANCESTOR)
    nodes[NON_ANCESTOR] = _sample_root(rng, NON_ANCESTOR)
    decoys = list(DECOY_NAMES[:n_decoys])
    for name in decoys:
        nodes[name] = _sample_root(rng, name)

    value_a = nodes[ANCESTOR].value
    value_c = rng.choice([v for v in range(10) if v != value_a])
    op_c = "+" if value_c > value_a else "-"
    nodes[TARGET] = Node(TARGET, ANCESTOR, op_c, str(abs(value_c - value_a)), value_c)

    op_e = rng.choice("+-")
    value_e = _apply(op_e, value_c, nodes[NON_ANCESTOR].value)
    if not 0 <= value_e <= 9:
        raise Reject("merge value out of range")
    nodes[MERGE] = Node(MERGE, TARGET, op_e, NON_ANCESTOR, value_e)

    # The ancestor and the matched non-ancestor sit on adjacent lines, so their
    # token distance to the read position differs by one line, the smallest
    # possible gap between two distinct steps. Their order is random, so the
    # residual mismatch does not point the same way in every item.
    pair = [ANCESTOR, NON_ANCESTOR]
    rng.shuffle(pair)
    rng.shuffle(decoys)
    cut = rng.randrange(len(decoys) + 1)
    order = tuple(decoys[:cut] + pair + decoys[cut:] + [TARGET, MERGE])

    tag_letters = rng.sample(TAG_POOL, len(order))
    tags = dict(zip(order, tag_letters))

    chunks = _render(nodes, order, tags)
    ids, sites = encode_chunks(chunks, encode)
    read_position = sites[f"result:{TARGET}"] - 1

    edges = transitive_reduction(dependency_edges(nodes))
    if ancestors(edges, TARGET) != {ANCESTOR}:
        raise Reject("unexpected ancestor set")

    item = DagItem(
        nodes=nodes,
        order=order,
        edges=tuple(sorted(edges)),
        target=TARGET,
        text="".join(text for text, _ in chunks),
        token_ids=tuple(ids),
        value_positions={name: sites[f"result:{name}"] for name in order},
        operand_positions={
            name: sites[f"operand:{name}"] for name in order
            if f"operand:{name}" in sites
        },
        read_position=read_position,
        condition=condition,
    )

    edits = [_value_edit(rng, item, nodes, order, tags, encode, ANCESTOR, "ancestor",
                         condition)]
    edits.append(
        _value_edit(rng, item, nodes, order, tags, encode, NON_ANCESTOR,
                    "non_ancestor", condition)
    )
    edits.append(_tag_edit(rng, item, nodes, order, tags, encode))
    for name in decoys:
        edits.append(_value_edit(rng, item, nodes, order, tags, encode, name, "null",
                                 condition))
    return dataclasses.replace(item, edits=tuple(edits))


def _finish_edit(item, nodes, order, tags, encode, kind, name, implied,
                 force_positions=None) -> Edit:
    ids, _ = encode_chunks(_render(nodes, order, tags), encode)
    if len(ids) != len(item.token_ids):
        raise Reject("donor length differs from clean")
    differing = tuple(i for i, (x, y) in enumerate(zip(ids, item.token_ids)) if x != y)
    if not differing:
        raise Reject("donor identical to clean")
    # Value edits declare both value sites whatever the condition changed, so
    # every condition writes the same number of residual states. One of the two
    # is then an identity patch, which the identity check already shows is inert.
    positions = tuple(force_positions) if force_positions else differing
    if not set(differing) <= set(positions):
        raise Reject("donor differs outside the declared positions")
    if max(positions) >= item.read_position:
        raise Reject("edit is not upstream of the read position")
    return Edit(
        kind=kind,
        node=name,
        positions=positions,
        token_ids=tuple(ids),
        implied_target_value=implied,
        distance_to_read=item.read_position - max(positions),
    )


def _value_edit(rng, item, nodes, order, tags, encode, name, kind, condition) -> Edit:
    edited = dict(nodes)
    edited[name], rerolled = _reroll_root(rng, nodes[name], condition)
    if name == ANCESTOR:
        target = nodes[TARGET]
        implied = _apply(target.op, rerolled, int(target.rhs))
        if not 0 <= implied <= 9 or implied == item.target_value:
            raise Reject("edited target value out of range or unchanged")
    else:
        implied = item.target_value
    sites = (item.operand_positions[name], item.value_positions[name])
    return _finish_edit(item, edited, order, tags, encode, kind, name, implied,
                        force_positions=tuple(sorted(sites)))


def _tag_edit(rng, item, nodes, order, tags, encode) -> Edit:
    """Surface null: replace two line tags with unused letters.

    Two positions, to match the token count of a value edit. Line tags carry no
    computational role, so a faithful model should not move.
    """
    spare = [letter for letter in TAG_POOL if letter not in tags.values()]
    if len(spare) < 2:
        raise Reject("not enough spare tag letters")
    upstream = [name for name in order if item.value_positions[name] < item.read_position]
    edited = dict(tags)
    for name, letter in zip(rng.sample(upstream, 2), rng.sample(spare, 2)):
        edited[name] = letter
    return _finish_edit(item, nodes, order, edited, encode, "surface_null", None,
                        item.target_value)


def generate_items(encode, *, n_items: int = 5, n_decoys: int = 6, seed: int = 0,
                   condition: str = "both"):
    """Sample ``n_items`` DAG items. ``encode`` maps text to a list of token ids.

    ``condition`` selects the donor condition for every value edit in the batch,
    so one run measures one condition against its own null spread.
    """
    if condition not in DONOR_CONDITIONS:
        raise ValueError(f"unknown donor condition {condition!r}, "
                         f"expected one of {DONOR_CONDITIONS}")
    rng = random.Random(seed)
    items = []
    reasons: dict[str, int] = {}
    while len(items) < n_items:
        for _ in range(2000):
            try:
                items.append(_build(rng, encode, n_decoys, condition))
                break
            except Reject as reject:
                reasons[str(reject)] = reasons.get(str(reject), 0) + 1
        else:
            ranked = sorted(reasons.items(), key=lambda pair: -pair[1])
            detail = "; ".join(f"{count}x {reason}" for reason, count in ranked[:5])
            raise RuntimeError(
                f"could not sample a valid DAG item after 2000 tries: {detail}"
            )
    return items


# --------------------------------------------------------------------------
# tokenizer precondition
# --------------------------------------------------------------------------


def check_tokenizers(*, n_items: int, n_decoys: int, seed: int, checkpoints,
                     condition: str = "both") -> dict:
    """Stop condition from the pre-registration.

    All three checkpoints must tokenize the same trace into the same id
    sequence. If they do not, token positions and layer bins do not align and
    the Base/Instruct/Distill comparison compares different objects.
    """
    from transformers import AutoTokenizer

    tokenizers = {
        name: AutoTokenizer.from_pretrained(name) for name in checkpoints
    }
    per_checkpoint = {}
    for name, tokenizer in tokenizers.items():
        items = generate_items(
            lambda text: tokenizer.encode(text, add_special_tokens=False),
            n_items=n_items,
            n_decoys=n_decoys,
            seed=seed,
            condition=condition,
        )
        per_checkpoint[name] = items

    reference = checkpoints[0]
    mismatches = []
    for name in checkpoints[1:]:
        for index, (left, right) in enumerate(
            zip(per_checkpoint[reference], per_checkpoint[name])
        ):
            if left.token_ids != right.token_ids:
                mismatches.append({"item": index, "checkpoint": name})
    return {
        "checkpoints": list(checkpoints),
        "n_items": n_items,
        "reference": reference,
        "n_tokens": [len(item.token_ids) for item in per_checkpoint[reference]],
        "mismatches": mismatches,
        "aligned": not mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check_tokenizers", action="store_true",
                        help="verify the three checkpoints agree; exit 1 if not")
    parser.add_argument("--n_items", type=int, default=5)
    parser.add_argument("--n_decoys", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", choices=DONOR_CONDITIONS, default="both")
    parser.add_argument("--model_name", default=CHECKPOINTS[0],
                        help="tokenizer used when printing sample items")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_tokenizers:
        report = check_tokenizers(
            n_items=args.n_items,
            n_decoys=args.n_decoys,
            seed=args.seed,
            checkpoints=CHECKPOINTS,
            condition=args.condition,
        )
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["aligned"] else 1)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    items = generate_items(
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        n_items=args.n_items,
        n_decoys=args.n_decoys,
        seed=args.seed,
        condition=args.condition,
    )
    for item in items:
        print(item.text)
        print(f"target {item.target} = {item.target_value}, "
              f"read_position {item.read_position}, edges {item.edges}")
        for edit in item.edits:
            print(f"  {edit.kind:<13} node={edit.node} pos={edit.positions} "
                  f"implied={edit.implied_target_value} dist={edit.distance_to_read}")
        print()


if __name__ == "__main__":
    main()
