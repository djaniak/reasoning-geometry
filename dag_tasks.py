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

``depth`` is the length of the path from the edited node to the target. At depth 1
the target reads the edited node directly. At depth k the trace states k-1
intermediate results in between, and those tokens stay clean in the patched run.
So the depth ladder does not measure graph distance on its own: it measures
whether a patched state still moves the answer when a written intermediate value
contradicts it. That is the honest question for a written chain of thought, but it
is not "distance in the graph", and the numbers must not be read that way.

``gap`` is the control that separates depth from length. It puts that many decoy
lines between the chain and the target, which raises the token distance to the
read position without adding a step on the path. A drop at depth 2 that a
distance-matched gap at depth 1 does not reproduce is about the extra step, not
about the extra tokens. Each edit records its own ``distance_to_read``, so the two
arms are matched on the measured distance rather than on an assumed one.

This module holds no torch and no transformers import. It runs, and is tested,
on CPU with a fake character-level encoder.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
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

# `v1_unpaired` is the 2026-08-13 family: its chain draws from the main stream,
# so an item re-rolls entirely with depth and the ladder compares different
# families. It is kept reachable only because the archived artifacts are
# re-derived by regenerating their items. `v2_paired` is what new runs use.
GENERATORS = ("v1_unpaired", "v2_paired")
DEFAULT_GENERATOR = "v2_paired"

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

    kind: str  # ancestor | non_ancestor | surface_null | null | cross_item
    node: str | None
    positions: tuple[int, ...]
    token_ids: tuple[int, ...]
    implied_target_value: int
    distance_to_read: int  # tokens from the last patched position to read_position
    # Set only for ``cross_item``: which item in the batch the donor trace is,
    # and the value its ancestor line states. The second is the digit a patch
    # that merely copied the token it overwrote would produce, which is not the
    # same prediction as ``implied_target_value``.
    donor_item: int | None = None
    donor_raw_value: int | None = None


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
    depth: int = 1  # steps from the ancestor to the target
    gap: int = 0  # decoy lines between the chain and the target
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


def _sample_step(rng: random.Random, name: str, parent: str, value: int) -> Node:
    """One chain line ``name = parent op digit``, result still a single digit.

    The operand is at least 1 for the same reason as in a root line: a zero
    operand turns the step into a copy, and a chain of copies is not a chain.
    """
    options = [
        (op, rhs) for op in "+-" for rhs in range(1, 10)
        if 0 <= _apply(op, value, rhs) <= 9
    ]
    op, rhs = rng.choice(options)
    return Node(name, parent, op, str(rhs), _apply(op, value, rhs))


def _propagate(nodes: dict[str, Node], chain: tuple[str, ...], value: int) -> int:
    """Re-evaluate the ancestor-to-target path from a new ancestor value.

    Every value on the way must stay a single digit, or the edit would imply a
    trace the format cannot express.
    """
    for name in (*chain, TARGET):
        node = nodes[name]
        value = _apply(node.op, value, int(node.rhs))
        if not 0 <= value <= 9:
            raise Reject("edited chain value out of range")
    return value


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


def _sample_chain(rng: random.Random, n_steps: int, start: int,
                  donor_start: int, delta: int, tries: int = 200):
    """Signed increments along the ancestor-to-target path, one per step.

    Every step is ``value ± rhs``, so the path as a whole is the affine map
    ``v -> v + delta`` and the net delta is fixed by the spine: the clean
    ancestor value has to land on the target value. The donor value therefore
    lands on ``donor_start + delta`` at every depth, which is what makes the
    ancestor edit assert the same counterfactual whether it is one step away or
    three.

    Only the intermediate values are free, and they must stay single digits on
    both the clean and the donor path. That is the one constraint resolved here,
    drawn from the chain stream, so failing it can never shift the spine.
    """
    if n_steps == 1:
        return (delta,)
    for _ in range(tries):
        steps, clean, donor = [], start, donor_start
        for _ in range(n_steps - 1):
            options = [
                step for step in range(-9, 10)
                if step and 0 <= clean + step <= 9 and 0 <= donor + step <= 9
            ]
            if not options:
                break
            steps.append(rng.choice(options))
            clean += steps[-1]
            donor += steps[-1]
        else:
            last = delta - sum(steps)
            if last and -9 <= last <= 9:
                return (*steps, last)
    raise Reject("no chain realises the required net delta")


def _build_paired(rng: random.Random, encode, n_decoys: int, condition: str,
                  depth: int, gap: int | None) -> DagItem:
    """Build an item whose spine does not move with ``depth``.

    Depth-1 item *i* and depth-3 item *i* must be the same trace apart from the
    chain, or the ladder measures a between-family difference. Two things make
    that hold: the chain is built from its own stream, seeded by a single draw
    taken whatever the depth; and every quantity the chain could otherwise
    perturb -- the target value, the ancestor's donor line, the tag assignment,
    the surface edit's target lines -- is drawn from the main stream *before* the
    chain exists, with a count that does not depend on depth.
    """
    chain_rng = random.Random(rng.randrange(2 ** 32))

    nodes: dict[str, Node] = {}
    nodes[ANCESTOR] = _sample_root(rng, ANCESTOR)
    nodes[NON_ANCESTOR] = _sample_root(rng, NON_ANCESTOR)
    decoys = list(DECOY_NAMES[:n_decoys])
    for name in decoys:
        nodes[name] = _sample_root(rng, name)

    # Drawn against the ancestor rather than against whatever the chain last
    # produced, so it does not move with depth. At depth 1 they are the same
    # node, so this is the rule the original design used.
    start = nodes[ANCESTOR].value
    value_c = rng.choice([value for value in range(10) if value != start])

    # The ancestor's donor line is drawn here, not with the other edits, so the
    # chain can be sampled knowing it. Both rejections below are decided by the
    # spine alone and therefore fire identically at every depth -- which is the
    # point: a depth-dependent rejection would desynchronise the family just as
    # surely as a depth-dependent draw.
    ancestor_donor = _reroll_root(rng, nodes[ANCESTOR], condition)
    rerolled = ancestor_donor[1]
    implied = rerolled + (value_c - start)
    if not 0 <= implied <= 9:
        raise Reject("edited chain value out of range")
    if implied == value_c:
        raise Reject("edited target value unchanged")

    op_e = rng.choice("+-")
    value_e = _apply(op_e, value_c, nodes[NON_ANCESTOR].value)
    if not 0 <= value_e <= 9:
        raise Reject("merge value out of range")
    nodes[MERGE] = Node(MERGE, TARGET, op_e, NON_ANCESTOR, value_e)

    pair = [ANCESTOR, NON_ANCESTOR]
    rng.shuffle(pair)
    rng.shuffle(decoys)
    cut = rng.randrange(len(decoys) + 1)
    if gap is not None:
        cut = len(decoys) - gap

    # A fixed number of spine tags and two spares for the surface edit, both
    # drawn whatever the depth. The chain's tags come out of what is left over,
    # from the chain stream.
    spine = (*decoys[:cut], *pair, *decoys[cut:], TARGET, MERGE)
    spine_letters = rng.sample(TAG_POOL, len(spine))
    left_over = [letter for letter in TAG_POOL if letter not in spine_letters]
    spare_letters = rng.sample(left_over, 2)
    # Which two lines the surface edit rewrites is a spine decision as well.
    upstream = [name for name in spine if name not in (TARGET, MERGE)]
    edited_tags = rng.sample(upstream, 2)

    chain = tuple(DECOY_NAMES[-(depth - 1):]) if depth > 1 else ()
    steps = _sample_chain(chain_rng, depth, start, rerolled, value_c - start)
    parent, running = ANCESTOR, start
    for name, step in zip(chain, steps):
        running += step
        nodes[name] = Node(name, parent, "+" if step > 0 else "-",
                           str(abs(step)), running)
        parent = name
    nodes[TARGET] = Node(TARGET, parent, "+" if steps[-1] > 0 else "-",
                         str(abs(steps[-1])), value_c)
    chain_letters = chain_rng.sample(
        [letter for letter in left_over if letter not in spare_letters], len(chain)
    )

    order = (*decoys[:cut], *pair, *chain, *decoys[cut:], TARGET, MERGE)
    tags = dict(zip(spine, spine_letters)) | dict(zip(chain, chain_letters))

    chunks = _render(nodes, order, tags)
    ids, sites = encode_chunks(chunks, encode)
    read_position = sites[f"result:{TARGET}"] - 1

    edges = transitive_reduction(dependency_edges(nodes))
    if ancestors(edges, TARGET) != {ANCESTOR, *chain}:
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
        depth=depth,
        gap=len(decoys) - cut,
    )

    edits = [
        _value_edit(rng, item, nodes, order, tags, encode, ANCESTOR, "ancestor",
                    condition, chain, donor=ancestor_donor),
        _value_edit(rng, item, nodes, order, tags, encode, NON_ANCESTOR,
                    "non_ancestor", condition, chain),
        _tag_edit(rng, item, nodes, order, tags, encode,
                  names=edited_tags, letters=spare_letters),
    ]
    for name in decoys:
        edits.append(_value_edit(rng, item, nodes, order, tags, encode, name,
                                 "null", condition, chain))
    return dataclasses.replace(item, edits=tuple(edits))


def _build_unpaired(rng: random.Random, encode, n_decoys: int, condition: str,
                    depth: int, gap: int | None) -> DagItem:
    """The generator as it stood for the 2026-08-13 pilot. Frozen, not fixed.

    Its chain steps draw from the main stream, so every draw after them lands at
    a different position and the whole item re-rolls with depth. That is the bug
    ``_build_paired`` exists to fix. This path stays byte-exact because the three
    v0 artifacts are re-derived by regenerating their items -- see
    ``dag_evidence.reconstruct_items``.
    """
    nodes: dict[str, Node] = {}
    nodes[ANCESTOR] = _sample_root(rng, ANCESTOR)
    nodes[NON_ANCESTOR] = _sample_root(rng, NON_ANCESTOR)
    decoys = list(DECOY_NAMES[:n_decoys])
    for name in decoys:
        nodes[name] = _sample_root(rng, name)

    # Chain steps take names from the far end of the decoy pool, so depth 1 draws
    # exactly the nodes and the random numbers the original design drew.
    chain = tuple(DECOY_NAMES[-(depth - 1):]) if depth > 1 else ()
    parent, value_p = ANCESTOR, nodes[ANCESTOR].value
    for name in chain:
        nodes[name] = _sample_step(rng, name, parent, value_p)
        parent, value_p = name, nodes[name].value

    value_c = rng.choice([v for v in range(10) if v != value_p])
    op_c = "+" if value_c > value_p else "-"
    nodes[TARGET] = Node(TARGET, parent, op_c, str(abs(value_c - value_p)), value_c)

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
    # Always drawn, so fixing the gap does not shift the random stream.
    cut = rng.randrange(len(decoys) + 1)
    if gap is not None:
        cut = len(decoys) - gap
    order = tuple(decoys[:cut] + pair + list(chain) + decoys[cut:] + [TARGET, MERGE])

    tag_letters = rng.sample(TAG_POOL, len(order))
    tags = dict(zip(order, tag_letters))

    chunks = _render(nodes, order, tags)
    ids, sites = encode_chunks(chunks, encode)
    read_position = sites[f"result:{TARGET}"] - 1

    edges = transitive_reduction(dependency_edges(nodes))
    if ancestors(edges, TARGET) != {ANCESTOR, *chain}:
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
        depth=depth,
        gap=len(decoys) - cut,
    )

    edits = [_value_edit(rng, item, nodes, order, tags, encode, ANCESTOR, "ancestor",
                         condition, chain)]
    edits.append(
        _value_edit(rng, item, nodes, order, tags, encode, NON_ANCESTOR,
                    "non_ancestor", condition, chain)
    )
    edits.append(_tag_edit(rng, item, nodes, order, tags, encode))
    for name in decoys:
        edits.append(_value_edit(rng, item, nodes, order, tags, encode, name, "null",
                                 condition, chain))
    return dataclasses.replace(item, edits=tuple(edits))


def _finish_edit(item, nodes, order, tags, encode, kind, name, implied,
                 force_positions=None, donor_raw_value=None) -> Edit:
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
        donor_raw_value=donor_raw_value,
    )


def _value_edit(rng, item, nodes, order, tags, encode, name, kind, condition,
                chain, donor=None) -> Edit:
    edited = dict(nodes)
    # `donor` is the paired generator handing over a reroll it had to draw before
    # the chain existed; drawing it again here would move the stream.
    edited[name], rerolled = donor or _reroll_root(rng, nodes[name], condition)
    if name == ANCESTOR:
        implied = _propagate(nodes, chain, rerolled)
        if implied == item.target_value:
            raise Reject("edited target value unchanged")
    else:
        implied = item.target_value
    sites = (item.operand_positions[name], item.value_positions[name])
    # The digit standing at the patched result position in the donor trace, which
    # is what a readout that copies what it finds there would emit. Deliberately
    # not ``rerolled``: under ``operand_only`` the donor leaves the result token
    # alone, so copying predicts no movement while the implied value predicts a
    # changed answer. Separating the two is the point -- at depth 1 the ancestor
    # gate cannot otherwise tell "propagated the value" from "copied the digit".
    return _finish_edit(item, edited, order, tags, encode, kind, name, implied,
                        force_positions=tuple(sorted(sites)),
                        donor_raw_value=edited[name].value)


def _tag_edit(rng, item, nodes, order, tags, encode, names=None,
              letters=None) -> Edit:
    """Surface null: replace two line tags with unused letters.

    Two positions, to match the token count of a value edit. Line tags carry no
    computational role, so a faithful model should not move.

    ``names`` and ``letters`` are the paired generator's: both the lines to
    rewrite and the spare letters have to be chosen from the spine, before the
    chain adds lines and consumes tags, or the surface edit lands somewhere else
    at every depth.
    """
    if names is None or letters is None:
        spare = [letter for letter in TAG_POOL if letter not in tags.values()]
        if len(spare) < 2:
            raise Reject("not enough spare tag letters")
        upstream = [name for name in order
                    if item.value_positions[name] < item.read_position]
        names, letters = rng.sample(upstream, 2), rng.sample(spare, 2)
    edited = dict(tags)
    for name, letter in zip(names, letters):
        edited[name] = letter
    return _finish_edit(item, nodes, order, edited, encode, "surface_null", None,
                        item.target_value)


# --------------------------------------------------------------------------
# cross-item donors
#
# Every other edit rewrites the recipient's own trace, which leaves the sharpest
# objection to the ancestor gap standing: those two token positions might simply
# be perturbation-sensitive, and any state written there would move the readout.
# The cross-item donor is the matched test of that. It writes *another item's*
# residual state at the same positions -- same span, same width, same formatting
# -- so the only thing that varies is which item the state came from.
#
# It also makes a sharp prediction. The chain is affine, so a donor value ``v_j``
# read through recipient ``i``'s chain implies ``v_j + delta_i``: not the clean
# answer, and not the donor's own digit either. Those three digits being distinct
# is what separates "propagated the donor's value" from "copied the patched
# token" from "did not move", so the batch is selected to keep them distinct.
#
# The donor is the other item's *clean* trace, so the donor condition does not
# apply to this edit: there is nothing rerolled to state falsely.
# --------------------------------------------------------------------------

# Candidates sampled per requested item, and how much of the matched group the
# subset search looks at. Both only widen the search; neither selects on
# anything the measurement reads.
CROSS_ITEM_OVERSAMPLE = 8
CROSS_ITEM_SEARCH = 3


def _ancestor_sites(item: DagItem) -> tuple[int, ...]:
    return tuple(sorted((item.operand_positions[ANCESTOR],
                         item.value_positions[ANCESTOR])))


def _implied_by_donor(recipient: DagItem, donor: DagItem) -> int | None:
    """Target value the donor's ancestor state implies in ``recipient``.

    The chain is the affine map ``v -> v + delta``, and under ``v2_paired``
    ``delta`` is fixed by the spine: both the target value and the ancestor value
    are drawn before the chain exists. So this reads the net delta off the spine
    instead of walking the chain, and eligibility does not move with depth.

    That is deliberate, and it is the same rule that fixed the ladder. Walking
    the chain with ``_propagate`` would also reject a pair whose *intermediate*
    values leave 0..9, which is a depth-dependent condition -- it would hand a
    different donor map to depth 1 and depth 3 and desynchronise the arm exactly
    as the unpaired generator did. (Those intermediates are unconstrained for an
    arbitrary donor value, so at depth > 1 the prediction assumes the model
    carries a value the written chain never states. At depth 1 no intermediate
    exists and the question does not arise.)

    ``None`` means the pair is unusable: the counterfactual is not a digit, or it
    lands back on the clean answer and there is nothing for the readout to show.
    Self-donation fails the second test by construction.
    """
    implied = (donor.nodes[ANCESTOR].value
               + recipient.target_value - recipient.nodes[ANCESTOR].value)
    if not 0 <= implied <= 9 or implied == recipient.target_value:
        return None
    return implied


def _perfect_matching(n: int, allowed: list[list[int]]) -> list[int] | None:
    """A permutation with ``sigma[i]`` drawn from ``allowed[i]``, or ``None``.

    Kuhn's augmenting-path search. ``allowed[i]`` never contains ``i``, so any
    perfect matching it returns is fixed-point-free -- the derangement the
    control needs, rather than a permutation that has to be rejected afterwards.
    """
    donor_to_recipient = [-1] * n

    def augment(recipient: int, seen: list[bool]) -> bool:
        for donor in allowed[recipient]:
            if seen[donor]:
                continue
            seen[donor] = True
            if (donor_to_recipient[donor] == -1
                    or augment(donor_to_recipient[donor], seen)):
                donor_to_recipient[donor] = recipient
                return True
        return False

    for recipient in range(n):
        if not augment(recipient, [False] * n):
            return None
    sigma = [-1] * n
    for donor, recipient in enumerate(donor_to_recipient):
        sigma[recipient] = donor
    return sigma


def _select_cross_item_batch(candidates: list[DagItem], n_items: int):
    """Pick ``n_items`` mutually donatable candidates, and who donates to whom.

    Two selections happen here and they are deliberately different in kind. The
    first groups candidates by where the ancestor line sits, so donor and
    recipient share the patched positions -- that is formatting, and nothing the
    readout measures depends on it. The second keeps a subset whose members can
    donate to each other at all, which depends on the sampled values; it is a
    constraint of the ten-way digit readout, not a preference over outcomes, but
    it does mean this arm is not the same value distribution as the ladder.
    Earliest-first order keeps both deterministic and as close to a free sample
    as the constraint allows.
    """
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, item in enumerate(candidates):
        groups.setdefault(_ancestor_sites(item), []).append(index)
    group = max(groups.values(), key=lambda members: (len(members), -members[0]))

    for subset in itertools.combinations(group[:CROSS_ITEM_SEARCH * n_items],
                                         n_items):
        allowed = [
            [slot for slot, donor in enumerate(subset)
             if _implied_by_donor(candidates[recipient],
                                  candidates[donor]) is not None]
            for recipient in subset
        ]
        sigma = _perfect_matching(n_items, allowed)
        if sigma is not None:
            return [candidates[index] for index in subset], sigma
    raise ValueError(
        f"no derangement over {n_items} of {len(candidates)} candidates: the "
        f"largest position-matched group has {len(group)} members and no "
        f"{n_items} of them can all donate to one another. Raise oversample."
    )


def _attach_cross_item_edits(items: list[DagItem], sigma: list[int]):
    attached = []
    for index, item in enumerate(items):
        donor = items[sigma[index]]
        sites = _ancestor_sites(item)
        edit = Edit(
            kind="cross_item",
            node=ANCESTOR,
            positions=sites,
            token_ids=donor.token_ids,
            implied_target_value=_implied_by_donor(item, donor),
            distance_to_read=item.read_position - max(sites),
            donor_item=sigma[index],
            donor_raw_value=donor.nodes[ANCESTOR].value,
        )
        attached.append(dataclasses.replace(item, edits=(*item.edits, edit)))
    return attached


def generate_items(encode, *, n_items: int = 5, n_decoys: int = 6, seed: int = 0,
                   condition: str = "both", depth: int = 1, gap: int | None = None,
                   generator: str = DEFAULT_GENERATOR,
                   cross_item: bool = False, oversample: int | None = None):
    """Sample ``n_items`` DAG items. ``encode`` maps text to a list of token ids.

    ``condition`` selects the donor condition for every value edit in the batch,
    so one run measures one condition against its own null spread. ``depth`` and
    ``gap`` set the ancestor-to-target path length and the decoy padding in front
    of the target; see the module docstring for why both are needed. ``gap``
    defaults to a random split of the decoys around the edited pair.

    ``generator`` picks the item family. ``v2_paired`` is the default and the
    only one to run new experiments on: item *i* is the same trace at every
    depth apart from its chain. ``v1_unpaired`` is the 2026-08-13 family, kept
    reachable only so the archived artifacts stay re-derivable; its depth arms
    are different families and cannot be compared item by item.

    ``cross_item`` adds the cross-item donor control, which needs the batch to
    be mutually donatable. That is a constraint on which sampled items are kept,
    so it changes the batch: a cross-item run is its own arm and its numbers are
    not the ladder's. ``oversample`` is how many candidates to sample from.
    """
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}, "
                         f"expected one of {GENERATORS}")
    if condition not in DONOR_CONDITIONS:
        raise ValueError(f"unknown donor condition {condition!r}, "
                         f"expected one of {DONOR_CONDITIONS}")
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    # Chain steps consume decoy names, every line consumes a line tag, and the
    # surface-null edit needs two tags left over.
    budget = min(len(DECOY_NAMES) + 1, len(TAG_POOL) - 5)
    if n_decoys + depth > budget:
        raise ValueError(f"depth {depth} with {n_decoys} decoys does not fit the "
                         f"name and tag pools; keep n_decoys + depth <= {budget}")
    if gap is not None and not 0 <= gap <= n_decoys:
        raise ValueError(f"gap must be between 0 and n_decoys ({n_decoys}), got {gap}")
    build = _build_paired if generator == "v2_paired" else _build_unpaired
    # Sampling a wider pool and then selecting is what lets the cross-item batch
    # be mutually donatable. Without the control the pool is the batch, so the
    # ladder's items are untouched by any of this.
    n_sampled = (oversample or CROSS_ITEM_OVERSAMPLE * n_items) if cross_item \
        else n_items
    rng = random.Random(seed)
    items = []
    reasons: dict[str, int] = {}
    while len(items) < n_sampled:
        for _ in range(2000):
            try:
                items.append(build(rng, encode, n_decoys, condition, depth, gap))
                break
            except Reject as reject:
                reasons[str(reject)] = reasons.get(str(reject), 0) + 1
        else:
            ranked = sorted(reasons.items(), key=lambda pair: -pair[1])
            detail = "; ".join(f"{count}x {reason}" for reason, count in ranked[:5])
            raise RuntimeError(
                f"could not sample a valid DAG item after 2000 tries: {detail}"
            )
    if cross_item:
        items, sigma = _select_cross_item_batch(items, n_items)
        items = _attach_cross_item_edits(items, sigma)
    return items


# --------------------------------------------------------------------------
# tokenizer precondition
# --------------------------------------------------------------------------


def check_tokenizers(*, n_items: int, n_decoys: int, seed: int, checkpoints,
                     condition: str = "both", depth: int = 1,
                     gap: int | None = None,
                     generator: str = DEFAULT_GENERATOR) -> dict:
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
            depth=depth,
            gap=gap,
            generator=generator,
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
        "generator": generator,
        "depth": depth,
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
    parser.add_argument("--depth", type=int, default=1,
                        help="steps from the edited ancestor to the target")
    parser.add_argument("--gap", type=int, default=None,
                        help="decoy lines between the chain and the target "
                             "(default: a random split)")
    parser.add_argument("--generator", choices=GENERATORS,
                        default=DEFAULT_GENERATOR,
                        help="item family; v1_unpaired is the archived one "
                             "and is not paired across depth")
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
            depth=args.depth,
            gap=args.gap,
            generator=args.generator,
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
        depth=args.depth,
        gap=args.gap,
        generator=args.generator,
    )
    for item in items:
        print(item.text)
        print(f"target {item.target} = {item.target_value}, "
              f"read_position {item.read_position}, depth {item.depth}, "
              f"gap {item.gap}, edges {item.edges}")
        for edit in item.edits:
            print(f"  {edit.kind:<13} node={edit.node} pos={edit.positions} "
                  f"implied={edit.implied_target_value} dist={edit.distance_to_read}")
        print()


if __name__ == "__main__":
    main()
