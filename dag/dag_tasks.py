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

The depth ladder patches the ancestor and leaves every written intermediate
alone, so a null result at depth 2 has two readings: the model did not carry the
value across a step, or it did not carry it across those tokens. ``chain_edits``
adds the missing arm. It patches the intermediate itself -- same two positions,
same donor arithmetic, same clean readout, one step from the target instead of
``depth`` -- so the ancestor and the intermediate are compared *within one item*
rather than across arms. Step distance and token distance still fall together
between the two sites, which is what the gap arms are for; each edit records both
``steps_to_target`` and ``distance_to_read`` so neither has to be assumed.

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
GENERATORS = ("v1_unpaired", "v2_paired", "v3_distinct")
DEFAULT_GENERATOR = "v3_distinct"

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
    every kind except ``ancestor`` and ``chain`` that is the clean target value,
    because a faithful model should not move at all.
    """

    kind: str  # ancestor | chain | non_ancestor | surface_null | null | cross_item
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
    # Steps along the dependency path from the edited node to the target, for the
    # two kinds that sit on that path: ``depth`` for ``ancestor``, and one fewer
    # per chain line already crossed for ``chain``. ``distance_to_read`` is the
    # token distance and answers a different question; the chain contrast needs
    # both, since the two move together as depth grows and the gap arms are what
    # tell them apart.
    steps_to_target: int | None = None


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
    # Chain nodes whose result the trace does not state, and the token-matched
    # filler standing in for each. Empty is the written format; at depth 1 there
    # is nothing between the ancestor and the target, so it is empty either way.
    omit_pad: dict[str, str] = field(default_factory=dict)
    edits: tuple[Edit, ...] = field(default_factory=tuple)

    @property
    def omit(self) -> tuple[str, ...]:
        return tuple(name for name in self.order if name in self.omit_pad)

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


def _render(nodes: dict[str, Node], order: tuple[str, ...], tags: dict[str, str],
            omit_pad: dict[str, str] | None = None):
    """Build the trace as (text, site) chunks, one line per node.

    ``site`` is ``None`` for structural text, or a key naming a patchable
    single-token position. Chunking is what gives exact position knowledge;
    ``encode_chunks`` verifies it against whole-string tokenization.

    Chunk boundaries are not free. The Qwen tokenizers split a space from a
    following digit but merge it with a following letter, and merge ``"["`` with
    the letter after it. So digit sites are bare with the space left in the
    preceding chunk, and the line tag carries its own leading space and sits at
    the end of the line, where nothing can merge into it.

    ``omit_pad`` names the lines that state no result, each mapped to the filler
    standing in for its ``" = <digit>"``. The line still defines the node, so the
    graph is unchanged and the value is computable; it is simply not written
    down. The filler goes *before* the tag, not after: the tag has to stay last
    on the line or a trailing marker merges with the newline and the token count
    stops matching the written format.
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
        if omit_pad and name in omit_pad:
            chunks.append((" #" + omit_pad[name], None))
        else:
            chunks.append((" = ", None))
            chunks.append((str(node.value), f"result:{name}"))
            chunks.append((" #", None))
        chunks.append((f" {tags[name]}", f"tag:{name}"))
        chunks.append(("\n", None))
    return chunks


# What stands in for an omitted ``" = <digit>"``: repeated comment markers,
# which cannot be read as a variable name or a tag and which the surface control
# already shows are inert. The repeat count is solved against the tokenizer
# rather than fixed, because " = " is two tokens under the Qwen vocabularies and
# three under a character tokenizer.
OMITTED_VALUE_MARKER = " #"

# Which lines state no result. `chain` is the experiment -- the values between
# the ancestor and the target; `decoy` is its control, the same count of values
# from lines the target does not depend on.
OMIT_MODES = ("none", "chain", "decoy")


def _omission_pad(encode, value: str) -> str:
    """Filler matching the token count of the ``" = <digit>"`` it replaces.

    Token-matched or nothing: a pad that is a token short would shift every
    position downstream and turn the written/omitted contrast back into the
    token-distance confound the paired ladder exists to remove.
    """
    removed = len(list(encode(" = "))) + len(list(encode(value)))
    for repeat in range(1, removed + 1):
        pad = OMITTED_VALUE_MARKER * repeat
        if len(list(encode(pad))) == removed:
            return pad
    raise Reject("no comment pad matches the omitted value's token count")


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


def _chain_donor_options(start: int, steps: tuple[int, ...], index: int,
                         distinct: bool, target_value: int
                         ) -> list[tuple[int, int, int]]:
    """``(rhs, value, implied)`` a chain line can be rerolled to, at ``index``.

    ``steps`` is the whole ancestor-to-target path, one signed increment per
    line; the chain line at ``index`` is written by ``steps[index]`` and reaches
    the target through ``steps[index + 1:]``, which is never empty because the
    target's own step is in it.

    The operator is not a candidate for rerolling. It sits at no declared patch
    position, so a donor that flipped it would differ outside the two sites every
    value edit declares, and ``_finish_edit`` would refuse it. That is what makes
    the option set finite enough to be empty, which is why the sampler consults
    this function before committing to a chain rather than discovering it here.

    Under ``distinct`` the same three-way separation the ancestor edit gets is
    required of this one: the digit the donor line states must be neither the
    clean answer nor the value it implies, or "carried the value through the
    remaining steps" and "copied the digit sitting at the patched position"
    stop being different predictions. The second can only fail for a whole line
    at once -- the steps below it sum to zero or they do not -- so at depth 3 it
    is a constraint on which chains are sampled, not on which digit is drawn.
    """
    parent = start + sum(steps[:index])
    sign = 1 if steps[index] > 0 else -1
    options = []
    for rhs in range(1, 10):
        if rhs == abs(steps[index]):
            continue
        value = parent + sign * rhs
        if not 0 <= value <= 9:
            continue
        implied, in_range = value, True
        for step in steps[index + 1:]:
            implied += step
            if not 0 <= implied <= 9:
                in_range = False
                break
        if not in_range:
            continue
        if distinct and (value == target_value or implied == value):
            continue
        options.append((rhs, value, implied))
    return options


def _sample_chain(rng: random.Random, n_steps: int, start: int,
                  donor_start: int, delta: int, tries: int = 200,
                  donors: bool = False, distinct: bool = False):
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

    ``donors`` adds one more constraint, and only when a run asks for chain
    edits: every chain line must admit a donor of its own. It is resolved here
    rather than at the edit, because a chain edit that could be missing from some
    items and not others would break the fixed rows-per-item layout the scorer
    reads a report back through, and rejecting the item instead would re-roll it
    off the *main* stream and desynchronise the depth arms. Restricting the chain
    keeps both failures out of reach: the chain is the one part of the item the
    spine does not fix, so narrowing it moves no other line. Chains differ from
    those a run without chain edits samples, and the ancestor edit does not --
    the net delta is fixed by the spine either way.
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
                full = (*steps, last)
                if not donors or all(
                    _chain_donor_options(start, full, index, distinct, start + delta)
                    for index in range(n_steps - 1)
                ):
                    return full
    raise Reject("no chain realises the required net delta")


def _build_paired(rng: random.Random, encode, n_decoys: int, condition: str,
                  depth: int, gap: int | None, distinct: bool = False,
                  omit: str = "none", chain_edits: bool = False) -> DagItem:
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
    if distinct and rerolled == value_c:
        # `v3_distinct`. The digit the donor line states at the patched result
        # position is what a readout that copies what it finds there would emit.
        # Letting it equal the clean answer collapses "copied the digit" and
        # "did not move" into one prediction, which is the comparison the
        # cross-item control turned out to hinge on.
        #
        # Tested on `rerolled` rather than on the digit this particular
        # condition renders, so it fires identically in all three: the clean
        # trace has to be the same under every condition, and a rejection that
        # consulted the rendered digit would re-roll `operand_only` at different
        # times than `both`. Under `operand_only` the rendered digit is the
        # ancestor's own value, which `value_c` is already drawn to avoid.
        #
        # Spine-only, like the two above: `value_c` and `start` are both drawn
        # before the chain exists. A depth-dependent rejection here would
        # desynchronise the family exactly as `v1_unpaired` was.
        raise Reject("donor states the clean answer at the patched position")

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
    steps = _sample_chain(chain_rng, depth, start, rerolled, value_c - start,
                          donors=chain_edits, distinct=distinct)
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

    # `chain` omits the values between the ancestor and the target: the target's
    # own value is what the readout reads and the ancestor's is what the patch
    # overwrites, so neither can go. At depth 1 the chain is empty and the flag
    # is a no-op, which makes depth 1 the contrast's own control.
    #
    # `decoy` omits the same number of values from lines the target does not
    # depend on. Same notation, same token budget, but the answer stays
    # computable from what is written. It is the control for the notation
    # itself: if the model fails here too, `chain`'s failure says nothing about
    # carrying a value, only that it cannot read the format.
    omitted = {"chain": chain,
               "decoy": tuple(decoys[:depth - 1]),
               "none": ()}[omit]
    omit_pad = {name: _omission_pad(encode, str(nodes[name].value))
                for name in omitted}

    chunks = _render(nodes, order, tags, omit_pad)
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
        value_positions={name: sites[f"result:{name}"] for name in order
                         if f"result:{name}" in sites},
        operand_positions={
            name: sites[f"operand:{name}"] for name in order
            if f"operand:{name}" in sites
        },
        read_position=read_position,
        condition=condition,
        depth=depth,
        gap=len(decoys) - cut,
        omit_pad=omit_pad,
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
        # An omitted line has no result position to rewrite, so it contributes
        # no null edit. The null spread is then over one fewer decoy per omitted
        # line, which the per-layer quorum already takes from the row count.
        if name in omit_pad:
            continue
        edits.append(_value_edit(rng, item, nodes, order, tags, encode, name,
                                 "null", condition, chain))
    # Appended last, and drawn from the chain stream rather than the main one, so
    # that turning them on leaves every line above and every edit above it
    # untouched: a chain-edit run is the same batch as a plain one at the same
    # seed apart from its chain values, which `_sample_chain` narrows. One per
    # chain line, always, so the rows-per-item block layout stays fixed.
    for name in chain if chain_edits else ():
        edits.append(_chain_edit(chain_rng, item, nodes, order, tags, encode,
                                 name, "chain", condition, chain, steps, start,
                                 distinct))
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
                 force_positions=None, donor_raw_value=None,
                 steps_to_target=None) -> Edit:
    ids, _ = encode_chunks(_render(nodes, order, tags, item.omit_pad), encode)
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
        steps_to_target=steps_to_target,
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
                        donor_raw_value=edited[name].value,
                        # Only the ancestor is on the path to the target; a
                        # non-ancestor or a decoy has no step count to report.
                        steps_to_target=len(chain) + 1 if name == ANCESTOR else None)


def _chain_edit(rng, item, nodes, order, tags, encode, name, kind, condition,
                chain, steps, start, distinct) -> Edit:
    """Patch a written intermediate instead of the ancestor.

    This edit and the ancestor edit are the same intervention: two token
    positions on one line, one donor value, the same affine counterfactual
    through whatever the trace states after it. They differ in one thing, which
    is how many written lines stand between the patched value and the target --
    ``depth`` for the ancestor, one for the last chain line whatever the depth.

    That makes the pair a *within-item* contrast. The depth ladder compares an
    ancestor edit against another item's ancestor edit at another depth, and
    carries the objection that the two arms differ in token distance as well as
    in steps. Here both edits land in the same trace, on the same clean readout,
    against the same null spread, and one of them is one step from the target by
    construction. If the ancestor edit is inert at depth 2 and this one is not,
    the trace's written values are doing the carrying.

    The step distance and the token distance still move together -- the chain
    line sits nearer the target than the ancestor does -- so this is a
    dissociation between two patch sites, not yet an unconfounded one. The
    matched gap arms are what separate the two, and ``steps_to_target`` is
    recorded beside ``distance_to_read`` so the analysis can hold one fixed.
    """
    index = chain.index(name)
    options = _chain_donor_options(start, steps, index, distinct,
                                   item.target_value)
    if not options:  # pragma: no cover -- the sampler is asked to rule this out
        raise Reject("chain line admits no donor")
    rhs, value, implied = rng.choice(options)
    node = nodes[name]
    if condition == "result_only":
        donor = dataclasses.replace(node, value=value)
    elif condition == "operand_only":
        donor = dataclasses.replace(node, rhs=str(rhs))
    else:
        donor = dataclasses.replace(node, rhs=str(rhs), value=value)
    sites = (item.operand_positions[name], item.value_positions[name])
    return _finish_edit(item, dict(nodes) | {name: donor}, order, tags, encode,
                        kind, name, implied,
                        force_positions=tuple(sorted(sites)),
                        # The digit the donor line states, which under
                        # ``operand_only`` is still the clean one. Same rule as
                        # ``_value_edit``; see the note there.
                        donor_raw_value=donor.value,
                        steps_to_target=len(chain) - index)


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
                    if name in item.value_positions
                    and item.value_positions[name] < item.read_position]
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


def _implied_by_donor(recipient: DagItem, donor: DagItem,
                      distinct: bool = False) -> int | None:
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

    ``distinct`` is ``v3_distinct``: it also rejects a donor whose stated
    ancestor digit is the recipient's clean answer. That digit is what a readout
    copying the patched token would emit, and the within-item rejection cannot
    cover it because it comes from a different item.
    """
    implied = (donor.nodes[ANCESTOR].value
               + recipient.target_value - recipient.nodes[ANCESTOR].value)
    if not 0 <= implied <= 9 or implied == recipient.target_value:
        return None
    if distinct and donor.nodes[ANCESTOR].value == recipient.target_value:
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


def _select_cross_item_batch(candidates: list[DagItem], n_items: int,
                             distinct: bool = False):
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
                                  candidates[donor], distinct) is not None]
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


def _attach_cross_item_edits(items: list[DagItem], sigma: list[int],
                             distinct: bool = False):
    attached = []
    for index, item in enumerate(items):
        donor = items[sigma[index]]
        sites = _ancestor_sites(item)
        edit = Edit(
            kind="cross_item",
            node=ANCESTOR,
            positions=sites,
            token_ids=donor.token_ids,
            implied_target_value=_implied_by_donor(item, donor, distinct),
            distance_to_read=item.read_position - max(sites),
            donor_item=sigma[index],
            donor_raw_value=donor.nodes[ANCESTOR].value,
        )
        attached.append(dataclasses.replace(item, edits=(*item.edits, edit)))
    return attached


def generate_items(encode, *, n_items: int = 5, n_decoys: int = 6, seed: int = 0,
                   condition: str = "both", depth: int = 1, gap: int | None = None,
                   generator: str = DEFAULT_GENERATOR,
                   cross_item: bool = False, oversample: int | None = None,
                   omit: str = "none", chain_edits: bool = False):
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

    ``omit`` renders some lines without stating their results: ``chain`` drops
    the values between the ancestor and the target, ``decoy`` drops the same
    number from lines the target does not depend on. It is a rendering choice
    and consumes nothing from the stream, so an omitted batch is the *same*
    batch as the written one, item for item.

    ``chain_edits`` adds one edit per written intermediate, patching the values
    the depth ladder leaves clean. It is off by default because the archived
    artifacts were measured without it and are re-derived by regenerating their
    items: an extra edit in the list would fail the replay in ``dag_evidence``.
    On, it narrows which chains are sampled -- see ``_sample_chain`` -- so a
    depth arm's chain values are not the ones a plain run at the same seed draws.
    Everything the spine fixes, the ancestor edit included, is unchanged.
    """
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}, "
                         f"expected one of {GENERATORS}")
    if condition not in DONOR_CONDITIONS:
        raise ValueError(f"unknown donor condition {condition!r}, "
                         f"expected one of {DONOR_CONDITIONS}")
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    if omit not in OMIT_MODES:
        raise ValueError(f"unknown omit mode {omit!r}, expected one of {OMIT_MODES}")
    if chain_edits and omit == "chain":
        # The chain edit rewrites a written intermediate's result digit, and
        # `omit="chain"` is the arm that does not write one. Asking for both is
        # asking to patch a position that does not exist.
        raise ValueError("chain edits patch the written intermediates, which "
                         "omit='chain' does not write; pick one")
    # No guard for depth 1: there is no intermediate to patch, so the flag is a
    # no-op there and the arm is the contrast's own control -- the same rule
    # `omit` already follows. The realised chain nodes are recorded per item, so
    # "asked for and got none" stays distinguishable from "did not ask".
    # Chain steps consume decoy names, every line consumes a line tag, and the
    # surface-null edit needs two tags left over.
    budget = min(len(DECOY_NAMES) + 1, len(TAG_POOL) - 5)
    if n_decoys + depth > budget:
        raise ValueError(f"depth {depth} with {n_decoys} decoys does not fit the "
                         f"name and tag pools; keep n_decoys + depth <= {budget}")
    if gap is not None and not 0 <= gap <= n_decoys:
        raise ValueError(f"gap must be between 0 and n_decoys ({n_decoys}), got {gap}")
    distinct = generator == "v3_distinct"
    if generator == "v1_unpaired":
        if omit != "none" or chain_edits:
            raise ValueError("v1_unpaired predates omission and chain edits and "
                             "is frozen; use v2_paired or v3_distinct")
        build = _build_unpaired
    else:
        def build(rng, encode, n_decoys, condition, depth, gap):
            return _build_paired(rng, encode, n_decoys, condition, depth,
                                 gap, distinct=distinct, omit=omit,
                                 chain_edits=chain_edits)
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
        items, sigma = _select_cross_item_batch(items, n_items, distinct)
        items = _attach_cross_item_edits(items, sigma, distinct)
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
