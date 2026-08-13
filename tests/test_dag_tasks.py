"""Invariants the DAG generator must hold before any model is loaded.

A violation here silently misplaces every patch, so these are preconditions for
the experiment rather than ordinary unit tests.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dag_tasks import (
    ANCESTOR,
    NON_ANCESTOR,
    TARGET,
    ancestors,
    dependency_edges,
    generate_items,
    transitive_reduction,
)


def char_encode(text):
    """Stand-in tokenizer with the merge behaviour that actually matters.

    The Qwen tokenizers keep a space separate from a following digit but merge
    it with a following letter. The generator's chunk boundaries depend on
    exactly that, so a fake that merges nothing would accept a trace the real
    tokenizers reject -- which is how the line tag shipped broken.
    """
    tokens, index = [], 0
    while index < len(text):
        if text[index] == " " and index + 1 < len(text) and text[index + 1].isalpha():
            tokens.append(text[index:index + 2])
            index += 2
        else:
            tokens.append(text[index])
            index += 1
    return [int.from_bytes(token.encode(), "big") for token in tokens]


def char_decode(token_ids):
    return "".join(
        token.to_bytes((token.bit_length() + 7) // 8, "big").decode()
        for token in token_ids
    )


@pytest.fixture(scope="module")
def items():
    return generate_items(char_encode, n_items=5, n_decoys=6, seed=0)


def test_generates_the_requested_number_of_items(items):
    assert len(items) == 5


def test_every_node_value_is_a_single_digit(items):
    for item in items:
        for node in item.nodes.values():
            assert 0 <= node.value <= 9


def test_every_edit_implied_value_is_a_single_digit(items):
    for item in items:
        for edit in item.edits:
            assert 0 <= edit.implied_target_value <= 9


def test_donor_traces_are_token_aligned_with_the_clean_trace(items):
    for item in items:
        for edit in item.edits:
            assert len(edit.token_ids) == len(item.token_ids)
            differing = {
                i for i, (x, y) in enumerate(zip(edit.token_ids, item.token_ids))
                if x != y
            }
            assert differing == set(edit.positions)


def test_every_edit_is_upstream_of_the_read_position(items):
    # A patch at or after the read position is masked out by causal attention and
    # would give an identically zero effect, making the null spread vacuous.
    for item in items:
        for edit in item.edits:
            assert max(edit.positions) < item.read_position


def test_value_edits_touch_two_positions(items):
    # Operand and stated result, so the donor stays arithmetically consistent.
    for item in items:
        for edit in item.edits:
            assert len(edit.positions) == 2, edit.kind


def test_read_position_predicts_the_target_digit(items):
    for item in items:
        assert item.read_position == item.value_positions[item.target] - 1


def test_target_result_token_is_the_target_value(items):
    for item in items:
        position = item.value_positions[item.target]
        assert item.token_ids[position] == ord(str(item.target_value))


def test_ancestor_edit_implies_a_changed_target_value(items):
    for item in items:
        edit = next(e for e in item.edits if e.kind == "ancestor")
        assert edit.implied_target_value != item.target_value


def test_non_ancestor_and_null_edits_imply_an_unchanged_target(items):
    for item in items:
        for edit in item.edits:
            if edit.kind in {"non_ancestor", "surface_null", "null"}:
                assert edit.implied_target_value == item.target_value


def test_edit_kinds_match_the_prototype_specification(items):
    for item in items:
        kinds = [edit.kind for edit in item.edits]
        assert kinds.count("ancestor") == 1
        assert kinds.count("non_ancestor") == 1
        assert kinds.count("surface_null") == 1
        assert 5 <= kinds.count("null") <= 10


def test_ancestor_set_of_the_target_is_exactly_the_ancestor_node(items):
    for item in items:
        edges = set(item.edges)
        assert ancestors(edges, TARGET) == {ANCESTOR}
        assert NON_ANCESTOR not in ancestors(edges, TARGET)


def test_edges_are_the_transitive_reduction(items):
    for item in items:
        full = dependency_edges(item.nodes)
        assert set(item.edges) == transitive_reduction(full)


def test_transitive_reduction_drops_the_implied_edge():
    # a -> c -> e plus the shortcut a -> e; only the shortcut may be dropped.
    edges = {("a", "c"), ("c", "e"), ("a", "e")}
    assert transitive_reduction(edges) == {("a", "c"), ("c", "e")}


def test_ancestor_edit_recomputes_the_target_correctly(items):
    for item in items:
        edit = next(e for e in item.edits if e.kind == "ancestor")
        # The donor's stated value for the ancestor node, read back off the trace.
        donor_ancestor = int(
            char_decode([edit.token_ids[item.value_positions[ANCESTOR]]])
        )
        target = item.nodes[TARGET]
        expected = (
            donor_ancestor + int(target.rhs)
            if target.op == "+"
            else donor_ancestor - int(target.rhs)
        )
        assert edit.implied_target_value == expected


def test_generation_is_deterministic_given_a_seed():
    first = generate_items(char_encode, n_items=3, seed=7)
    second = generate_items(char_encode, n_items=3, seed=7)
    assert [item.token_ids for item in first] == [item.token_ids for item in second]


def test_different_seeds_give_different_items():
    first = generate_items(char_encode, n_items=3, seed=1)
    second = generate_items(char_encode, n_items=3, seed=2)
    assert [item.token_ids for item in first] != [item.token_ids for item in second]


def test_ancestor_and_non_ancestor_sit_on_adjacent_lines(items):
    # The distance match is what makes the non-ancestor a control rather than
    # just another node. Adjacent lines is the closest two distinct steps get.
    for item in items:
        positions = [item.order.index(ANCESTOR), item.order.index(NON_ANCESTOR)]
        assert abs(positions[0] - positions[1]) == 1


def test_ancestor_and_non_ancestor_distances_are_not_systematically_ordered():
    # Their order is random, so the residual one-line mismatch does not point the
    # same way in every item.
    items = generate_items(char_encode, n_items=12, seed=3)
    closer = 0
    for item in items:
        ancestor = next(e for e in item.edits if e.kind == "ancestor")
        non_ancestor = next(e for e in item.edits if e.kind == "non_ancestor")
        closer += ancestor.distance_to_read < non_ancestor.distance_to_read
    assert 0 < closer < len(items)


def test_no_step_uses_a_zero_operand(items):
    # A zero operand turns the step into a copy, so an "edit the parent" probe
    # would test copying rather than computation.
    for item in items:
        for node in item.nodes.values():
            for operand in (node.lhs, node.rhs):
                if operand.isdigit():
                    assert int(operand) != 0


def test_donor_traces_also_avoid_zero_operands(items):
    # A node value of 0 is fine; only the two operand slots must stay non-zero.
    for item in items:
        for edit in item.edits:
            for line in char_decode(edit.token_ids).splitlines():
                lhs, op, rhs = line.split(" = ")[1].split(" # ")[0].split(" ")
                assert lhs != "0" and rhs != "0", line


def test_line_tags_sit_at_the_end_of_the_line_behind_a_space(items):
    # The first version put the tag right after "[". Qwen merges "[o" into one
    # token, so every candidate was rejected and the generator could not produce
    # anything at all. Only a space may precede a tag.
    for item in items:
        for line in item.text.splitlines():
            body, _, tag = line.rpartition(" # ")
            assert body and len(tag) == 1 and tag.isalpha()


def test_survives_a_tokenizer_that_merges_punctuation_with_a_following_letter():
    def merging_encode(text):
        tokens, index = [], 0
        while index < len(text):
            following = text[index + 1] if index + 1 < len(text) else ""
            if not text[index].isalnum() and following.isalpha():
                tokens.append(text[index:index + 2])
                index += 2
            else:
                tokens.append(text[index])
                index += 1
        return [int.from_bytes(token.encode(), "big") for token in tokens]

    assert len(generate_items(merging_encode, n_items=2, seed=0)) == 2


def test_rejects_a_tokenizer_that_splits_digits():
    def splitting_encode(text):
        out = []
        for ch in text:
            out.extend([ord(ch), ord(ch)] if ch.isdigit() else [ord(ch)])
        return out

    with pytest.raises(RuntimeError, match="could not sample"):
        generate_items(splitting_encode, n_items=1, seed=0)


def test_the_failure_message_names_the_reason():
    # Without this the only signal is "check the tokenizer", which is what made
    # the line-tag merge take a round trip to diagnose.
    def splitting_encode(text):
        out = []
        for ch in text:
            out.extend([ord(ch), ord(ch)] if ch.isdigit() else [ord(ch)])
        return out

    with pytest.raises(RuntimeError, match=r"\d+x site 'operand:\w+' is 2 tokens"):
        generate_items(splitting_encode, n_items=1, seed=0)
