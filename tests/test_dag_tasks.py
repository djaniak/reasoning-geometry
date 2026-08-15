"""Invariants the DAG generator must hold before any model is loaded.

A violation here silently misplaces every patch, so these are preconditions for
the experiment rather than ordinary unit tests.
"""

import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dag.dag_tasks import (
    ANCESTOR,
    DONOR_CONDITIONS,
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
    # A donor may differ at fewer positions than it declares -- the conditions
    # that change one half of the line leave the other half as an identity patch
    # -- but never at a position it does not declare.
    for item in items:
        for edit in item.edits:
            assert len(edit.token_ids) == len(item.token_ids)
            differing = {
                i for i, (x, y) in enumerate(zip(edit.token_ids, item.token_ids))
                if x != y
            }
            assert differing <= set(edit.positions)
            assert differing


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


# --------------------------------------------------------------------------
# donor conditions
# --------------------------------------------------------------------------


def condition_items(condition):
    return generate_items(char_encode, n_items=5, n_decoys=6, seed=0,
                          condition=condition)


def parse_line(text, name):
    """Return (lhs, op, rhs, result) of ``name``'s line, as integers and a str."""
    for line in text.splitlines():
        if line.startswith(f"{name} = "):
            _, expression, result = line.split(" # ")[0].split(" = ")
            lhs, op, rhs = expression.split(" ")
            return int(lhs), op, int(rhs), int(result)
    raise AssertionError(f"no line for {name} in {text!r}")


def ancestor_edit(item):
    return next(edit for edit in item.edits if edit.kind == "ancestor")


@pytest.mark.parametrize("condition", DONOR_CONDITIONS)
def test_every_condition_generates(condition):
    assert len(condition_items(condition)) == 5


def test_the_clean_trace_is_the_same_under_every_condition():
    # The conditions must differ in the donors only. If the clean traces drifted,
    # the three runs would not be measuring the same items.
    traces = {
        condition: [item.token_ids for item in condition_items(condition)]
        for condition in DONOR_CONDITIONS
    }
    assert traces["result_only"] == traces["both"]
    assert traces["operand_only"] == traces["both"]


def test_all_conditions_imply_the_same_target_value():
    # This is what makes the three runs comparable: the same directional
    # statistic, so a difference between them is about mechanism, not target.
    implied = {
        condition: [ancestor_edit(item).implied_target_value
                    for item in condition_items(condition)]
        for condition in DONOR_CONDITIONS
    }
    assert implied["result_only"] == implied["both"]
    assert implied["operand_only"] == implied["both"]


def test_all_conditions_patch_the_same_positions():
    # Equal number of residual states written, so a smaller effect cannot be
    # explained by having patched fewer positions.
    positions = {
        condition: [ancestor_edit(item).positions
                    for item in condition_items(condition)]
        for condition in DONOR_CONDITIONS
    }
    assert positions["result_only"] == positions["both"]
    assert positions["operand_only"] == positions["both"]


def test_both_still_differs_at_exactly_its_declared_positions():
    # Under "both" the forced positions coincide with the positions that actually
    # differ, so adding the conditions did not change the arm already measured.
    for item in condition_items("both"):
        for edit in item.edits:
            differing = {
                i for i, (x, y) in enumerate(zip(edit.token_ids, item.token_ids))
                if x != y
            }
            assert differing == set(edit.positions)


def test_both_keeps_the_edited_line_arithmetically_consistent():
    for item in condition_items("both"):
        lhs, op, rhs, result = parse_line(
            char_decode(ancestor_edit(item).token_ids), ANCESTOR
        )
        assert (lhs + rhs if op == "+" else lhs - rhs) == result


def test_result_only_changes_the_result_and_leaves_the_operands_alone():
    for item in condition_items("result_only"):
        clean = parse_line(item.text, ANCESTOR)
        donor = parse_line(char_decode(ancestor_edit(item).token_ids), ANCESTOR)
        assert donor[:3] == clean[:3]
        assert donor[3] != clean[3]


def test_operand_only_changes_the_operand_and_leaves_the_result_alone():
    for item in condition_items("operand_only"):
        clean = parse_line(item.text, ANCESTOR)
        donor = parse_line(char_decode(ancestor_edit(item).token_ids), ANCESTOR)
        assert donor[2] != clean[2]
        assert donor[3] == clean[3]


def test_the_two_split_conditions_state_something_false():
    # Deliberate. Each leaves exactly one mechanism able to move the answer.
    for condition in ("result_only", "operand_only"):
        for item in condition_items(condition):
            lhs, op, rhs, result = parse_line(
                char_decode(ancestor_edit(item).token_ids), ANCESTOR
            )
            assert (lhs + rhs if op == "+" else lhs - rhs) != result


def test_an_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="unknown donor condition"):
        generate_items(char_encode, n_items=1, seed=0, condition="operand")


# --------------------------------------------------------------------------
# depth ladder
# --------------------------------------------------------------------------


def ladder_items(depth=1, gap=None, n_decoys=6, seed=0):
    return generate_items(char_encode, n_items=5, n_decoys=n_decoys, seed=seed,
                          depth=depth, gap=gap)


def chain_of(item):
    """The ancestor-to-target path, in trace order."""
    path = ancestors(set(item.edges), TARGET) - {ANCESTOR}
    return [name for name in item.order if name in path]


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_the_path_to_the_target_has_the_requested_depth(depth):
    for item in ladder_items(depth=depth):
        assert item.depth == depth
        assert len(ancestors(set(item.edges), TARGET)) == depth


@pytest.mark.parametrize("generator", ["v1_unpaired", "v2_paired"])
def test_depth_one_is_whatever_that_generator_produces_by_default(generator):
    # Asking for depth 1 explicitly must not re-roll the arm; it is the default.
    # Pinned per generator, because the two are deliberately different families.
    assert [item.token_ids for item in
            generate_items(char_encode, n_items=5, seed=0, depth=1,
                           generator=generator)] == \
        [item.token_ids for item in
         generate_items(char_encode, n_items=5, seed=0, generator=generator)]


def test_the_chain_is_a_path_from_the_ancestor_to_the_target():
    for item in ladder_items(depth=3):
        chain = chain_of(item)
        edges = set(item.edges)
        for parent, child in zip([ANCESTOR] + chain, chain + [TARGET]):
            assert (parent, child) in edges
        assert NON_ANCESTOR not in ancestors(edges, TARGET)


def test_every_chain_step_reads_the_step_before_it():
    for item in ladder_items(depth=3):
        chain = chain_of(item)
        for parent, child in zip([ANCESTOR] + chain, chain + [TARGET]):
            node = item.nodes[child]
            assert node.lhs == parent
            assert node.rhs.isdigit() and int(node.rhs) != 0


def test_a_chain_line_is_written_before_the_line_that_reads_it():
    for item in ladder_items(depth=3):
        chain = chain_of(item)
        order = list(item.order)
        for parent, child in zip([ANCESTOR] + chain, chain + [TARGET]):
            assert order.index(parent) < order.index(child)


def test_the_ancestor_edit_implies_what_the_whole_chain_produces():
    for item in ladder_items(depth=3):
        edit = ancestor_edit(item)
        value = int(char_decode([edit.token_ids[item.value_positions[ANCESTOR]]]))
        for name in chain_of(item) + [TARGET]:
            node = item.nodes[name]
            value = value + int(node.rhs) if node.op == "+" else value - int(node.rhs)
        assert edit.implied_target_value == value
        assert 0 <= value <= 9


def test_a_deeper_chain_moves_the_ancestor_edit_further_from_the_read_position():
    # The confound the gap control exists for: depth costs tokens as well as steps.
    shallow = ladder_items(depth=1, gap=0)
    deep = ladder_items(depth=3, gap=0)
    for near, far in zip(shallow, deep):
        assert ancestor_edit(far).distance_to_read > \
            ancestor_edit(near).distance_to_read


def test_gap_buys_token_distance_without_buying_depth():
    for gap in range(4):
        items = ladder_items(depth=1, gap=gap)
        assert all(item.gap == gap for item in items)
        assert all(len(ancestors(set(item.edges), TARGET)) == 1 for item in items)
    distances = [
        ancestor_edit(ladder_items(depth=1, gap=gap)[0]).distance_to_read
        for gap in range(4)
    ]
    assert distances == sorted(distances) and distances[0] < distances[-1]


def test_a_gap_arm_can_be_matched_to_a_depth_arm_on_token_distance():
    # Without a matched pair the depth ladder is unreadable, so this checks the
    # matching is reachable at all rather than only in principle. The match is
    # close but not exact: a chain line names its operand instead of stating a
    # digit, which costs one token less than a decoy line.
    def median_distance(**kwargs):
        return statistics.median(
            ancestor_edit(item).distance_to_read for item in ladder_items(**kwargs)
        )

    deep = median_distance(depth=2, gap=0)
    matched = [median_distance(depth=1, gap=gap) for gap in range(7)]
    assert min(abs(distance - deep) for distance in matched) <= 2


def test_depth_and_decoys_together_must_fit_the_name_pools():
    with pytest.raises(ValueError, match=r"keep n_decoys \+ depth <= \d+"):
        generate_items(char_encode, n_items=1, seed=0, n_decoys=6, depth=4)


def test_a_depth_below_one_is_rejected():
    with pytest.raises(ValueError, match="depth must be at least 1"):
        generate_items(char_encode, n_items=1, seed=0, depth=0)


def test_a_gap_wider_than_the_decoy_pool_is_rejected():
    with pytest.raises(ValueError, match="gap must be between 0 and n_decoys"):
        generate_items(char_encode, n_items=1, seed=0, n_decoys=6, gap=7)


def test_the_deepest_ladder_rung_still_generates():
    items = ladder_items(depth=4, n_decoys=5)
    assert len(items) == 5
    for item in items:
        for node in item.nodes.values():
            assert 0 <= node.value <= 9
        assert max(ancestor_edit(item).positions) < item.read_position


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


# --------------------------------------------------------------------------
# cross-depth pairing
#
# The depth ladder is only readable if depth-1 item i and depth-3 item i are the
# same item apart from the chain. Under the unpaired generator they are not: the
# chain's `_sample_step` draws come out of the main stream, so every draw after
# them lands at a different position and the whole item re-rolls. That makes the
# depth contrast a between-family difference, which no amount of GPU turns into
# a paired estimate.
#
# `v2_paired` takes one chain seed from the main stream per item whatever the
# depth and builds the chain from a separate stream, so the spine is identical
# at every depth. This is the audit to run before spending a GPU on the ladder.
# --------------------------------------------------------------------------


def paired_items(depth, *, n_items=5, n_decoys=6, seed=0, gap=0):
    return generate_items(char_encode, n_items=n_items, n_decoys=n_decoys,
                          seed=seed, depth=depth, gap=gap,
                          generator="v2_paired")


def spine_lines(item):
    """Every line except the chain, as rendered text, in trace order."""
    chain = set(chain_of(item))
    return [line for name, line in zip(item.order, item.text.splitlines())
            if name not in chain]


DEPTHS = [1, 2, 3]


def test_the_spine_is_identical_at_every_depth():
    families = {depth: paired_items(depth) for depth in DEPTHS}
    for index in range(5):
        rendered = {depth: spine_lines(items[index])
                    for depth, items in families.items()}
        # The target line names its parent and states the step's operand, both
        # of which are the last link of the chain. Everything else must match.
        without_target = {depth: lines[:-2] + lines[-1:]
                          for depth, lines in rendered.items()}
        assert len(set(map(tuple, without_target.values()))) == 1, (
            f"item {index} spine differs across depths: {without_target}"
        )


def test_the_target_value_is_identical_at_every_depth():
    families = {depth: paired_items(depth) for depth in DEPTHS}
    for index in range(5):
        values = {items[index].target_value for items in families.values()}
        assert len(values) == 1, f"item {index} target value differs: {values}"


def test_every_non_chain_node_is_identical_at_every_depth():
    families = {depth: paired_items(depth) for depth in DEPTHS}
    for index in range(5):
        per_depth = []
        for depth, items in families.items():
            item = items[index]
            chain = set(chain_of(item))
            per_depth.append({
                name: node for name, node in item.nodes.items()
                if name not in chain and name != TARGET
            })
        assert all(other == per_depth[0] for other in per_depth[1:])


def test_the_ancestor_edit_implies_the_same_target_value_at_every_depth():
    # The chain is an affine map on the ancestor's value, so a paired family has
    # a fixed net delta: the counterfactual the ancestor edit asserts is the same
    # trace-level claim at every depth, only reached through more steps.
    families = {depth: paired_items(depth) for depth in DEPTHS}
    for index in range(5):
        implied = {ancestor_edit(items[index]).implied_target_value
                   for items in families.values()}
        assert len(implied) == 1, f"item {index} implied value differs: {implied}"


def test_the_irrelevant_lines_keep_their_tags_and_their_order():
    families = {depth: paired_items(depth) for depth in DEPTHS}
    for index in range(5):
        orders = set()
        for items in families.values():
            item = items[index]
            chain = set(chain_of(item))
            orders.add(tuple(n for n in item.order if n not in chain))
        assert len(orders) == 1, f"item {index} line order differs: {orders}"


def test_only_the_chain_lines_are_added_as_depth_grows():
    for index in range(5):
        counts = [len(paired_items(depth)[index].order) for depth in DEPTHS]
        assert counts == [counts[0] + depth - 1 for depth in DEPTHS]


def test_the_unpaired_generator_still_reproduces_what_was_already_measured():
    # Characterisation test, not new behaviour: the three v0 artifacts are
    # re-derived by regenerating their items, so the legacy path has to stay
    # byte-exact or `dag_evidence` can no longer verify the manifest.
    legacy = generate_items(char_encode, n_items=5, seed=0,
                            generator="v1_unpaired")
    assert [item.target_value for item in legacy] == [3, 7, 2, 7, 3]
    assert len({item.token_ids for item in legacy}) == 5


def test_the_two_generators_are_not_the_same_family():
    # The paired generator draws a chain seed and two spare tag letters that the
    # legacy one never drew, so it necessarily samples a different family. Said
    # out loud here so nobody reads a paired run as a replay of an archived one.
    assert [item.token_ids for item in paired_items(1, gap=None)] != \
        [item.token_ids for item in generate_items(char_encode, n_items=5,
                                                   seed=0,
                                                   generator="v1_unpaired")]


# --------------------------------------------------------------------------
# cross-item donors
#
# Every control in the pilot so far edits the *recipient's own* trace. That
# leaves the strongest objection standing: the ancestor gap could be generic
# sensitivity at those token positions rather than the edge being read. The
# cross-item donor is the matched test. It writes another item's residual state
# at the same positions, with the same span, width and formatting -- so the only
# thing that changes is which item the state came from.
#
# It predicts a specific digit. The chain is affine, so donor value `v_j` seen
# through recipient i's chain implies `v_j + delta_i`, which is neither the
# clean answer nor the donor's own digit. Those three being distinct is what
# makes the arm readable, so it is asserted here rather than hoped for.
# --------------------------------------------------------------------------


def cross_items(depth=1, *, n_items=5, n_decoys=6, seed=0, gap=0):
    return generate_items(char_encode, n_items=n_items, n_decoys=n_decoys,
                          seed=seed, depth=depth, gap=gap, cross_item=True)


def cross_edit(item):
    return next(edit for edit in item.edits if edit.kind == "cross_item")


def ancestor_sites(item):
    return tuple(sorted((item.operand_positions[ANCESTOR],
                         item.value_positions[ANCESTOR])))


def test_cross_item_edits_are_off_unless_asked_for():
    for item in ladder_items():
        assert not any(edit.kind == "cross_item" for edit in item.edits)


def test_every_item_receives_exactly_one_cross_item_edit():
    for item in cross_items():
        assert sum(edit.kind == "cross_item" for edit in item.edits) == 1


def test_the_donor_assignment_is_a_derangement():
    donors = [cross_edit(item).donor_item for item in cross_items()]
    assert sorted(donors) == list(range(5)), f"not a permutation: {donors}"
    assert all(donor != index for index, donor in enumerate(donors)), (
        f"an item donates to itself: {donors}"
    )


def test_the_donor_trace_is_another_items_clean_trace():
    items = cross_items()
    for item in items:
        edit = cross_edit(item)
        assert edit.token_ids == items[edit.donor_item].token_ids


def test_the_patch_lands_on_the_ancestors_own_two_sites():
    items = cross_items()
    for item in items:
        edit = cross_edit(item)
        assert edit.positions == ancestor_sites(item)


def test_donor_and_recipient_agree_on_where_the_ancestor_sits():
    # The state is lifted from position p in the donor and written to position p
    # in the recipient. If the two disagree the patch is not the intervention it
    # claims to be, so the batch is selected to make them agree.
    items = cross_items()
    for item in items:
        assert ancestor_sites(items[cross_edit(item).donor_item]) == \
            ancestor_sites(item)


def test_donor_and_recipient_are_the_same_width():
    items = cross_items()
    for item in items:
        assert len(cross_edit(item).token_ids) == len(item.token_ids)


def test_the_implied_value_is_the_donors_value_through_the_recipients_chain():
    items = cross_items(depth=2)
    for item in items:
        edit = cross_edit(item)
        donor_value = items[edit.donor_item].nodes[ANCESTOR].value
        assert edit.donor_raw_value == donor_value
        expected = donor_value
        for name in (*chain_of(item), TARGET):
            node = item.nodes[name]
            expected += int(node.rhs) if node.op == "+" else -int(node.rhs)
        assert edit.implied_target_value == expected


def test_the_prediction_is_neither_the_clean_answer_nor_the_donors_own_digit():
    # Three distinct digits, so the readout can tell "propagated the donor's
    # value" from "copied the patched digit" from "did not move".
    for item in cross_items():
        edit = cross_edit(item)
        assert edit.implied_target_value != item.target_value
        assert edit.implied_target_value != edit.donor_raw_value


def test_the_cross_item_edit_is_upstream_of_the_read_position():
    for item in cross_items():
        assert max(cross_edit(item).positions) < item.read_position


def test_the_cross_item_arm_is_paired_across_depth():
    # Same recipients, same donor map, same predicted digit at every depth --
    # otherwise this arm inherits the bug the depth ladder just had fixed.
    families = {depth: cross_items(depth) for depth in DEPTHS}
    for index in range(5):
        edits = [cross_edit(items[index]) for items in families.values()]
        assert len({edit.donor_item for edit in edits}) == 1
        assert len({edit.donor_raw_value for edit in edits}) == 1
        assert len({edit.implied_target_value for edit in edits}) == 1


def test_a_batch_with_no_possible_derangement_is_refused():
    # Better to fail loudly than to quietly run four recipients and call the
    # quorum on five.
    with pytest.raises(ValueError, match="derangement"):
        generate_items(char_encode, n_items=5, n_decoys=6, seed=0, depth=1,
                       gap=0, cross_item=True, oversample=5)


# --------------------------------------------------------------------------
# copy versus propagation
#
# The ancestor edit's implied value is the donor's stated value carried *through*
# the chain. A model that simply emits the digit it finds at the patched result
# position would score on that gate too, at depth 1, because the two predictions
# were never separated. Recording the donor's own stated value on every value
# edit is what separates them, and it costs no extra forward pass.
# --------------------------------------------------------------------------


def value_edits(item):
    return [edit for edit in item.edits
            if edit.kind in ("ancestor", "non_ancestor", "null")]


def test_every_value_edit_records_the_digit_it_writes():
    for item in ladder_items():
        for edit in value_edits(item):
            assert edit.donor_raw_value is not None, edit.kind
            assert 0 <= edit.donor_raw_value <= 9


def test_the_ancestor_edits_stated_digit_is_not_the_value_it_implies():
    # At depth 1 the chain still applies a non-zero step, so "copied the digit"
    # and "propagated the digit" are different predictions for every item.
    for item in ladder_items():
        edit = ancestor_edit(item)
        assert edit.donor_raw_value != edit.implied_target_value


def test_the_digit_a_value_edit_writes_is_the_one_its_donor_line_states():
    for item in ladder_items():
        for edit in value_edits(item):
            *_, stated = parse_line(char_decode(edit.token_ids), edit.node)
            assert edit.donor_raw_value == stated


@pytest.mark.parametrize("condition", DONOR_CONDITIONS)
def test_the_recorded_digit_is_the_one_sitting_at_the_patched_position(condition):
    # Not the value the reroll implies. Under `operand_only` the donor leaves the
    # result token alone, so a readout that merely copies what it finds there
    # predicts *no* movement -- which is a different prediction from the implied
    # value, and the whole reason to record this separately.
    for item in condition_items(condition):
        edit = ancestor_edit(item)
        assert edit.donor_raw_value == \
            edit.token_ids[item.value_positions[ANCESTOR]] - ord("0")
        if condition == "operand_only":
            assert edit.donor_raw_value == item.nodes[ANCESTOR].value


def test_the_surface_edit_has_no_digit_to_copy():
    for item in ladder_items():
        surface = next(e for e in item.edits if e.kind == "surface_null")
        assert surface.donor_raw_value is None


# --------------------------------------------------------------------------
# v3_distinct: keeping the three competing digits apart
# --------------------------------------------------------------------------
#
# A value edit sets up a three-way question at the read position: did the model
# carry the donor's value through the chain (`implied_target_value`), copy the
# digit standing at the patched position (`donor_raw_value`), or not move
# (`target_value`)? `v2_paired` keeps `implied` off `target`, but nothing keeps
# the *raw* digit off it -- 2 of 20 ancestor items in the paired ladder and 1 of
# 20 in the cross-item arm wrote the clean answer at the patched position, which
# makes "copied" and "did not move" the same prediction and the item unusable
# for the comparison that turned out to be the informative one.
#
# The fix is one more rejection, and it must be decided by the spine alone or it
# fires at some depths and not others and desynchronises the family -- the bug
# the ladder was rebuilt to remove. `start` and `value_c` are both drawn before
# the chain exists, and the raw digit is either the reroll or the ancestor's own
# value, so the test is spine-only by construction.
#
# It moves the random stream, so it is a new family rather than a fix to
# `v2_paired`, which stays reachable and unchanged for the artifacts already run
# against it.

V3 = "v3_distinct"


def moving_edits(item):
    """The edits that are supposed to change the answer.

    Only these pose the three-way question. A ``null`` or ``non_ancestor`` edit
    carries the *clean* target as its implied value by design -- a faithful model
    should not move -- so "implied" and "did not move" coincide there on purpose
    and the distinctness rule does not apply.
    """
    return [edit for edit in item.edits
            if edit.kind in ("ancestor", "cross_item")
            and edit.donor_raw_value is not None]


def test_v3_never_writes_the_clean_answer_at_the_patched_position():
    for seed in range(8):
        for condition in DONOR_CONDITIONS:
            items = generate_items(char_encode, n_items=5, seed=seed,
                                   condition=condition, generator=V3)
            for item in items:
                for edit in moving_edits(item):
                    assert edit.donor_raw_value != item.target_value


def test_v3_keeps_all_three_competing_digits_distinct():
    for seed in range(8):
        for item in generate_items(char_encode, n_items=5, seed=seed,
                                   generator=V3):
            for edit in moving_edits(item):
                assert len({edit.donor_raw_value, edit.implied_target_value,
                            item.target_value}) == 3


def test_v2_still_carries_the_defect_and_is_left_alone():
    # The frozen family must not be silently repaired: artifacts were run
    # against it, and a quiet fix would make them unreproducible.
    ill_posed = sum(
        edit.donor_raw_value == item.target_value
        for seed in range(8)
        for item in generate_items(char_encode, n_items=5, seed=seed,
                                   generator="v2_paired")
        for edit in moving_edits(item)
    )
    assert ill_posed > 0


def test_v3_is_still_paired_across_depth():
    # The whole reason the rejection is spine-only.
    families = {depth: generate_items(char_encode, n_items=5, seed=0,
                                      depth=depth, gap=0, generator=V3)
                for depth in DEPTHS}
    for index in range(5):
        items = [family[index] for family in families.values()]
        assert len({item.target_value for item in items}) == 1
        implied = [next(e for e in item.edits if e.kind == "ancestor")
                   .implied_target_value for item in items]
        assert len(set(implied)) == 1


def test_v3_cross_item_donors_are_well_posed_too():
    for seed in range(4):
        items = generate_items(char_encode, n_items=5, seed=seed, gap=0,
                               cross_item=True, generator=V3)
        for item in items:
            edit = next(e for e in item.edits if e.kind == "cross_item")
            assert len({edit.donor_raw_value, edit.implied_target_value,
                        item.target_value}) == 3


def test_v3_is_a_different_family_from_v2():
    # If the stream had not moved, the rejection would not be doing anything.
    v2 = generate_items(char_encode, n_items=5, seed=0, generator="v2_paired")
    v3 = generate_items(char_encode, n_items=5, seed=0, generator=V3)
    assert [item.target_value for item in v2] != [item.target_value for item in v3]


def test_an_unknown_generator_is_still_refused():
    with pytest.raises(ValueError, match="unknown generator"):
        generate_items(char_encode, n_items=1, generator="v4_imaginary")


# --------------------------------------------------------------------------
# omitting the downstream intermediate results
# --------------------------------------------------------------------------
#
# The depth ladder collapses after depth 1, and pairing does not explain why.
# Pairing fixed the item family and the token distance, but depth also adds
# *written correct intermediate values*: at depth 2 the trace states the chain
# node's result, at depth 3 it states two of them. Depth and the amount of
# correct scaffolding already in the text move together, and clean confidence
# rises from about 0.6 to about 0.99 across the same step. A teacher-forced
# trace overwriting or dominating the latent state would look exactly like this.
#
# The contrast that separates them changes only whether those values are stated.
# Same spine, same donor, same target operation, same item index, same patch
# anchor -- the ancestor line is always upstream of the chain, so its positions
# do not move -- and the omitted lines are padded with comment markers to the
# token count of the ` = <digit>` they replace, so nothing downstream shifts
# either.


def written_and_omitted(depth, **kwargs):
    common = dict(n_items=5, seed=0, gap=0, generator=V3, depth=depth, **kwargs)
    return (generate_items(char_encode, **common),
            generate_items(char_encode, omit="chain", **common))


def test_the_omitted_trace_does_not_state_the_intermediate_value():
    def line_of(item, name):
        return next(line for line in item.text.splitlines()
                    if line.startswith(f"{name} = "))

    for written, omitted in zip(*written_and_omitted(2)):
        (name,) = omitted.omit
        value = written.nodes[name].value
        # The written line states the result; the omitted one still defines the
        # node from its parent, so the value stays computable, just unwritten.
        assert line_of(written, name).count(" = ") == 2
        assert f"= {value} #" in line_of(written, name)
        assert line_of(omitted, name).count(" = ") == 1
        # No result position at all -- not merely a different digit there. The
        # operand digit stays, and may coincidentally equal the value.
        assert name in written.value_positions
        assert name not in omitted.value_positions


def test_omitting_a_value_costs_no_tokens_anywhere():
    # The pad is matched to the ` = <digit>` it replaces, so the two formats are
    # the same length and every position downstream of the chain is unmoved.
    for depth in (2, 3):
        for written, omitted in zip(*written_and_omitted(depth)):
            assert len(written.token_ids) == len(omitted.token_ids)
            assert written.read_position == omitted.read_position
            assert written.value_positions[ANCESTOR] == \
                omitted.value_positions[ANCESTOR]


def test_the_patch_lands_at_the_same_distance_under_both_formats():
    # If the ancestor edit sat closer to the read position in one format, the
    # contrast would be the depth/token-distance confound again, one level down.
    for depth in (2, 3):
        for written, omitted in zip(*written_and_omitted(depth)):
            for kind in ("ancestor", "non_ancestor", "surface_null"):
                one = next(e for e in written.edits if e.kind == kind)
                two = next(e for e in omitted.edits if e.kind == kind)
                assert one.positions == two.positions
                assert one.distance_to_read == two.distance_to_read


def test_the_two_formats_are_the_same_item():
    # Omission is a rendering choice, not a draw: it must not touch the stream,
    # or the contrast is between two families rather than two formats.
    for depth in (2, 3):
        for written, omitted in zip(*written_and_omitted(depth)):
            assert written.target_value == omitted.target_value
            assert written.order == omitted.order
            assert {n: v.value for n, v in written.nodes.items()} == \
                {n: v.value for n, v in omitted.nodes.items()}
            assert (next(e for e in written.edits if e.kind == "ancestor")
                    .implied_target_value
                    == next(e for e in omitted.edits if e.kind == "ancestor")
                    .implied_target_value)


def test_at_depth_one_there_is_nothing_to_omit():
    # No node stands between the ancestor and the target, so the flag is a no-op
    # and the two arms are the same trace. That is the experiment's own control:
    # any depth-1 difference would be an artefact of the flag itself.
    written, omitted = written_and_omitted(1)
    for one, two in zip(written, omitted):
        assert one.text == two.text
        assert one.omit == ()


def test_the_answer_and_the_patch_site_are_never_omitted():
    # The target's value is what we read, and the ancestor's is what we patch.
    for depth in (2, 3):
        for _, omitted in zip(*written_and_omitted(depth)):
            assert TARGET not in omitted.omit
            assert ANCESTOR not in omitted.omit
            assert len(omitted.omit) == depth - 1


def test_the_omitted_lines_still_carry_their_tag_last():
    # The surface control rewrites line tags, so every line keeps one -- and it
    # has to stay last on the line, which is why the pad goes before it: a
    # trailing marker merges with the newline and the token count stops matching.
    for _, omitted in zip(*written_and_omitted(3)):
        lines = dict(line.split(" = ", 1)[0:1] + [line]
                     for line in omitted.text.splitlines())
        for name in omitted.omit:
            assert lines[name].rstrip()[-1].isalpha()
            assert " #" in lines[name]


def test_cross_item_donors_still_work_under_omission():
    items = generate_items(char_encode, n_items=5, seed=0, gap=0, depth=2,
                           cross_item=True, generator=V3, omit="chain")
    for item in items:
        edit = next(e for e in item.edits if e.kind == "cross_item")
        assert len(edit.token_ids) == len(item.token_ids)
        assert len({edit.donor_raw_value, edit.implied_target_value,
                    item.target_value}) == 3


def test_the_decoy_mode_omits_as_many_values_off_the_path():
    # The control for the notation itself: same filler, same count, but the
    # target does not depend on those lines, so the answer stays computable from
    # what is written. A model that fails here fails at reading the format, not
    # at carrying a value.
    for depth in (2, 3):
        common = dict(n_items=5, seed=0, gap=0, generator=V3, depth=depth)
        chain = generate_items(char_encode, omit="chain", **common)
        decoy = generate_items(char_encode, omit="decoy", **common)
        written = generate_items(char_encode, **common)
        for one, two, three in zip(chain, decoy, written):
            assert len(one.omit) == len(two.omit) == depth - 1
            assert not set(one.omit) & set(two.omit)
            assert len(one.token_ids) == len(two.token_ids) == len(three.token_ids)
            # Nothing the target depends on is unwritten in the decoy arm.
            parents = {name for name, child in three.edges if child == TARGET}
            assert not set(two.omit) & ({ANCESTOR, TARGET} | parents)


def test_an_omitted_line_contributes_no_null_edit():
    # It has no result position left to rewrite. The null spread is then over
    # one fewer decoy, which the per-layer quorum takes from the row count.
    common = dict(n_items=5, seed=0, gap=0, generator=V3, depth=3)
    written = generate_items(char_encode, **common)
    decoy = generate_items(char_encode, omit="decoy", **common)
    for one, two in zip(written, decoy):
        nulls = lambda item: sum(e.kind == "null" for e in item.edits)
        assert nulls(one) - nulls(two) == 2


def test_an_unknown_omit_mode_is_refused():
    with pytest.raises(ValueError, match="unknown omit mode"):
        generate_items(char_encode, n_items=1, omit="everything")
