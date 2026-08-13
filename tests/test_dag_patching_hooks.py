"""Hook mechanics for the DAG patching prototype, on CPU with a toy model.

The toy model is a real ``nn.Module`` stack with the same ``model.model.layers``
shape the runner hooks, and it is causal. That is enough to test everything that
can silently go wrong with the intervention: reading and writing at the same
site, position targeting, causal masking, and hook cleanup. The science needs a
real checkpoint; the mechanics do not.

Skipped where torch is absent. ``pyproject.toml`` deliberately does not pin the
GPU stack, so these run on the machine that runs the experiment.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from dag_patching import (  # noqa: E402
    capture_states,
    digit_readout,
    evaluate_gates,
    identity_patch_check,
    layer_bins,
    measure_item,
    run_patched,
    verdict,
)

VOCAB = 32
HIDDEN = 8


class ToyLayer(torch.nn.Module):
    """Causal mixer: each position sees itself and everything before it."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN)

    def forward(self, hidden, **kwargs):
        cumulative = hidden.cumsum(dim=1) / torch.arange(
            1, hidden.shape[1] + 1, device=hidden.device
        ).view(1, -1, 1)
        return (torch.tanh(self.proj(hidden + cumulative)),)


class ToyInner(torch.nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.layers = torch.nn.ModuleList(ToyLayer() for _ in range(n_layers))

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


class ToyOutput:
    def __init__(self, logits):
        self.logits = logits


class ToyModel(torch.nn.Module):
    def __init__(self, n_layers=4):
        super().__init__()
        self.model = ToyInner(n_layers)
        self.head = torch.nn.Linear(HIDDEN, VOCAB)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, **kwargs):
        return ToyOutput(self.head(self.model(input_ids)))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ToyModel().eval()


@pytest.fixture
def tokens():
    return list(range(1, 17))


def test_identity_patch_leaves_the_logits_unchanged(model, tokens):
    result = identity_patch_check(model, tokens, layer_bins(4), [3, 5])
    assert result["passes"]
    assert result["max_abs_logit_change"] < 1e-5


def test_identity_patch_fails_when_read_and_write_sites_differ(model, tokens):
    # Simulates the off-by-one: capture at one layer, write at another.
    bins = layer_bins(4)
    states, clean = capture_states(model, tokens, bins, [3])
    patched = run_patched(model, tokens, bins[1], [3], states[bins[0]])
    assert (patched - clean).abs().max() > 1e-5


def test_patching_a_different_state_changes_the_logits(model, tokens):
    bins = layer_bins(4)
    donor_states, _ = capture_states(model, list(reversed(tokens)), bins, [3])
    _, clean = capture_states(model, tokens, bins, [3])
    patched = run_patched(model, tokens, bins[0], [3], donor_states[bins[0]])
    assert (patched - clean).abs().max() > 1e-5


def test_a_patch_cannot_affect_earlier_positions(model, tokens):
    # Causal masking. If this fails, every "effect" the runner measures could be
    # an artefact rather than propagation along the trace.
    bins = layer_bins(4)
    donor_states, _ = capture_states(model, list(reversed(tokens)), bins, [9])
    _, clean = capture_states(model, tokens, bins, [9])
    patched = run_patched(model, tokens, bins[0], [9], donor_states[bins[0]])
    assert torch.allclose(patched[:, :9], clean[:, :9], atol=1e-6)
    assert (patched[:, 9:] - clean[:, 9:]).abs().max() > 1e-5


def test_hooks_are_removed_after_each_call(model, tokens):
    bins = layer_bins(4)
    states, _ = capture_states(model, tokens, bins, [3])
    run_patched(model, tokens, bins[0], [3], states[bins[0]])
    for layer in model.model.layers:
        assert not layer._forward_hooks


def test_hooks_are_removed_even_when_the_forward_raises(model, tokens):
    class Exploding(ToyModel):
        def forward(self, input_ids, **kwargs):
            self.model(input_ids)  # let the hooks fire, then fail
            raise RuntimeError("boom")

    torch.manual_seed(0)
    exploding = Exploding().eval()
    with pytest.raises(RuntimeError, match="boom"):
        capture_states(exploding, tokens, layer_bins(4), [3])
    for layer in exploding.model.layers:
        assert not layer._forward_hooks


def test_capture_returns_one_state_per_bin_shaped_by_positions(model, tokens):
    bins = layer_bins(4)
    states, _ = capture_states(model, tokens, bins, [2, 7])
    assert set(states) == set(bins)
    for tensor in states.values():
        assert tuple(tensor.shape) == (1, 2, HIDDEN)


def test_digit_readout_normalises_over_the_ten_digits(model, tokens):
    logits = model(torch.as_tensor([tokens])).logits
    probs, logodds, mass = digit_readout(logits, 5, list(range(10)))
    assert probs.shape == (10,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= mass <= 1.0


def toy_encode(text):
    """Toy-vocabulary encoder that keeps the real space-merge rule.

    A space merges with a following letter but not with a following digit, as in
    the Qwen tokenizers. Dropping that rule here would let these tests pass on
    traces the real checkpoints reject.
    """
    tokens, index = [], 0
    while index < len(text):
        if text[index] == " " and index + 1 < len(text) and text[index + 1].isalpha():
            tokens.append(text[index:index + 2])
            index += 2
        else:
            tokens.append(text[index])
            index += 1
    return [int.from_bytes(token.encode(), "big") % VOCAB for token in tokens]


def toy_items(n_items=1):
    """Generator items encoded into the toy model's vocabulary."""
    from dag_tasks import generate_items

    return generate_items(toy_encode, n_items=n_items, seed=0)


TOY_DIGIT_IDS = [ord(str(d)) % VOCAB for d in range(10)]


def test_patching_all_positions_of_a_real_generated_edit(model):
    # The two edited positions patch together and move the read position.
    item = toy_items()[0]
    edit = next(e for e in item.edits if e.kind == "ancestor")
    bins = layer_bins(4)
    donor_states, _ = capture_states(model, edit.token_ids, bins, list(edit.positions))
    _, clean = capture_states(model, item.token_ids, bins, [item.read_position])
    patched = run_patched(
        model, item.token_ids, bins[0], list(edit.positions), donor_states[bins[0]]
    )
    assert (patched[:, item.read_position] - clean[:, item.read_position]).abs().max() > 1e-6


def test_measure_item_wires_through_to_a_verdict(model):
    # Integration: generator item -> patching -> rows -> gates -> verdict. The
    # toy model has no arithmetic, so the verdict itself is meaningless; what is
    # tested is that the shapes, keys, and bookkeeping line up.
    item = toy_items()[0]
    bins = layer_bins(4)
    rows, summary = measure_item(model, item, bins, TOY_DIGIT_IDS)

    assert len(rows) == len(item.edits) * len(bins)
    assert {row["layer"] for row in rows} == set(bins)
    assert summary["target_value"] == item.target_value
    for row in rows:
        assert 0.0 <= row["tv"] <= 1.0
        assert 0.0 <= row["digit_mass_patched"] <= 1.0

    assert verdict(evaluate_gates([rows], bins)) in {
        "positive", "scientific negative", "invalid test"
    }


def test_measure_item_rejects_a_donor_of_the_wrong_length(model):
    import dataclasses

    item = toy_items()[0]
    broken = dataclasses.replace(
        item,
        edits=(dataclasses.replace(
            item.edits[0], token_ids=item.edits[0].token_ids[:-1]
        ),),
    )
    with pytest.raises(ValueError, match="differ in length"):
        measure_item(model, broken, layer_bins(4), TOY_DIGIT_IDS)


def test_measure_item_rejects_an_edit_at_or_after_the_read_position(model):
    import dataclasses

    item = toy_items()[0]
    broken = dataclasses.replace(
        item,
        edits=(dataclasses.replace(
            item.edits[0], positions=(item.read_position,)
        ),),
    )
    with pytest.raises(ValueError, match="not upstream"):
        measure_item(model, broken, layer_bins(4), TOY_DIGIT_IDS)
