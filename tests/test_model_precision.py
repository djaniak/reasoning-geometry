"""What precision the *model* runs at, and whether it is recorded.

The eight archived patching runs are bfloat16, which puts the digit logits on a
0.125-nat grid: eight of the fifty-three depth-1 readouts have two digits at
bit-identical probability, and a bare argmax was breaking those by digit order.
None of the eight says so. The dtype had to be traced through ``dag_patching ->
collect_data.load_model -> from_pretrained`` after the fact.

That call chain is also the correction this file carries. The dtype reaches
``from_pretrained``, so it sets every matmul in the forward pass; it is not a
choice about how the final softmax is rounded. Until 2026-08-15 the constant was
called ``READOUT_DTYPE`` and this file was called ``test_readout_precision``,
which licensed a claim neither can support -- that E2's missing ties were the
*recording* precision rather than the model. A float32 run and a bfloat16 run are
two different models' worth of arithmetic, and E2 moved both at once. What E2 does
establish is internal: both depths ran at the same precision, so the contrast
between them is not a precision artefact.

So two things are pinned here. The precision is *choosable*, because the E2 run
registered on 2026-08-15 loads the model in float32. And it is *recorded from the
model that ran*, not from the flag that asked, so no future reader has to trace a
call chain to find out what a number was measured at.
"""

import inspect
import sys
from pathlib import Path

import numpy
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.collect_data as collect_data
import dag.dag_patching as dag_patching


class FakeModel:
    def __init__(self, dtype):
        self.dtype = dtype
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


@pytest.fixture
def loaded(monkeypatch):
    """Capture the kwargs ``load_model`` hands to ``from_pretrained``."""
    seen = {}

    def fake_model(model_name, **kwargs):
        seen.update(kwargs)
        seen["model_name"] = model_name
        return FakeModel(kwargs.get("torch_dtype", torch.bfloat16))

    monkeypatch.setattr(collect_data.AutoModelForCausalLM,
                        "from_pretrained", staticmethod(fake_model))
    monkeypatch.setattr(collect_data.AutoTokenizer,
                        "from_pretrained", staticmethod(lambda *a, **k: object()))
    return seen


# --------------------------------------------------------------------------
# The precision is choosable, and choosing nothing changes nothing
# --------------------------------------------------------------------------


def test_the_default_is_still_bfloat16(loaded):
    """Every existing caller must keep loading exactly what it loaded before.

    The archived runs are only re-derivable if the default path is untouched.
    """
    collect_data.load_model(False, model_name="m")
    assert loaded["torch_dtype"] is torch.bfloat16


def test_a_requested_dtype_reaches_the_model(loaded):
    collect_data.load_model(False, model_name="m", dtype=torch.float32)
    assert loaded["torch_dtype"] is torch.float32


def test_a_requested_dtype_reaches_the_quantized_compute_path_too(loaded):
    """Otherwise ``--quantize`` would silently ignore the request.

    A flag that is honoured on one branch and dropped on the other is worse than
    one that does not exist, because the report would record a request that did
    not happen.
    """
    collect_data.load_model(True, model_name="m", dtype=torch.float32)
    assert loaded["quantization_config"].bnb_4bit_compute_dtype is torch.float32


# --------------------------------------------------------------------------
# What is recorded is what ran
# --------------------------------------------------------------------------


def test_the_recorded_precision_comes_from_the_model_not_the_request():
    """The failure this guards against is asking for float32 and getting bf16.

    A quantized load, a dtype the backend does not support, or a config that
    overrides the request all produce a model that is not what was asked for.
    Reading it back off the model is the only version of this field worth
    having.
    """
    assert dag_patching.model_dtype(FakeModel(torch.bfloat16)) == "bfloat16"
    assert dag_patching.model_dtype(FakeModel(torch.float32)) == "float32"


def test_the_patching_run_asks_for_float32():
    """Registered in `EXPERIMENT_LOG.md`, 2026-08-15, before E2 was generated.

    A name rather than a ``torch.dtype`` because ``dag_patching`` imports torch
    lazily, inside the functions that need it, so the gate logic stays testable
    without it. The name is resolved at load time.
    """
    assert dag_patching.MODEL_DTYPE == "float32"
    assert getattr(torch, dag_patching.MODEL_DTYPE) is torch.float32


def test_the_recorded_field_keeps_its_narrower_key():
    """``readout_dtype`` is the wrong name for the model's compute dtype, and it
    stays anyway.

    The committed E2 artifacts carry the key and `dag_pooling` refuses to pool
    across a difference in it, so renaming the key would either break those files
    or require rewriting them. The value was never wrong -- only the name was
    narrower than the thing it named -- so the fix is the docstring and the
    constant, not the schema.
    """
    assert "readout_dtype" in inspect.getsource(dag_patching.run)
    assert "model_dtype(model)" in inspect.getsource(dag_patching.run)


# --------------------------------------------------------------------------
# The readout was never the variable
# --------------------------------------------------------------------------


def test_the_readout_is_float32_whatever_the_model_ran_at():
    """`digit_readout` casts before it softmaxes, and always did.

    This is what makes "the ties were the recording precision" false rather than
    merely imprecise: the archived bfloat16 runs took their readout in float32
    too. The only thing E2 changed was the arithmetic that produced the logits.
    """
    torch.manual_seed(0)
    logits = (torch.randn(1, 4, 40) * 3).to(torch.bfloat16)
    probs, logodds, mass = dag_patching.digit_readout(logits, 2, list(range(10)))
    assert probs.dtype == numpy.float32
    assert logodds.dtype == numpy.float32


def test_a_float32_readout_cannot_tie_two_distinct_bfloat16_logits():
    """So an exact tie in an archived readout is a tie in the logits themselves.

    One step of the bfloat16 grid moves the probability by three orders of
    magnitude more than float32 can resolve at that probability, so two digits
    recorded at bit-identical probability were computed to bit-identical logits.
    No amount of readout precision would have separated them. Casting was never
    the fix; a different model run was.
    """
    torch.manual_seed(0)
    logits = (torch.randn(1, 4, 40) * 3).to(torch.bfloat16)
    stepped = logits.clone()
    stepped[0, 2, 5] = (logits[0, 2, 5].view(torch.int16) + 1).view(torch.bfloat16)

    before = float(dag_patching.digit_readout(logits, 2, list(range(10)))[0][5])
    after = float(dag_patching.digit_readout(stepped, 2, list(range(10)))[0][5])
    ulp = float(numpy.finfo(numpy.float32).eps) * before
    assert before != after
    assert abs(after - before) > 1000 * ulp
