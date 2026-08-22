# Written versus omitted intermediate results

Does stating the intermediate values suppress the ancestor patch? Eight arms
testing whether the depth collapse is about graph depth or about how much
correct arithmetic the trace has already written down. See `EXPERIMENT_LOG.md`,
2026-08-15, "Omitting the intermediate result does not restore the patch, and
the control says why."

## The manipulation

At depth 2 and 3 the trace states the chain's results. `--omit chain` renders
those lines without them:

```
m = a - 2 = 3 # w        ->    m = a - 2 # # # # w
```

The line still defines the node, so the graph and the value are unchanged; the
value is simply not written and has to be carried. The filler is comment markers
solved against the tokenizer to the exact token count of the ` = <digit>` it
replaces, so the two renderings are the same length -- 139 tokens either way at
depth 2 -- and every position downstream is unmoved. `_omission_pad` rejects
rather than emit a pad that is a token short.

Omission is a rendering choice and draws nothing from the random stream, so the
omitted batch is the **same batch** as the written one, item for item.

| File | Arm |
|:---|:---|
| `depth{1,2,3}_none.json` | written: every line states its result |
| `depth{1,2,3}_chain.json` | the ancestor-to-target values unwritten |
| `depth{2,3}_decoy.json` | as many values unwritten, off the path |

Two controls, and they carry the result:

- **Depth 1** has no chain, so `--omit chain` is a no-op there. `depth1_none`
  and `depth1_chain` are identical files apart from the recorded flag. Any
  depth-1 difference would have been an artefact of the manipulation itself.
- **`decoy`** omits the same number of values from lines the target does not
  depend on. Same notation, same token budget, but the answer stays computable
  from what is written. It separates "the model cannot read this format" from
  "the model needs that particular value".

An omitted line has no result position left to rewrite, so it contributes no
null edit: the decoy arms have 35 and 30 control rows where the others have 40.

```
uv run python dag_patching.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --condition both --seed 0 --n_items 5 --depth 2 --gap 0 \
  --generator v3_distinct --omit chain \
  --output results/dag_patching/written_vs_omitted/depth2_chain.json
```

Seed 0, `n_items 5`, `n_decoys 6`, `condition both`, generator `v3_distinct`.
Not in `MANIFEST.json`, which covers only the eight archived runs.

## What the arms say

At layer 13, the discovery layer fixed from the `v3_distinct` table:

| Arm | verdict | clean top = target | clean p(target) | ancestor lands on implied | ancestor moved | controls moved |
|:---|:---|---:|---:|---:|---:|---:|
| `depth1_none` | positive | 4/5 | 0.666 | **5/5** | 5/5 | 4/40 |
| `depth1_chain` | positive | 4/5 | 0.666 | **5/5** | 5/5 | 4/40 |
| `depth2_none` | negative | 5/5 | 0.996 | 0/5 | 0/5 | 0/40 |
| `depth2_decoy` | negative | 5/5 | 0.997 | 0/5 | 0/5 | 0/35 |
| `depth2_chain` | *positive* | 2/5 | 0.240 | 1/5 | 4/5 | **23/40** |
| `depth3_none` | negative | 5/5 | 0.999 | 0/5 | 0/5 | 0/40 |
| `depth3_decoy` | negative | 5/5 | 0.999 | 0/5 | 0/5 | 0/30 |
| `depth3_chain` | negative | 1/5 | 0.050 | 0/5 | 4/5 | **33/40** |

**`depth2_chain`'s stored verdict is `positive` and should not be believed.**
Every gate in the scorer is relative, so an arm where the background moves as
much as the ancestor does clears all of them. The `control_specificity`
diagnostic beside the verdict is what says otherwise: nulls flip the answer
23/40 and a comment-tag rewrite flips it 3/5, while the ancestor lands on the
digit it predicts only 1/5 of the time. That is a fragile readout, not a
propagated value. The verdict is left as the scorer produced it rather than
patched by hand; the diagnostic is in the same file.

## Reading it

The decoy control is what makes the **clean-behaviour ablation** interpretable.
With the same notation and the same token count, but the unwritten values off
the dependency path, the model is at 5/5 clean and p(target) 0.997 --
**identical to the written arm**. The format is legible, so the collapse in the
`chain` arms, to 2/5 and 1/5 with p(target) 0.240 and 0.050, is not the model
failing to read ` # # # #`.

It does not make the *patching* arms interpretable. Those hit the pre-registered
stop condition, and a control added afterwards does not convert a stopped
contrast into a valid causal test.

Two limits on how far the decoy carries even for the ablation. It omits
`decoys[:depth-1]` -- the first decoys, not position-matched substitutes for the
path lines -- so it isolates notation legibility rather than every path-specific
effect. And the model still answers correctly 2/5 and 1/5 with the path value
unwritten: clean behaviour *collapses*, it does not vanish.

What the arms support is that **no behaviourally usable carried intermediate was
detected**. That is weaker than "there is no latent computation", and
deliberately so: a behavioural failure after removing a written value cannot
separate computing the value from binding, retaining, or retrieving it, and
nothing here read an activation at a matched slot for the omitted result. The
mechanism the depth collapse is consistent with -- the answer at depth 2 and 3
being fixed by the written intermediate token, which the patch does not touch --
remains a hypothesis this experiment did not test.

Five items, one seed, one checkpoint. The counts are descriptive.

The readout is bfloat16, so the digit logits sit on a 0.125-nat grid and exact
two-digit ties are ordinary. `EXPERIMENT_LOG.md` carries the tie-aware counts;
the `ancestor -> implied` column above is a bare argmax.
