# Budget-indexed outcomes for the RMD prompt-level result

Built by `analysis/budget_outcomes.py` from committed results and OOF rows.
The operating points rebuild the frozen prompt-level readouts; hidden-state
references and generation remain fixed.

`C_B` is correctness available by the generation budget under the fixed
decoding and extraction rule. It is the `full_population` row: an unparsed
prompt is an observed failure, not a missing value. Every row below it is a
conditional population, and `cap_free_valid_plurality` -- the current headline
-- conditions on a difficulty-related event.

## 1. Selection ladder

| Model | Population | n | Retained | Base acc. | Capped prompts | Auto-failures |
|---|---|---:|---:|---:|---:|---:|
| qwen | `full_population` | 500 | 100.0% | 0.620 | 108 | 2 |
| qwen | `valid_plurality` | 498 | 99.6% | 0.622 | 106 | 0 |
| qwen | `cap_free_valid_plurality` | 392 | 78.4% | 0.691 | 0 | 0 |
| qwen | `all_eight_parseable` | 392 | 78.4% | 0.691 | 1 | 0 |
| deepseek | `full_population` | 500 | 100.0% | 0.750 | 107 | 7 |
| deepseek | `valid_plurality` | 493 | 98.6% | 0.761 | 100 | 0 |
| deepseek | `cap_free_valid_plurality` | 393 | 78.6% | 0.796 | 0 | 0 |
| deepseek | `all_eight_parseable` | 384 | 76.8% | 0.799 | 4 | 0 |
| deepseek_llama | `full_population` | 500 | 100.0% | 0.634 | 92 | 1 |
| deepseek_llama | `valid_plurality` | 499 | 99.8% | 0.635 | 91 | 0 |
| deepseek_llama | `cap_free_valid_plurality` | 408 | 81.6% | 0.674 | 0 | 0 |
| deepseek_llama | `all_eight_parseable` | 411 | 82.2% | 0.672 | 3 | 0 |

## 2. Does the increment survive the outcome definition?

Paired bootstrap `B1 - B0` on AURC. Negative favours B1 (RMD added to the
output-side baseline). These intervals hold the fitted pipeline fixed and so
understate uncertainty; see the outer-refit blocker.
`vs headline` is the increment as a fraction of the `cap_free_valid_plurality`
estimate. Below 100% means the headline population overstates the increment.

| Model | Population | B1-B0 AURC | 95% CI | p | CI excludes 0 | vs headline |
|---|---|---:|---|---:|:--:|---:|
| qwen | `full_population` | -0.0520 | [-0.0845, -0.0218] | 0.0000 | yes | 89% |
| qwen | `valid_plurality` | -0.0518 | [-0.0835, -0.0201] | 0.0040 | yes | 89% |
| qwen | `cap_free_valid_plurality` | -0.0585 | [-0.1026, -0.0182] | 0.0040 | yes | 100% |
| qwen | `all_eight_parseable` | -0.0586 | [-0.0972, -0.0233] | 0.0000 | yes | 100% |
| deepseek | `full_population` | -0.0284 | [-0.0526, -0.0048] | 0.0100 | yes | 80% |
| deepseek | `valid_plurality` | -0.0290 | [-0.0541, -0.0061] | 0.0120 | yes | 82% |
| deepseek | `cap_free_valid_plurality` | -0.0355 | [-0.0642, -0.0097] | 0.0040 | yes | 100% |
| deepseek | `all_eight_parseable` | -0.0387 | [-0.0670, -0.0094] | 0.0040 | yes | 109% |
| deepseek_llama | `full_population` | -0.0469 | [-0.0743, -0.0162] | 0.0040 | yes | 84% |
| deepseek_llama | `valid_plurality` | -0.0471 | [-0.0777, -0.0172] | 0.0020 | yes | 84% |
| deepseek_llama | `cap_free_valid_plurality` | -0.0560 | [-0.0910, -0.0232] | 0.0000 | yes | 100% |
| deepseek_llama | `all_eight_parseable` | -0.0547 | [-0.0915, -0.0206] | 0.0000 | yes | 98% |

## 3. Operational accuracy after abstention

Accuracy among answered prompts on `full_population`. Each readout ranks all
500 prompts; the protocol abstains on the lowest-ranked fraction. Intervals
are pointwise prompt-bootstrap intervals with the fitted pipeline held fixed.

| Model | Readout | 0% abstain | 20% abstain | 50% abstain |
|---|---|---:|---:|---:|
| qwen | `B0` | 0.620 | 0.713 | 0.780 |
| qwen | `B1` | 0.620 | 0.730 | 0.876 |
| qwen | `entropy` | 0.620 | 0.670 | 0.684 |
| qwen | `length` | 0.620 | 0.690 | 0.744 |
| deepseek | `B0` | 0.750 | 0.805 | 0.836 |
| deepseek | `B1` | 0.750 | 0.818 | 0.880 |
| deepseek | `entropy` | 0.750 | 0.777 | 0.796 |
| deepseek | `length` | 0.750 | 0.797 | 0.828 |
| deepseek_llama | `B0` | 0.634 | 0.693 | 0.736 |
| deepseek_llama | `B1` | 0.634 | 0.703 | 0.796 |
| deepseek_llama | `entropy` | 0.634 | 0.647 | 0.656 |
| deepseek_llama | `length` | 0.634 | 0.657 | 0.696 |

## 4. Capped traces that still carry an answer

Their stopping time is censored; their answer at `B` is observed and is
scored. Dropping them treats an observed outcome as missing.

| Model | Cap | Provenance | Traces | Capped | Capped & parseable | Capped & unparsed | Acc. capped-parseable | Acc. uncapped-parseable |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| qwen | 1024 | confirmed by dvc.lock, dvc.yaml/params.yaml | 4000 | 338 | 11 | 327 | 0.000 | 0.608 |
| deepseek | 8192 | confirmed by dvc.lock, dvc.yaml/params.yaml | 4000 | 374 | 38 | 336 | 0.553 | 0.771 |
| deepseek_llama | 12288 | confirmed by dvc.lock, dvc.yaml/params.yaml | 4000 | 250 | 21 | 229 | 0.286 | 0.587 |

## 5. Continuation case study, C_{B->B'}

DeepSeek only. 50 traces sampled from the capped
population and resumed from 8192 for a further
8192 tokens at temperature 0.6.
This is a one-model sensitivity case. It is not a label for the other two
models, for the unsampled capped traces, or for the dataset.

| Outcome | n | Share |
|---|---:|---:|
| completed_correct | 16 | 32.0% |
| completed_incorrect | 18 | 36.0% |
| still_unfinished | 13 | 26.0% |
| degenerate_loop | 3 | 6.0% |

Of the 50 resumed traces, 34 are labelled completed (correct or incorrect), and 0.471 of those are correct.

The stored `accuracy_of_completions` is 0.4571, which is a different quantity: `continue_capped` divides by traces that *terminated*, and a degenerate loop that ran to a stop sits in that denominator while never counting as correct. Both are defensible; say which one is being quoted.

## Reporting rule

1. Report `full_population` (`C_B`) as the primary outcome.
2. Report cap-free numbers as conditional, with the retained fraction beside them.
3. Report the continuation study separately, as DeepSeek-only evidence about
   what capped prefixes do next -- never as `C_B` for any population.
