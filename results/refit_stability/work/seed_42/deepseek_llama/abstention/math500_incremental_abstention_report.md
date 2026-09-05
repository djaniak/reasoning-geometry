# Incremental abstention analysis — deepseek_llama / math500 (L24)

Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.

`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.

## full_population

500 prompts; base accuracy 0.634; automatic failures 1; capped prompts 92; unparsed traces 229.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.750 | 0.248 | 0.210 | 0.608 |
| B1 | 0.797 | 0.201 | 0.197 | 0.577 |
| dumb_cap_count | 0.639 | 0.359 | 0.223 | 0.639 |
| dumb_unparsed_count | 0.638 | 0.360 | 0.224 | 0.641 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.047 [0.014, 0.077] p=0.000 | -0.047 [-0.075, -0.019] p=0.000 | -0.013 [-0.023, -0.002] p=0.010 | -0.031 [-0.059, -0.007] p=0.010 |

## valid_plurality

499 prompts; base accuracy 0.635; automatic failures 0; capped prompts 91; unparsed traces 221.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.750 | 0.248 | 0.211 | 0.609 |
| B1 | 0.797 | 0.201 | 0.197 | 0.578 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.047 [0.013, 0.074] p=0.000 | -0.047 [-0.077, -0.021] p=0.000 | -0.013 [-0.021, -0.003] p=0.010 | -0.031 [-0.055, -0.010] p=0.010 |

## cap_free_valid_plurality

408 prompts; base accuracy 0.674; automatic failures 0; capped prompts 0; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.761 | 0.237 | 0.211 | 0.611 |
| B1 | 0.817 | 0.181 | 0.197 | 0.575 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.056 [0.020, 0.092] p=0.000 | -0.056 [-0.091, -0.025] p=0.000 | -0.015 [-0.026, -0.002] p=0.020 | -0.037 [-0.067, -0.011] p=0.010 |

## cap_free_full_population

408 prompts; base accuracy 0.674; automatic failures 0; capped prompts 0; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.761 | 0.237 | 0.211 | 0.611 |
| B1 | 0.817 | 0.181 | 0.197 | 0.575 |
| dumb_cap_count | 0.634 | 0.364 | 0.222 | 0.635 |
| dumb_unparsed_count | 0.634 | 0.364 | 0.222 | 0.635 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.056 [0.020, 0.092] p=0.000 | -0.056 [-0.091, -0.025] p=0.000 | -0.015 [-0.026, -0.002] p=0.020 | -0.037 [-0.067, -0.011] p=0.010 |

## all_eight_parseable

411 prompts; base accuracy 0.672; automatic failures 0; capped prompts 3; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.762 | 0.235 | 0.210 | 0.609 |
| B1 | 0.817 | 0.180 | 0.196 | 0.573 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.055 [0.026, 0.085] p=0.010 | -0.055 [-0.093, -0.023] p=0.000 | -0.015 [-0.026, -0.005] p=0.010 | -0.037 [-0.066, -0.012] p=0.000 |
