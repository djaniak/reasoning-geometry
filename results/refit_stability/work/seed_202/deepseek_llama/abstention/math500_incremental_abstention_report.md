# Incremental abstention analysis — deepseek_llama / math500 (L24)

Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.

`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.

## full_population

500 prompts; base accuracy 0.634; automatic failures 1; capped prompts 92; unparsed traces 229.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.757 | 0.241 | 0.209 | 0.605 |
| B1 | 0.786 | 0.212 | 0.199 | 0.582 |
| dumb_cap_count | 0.638 | 0.360 | 0.223 | 0.638 |
| dumb_unparsed_count | 0.634 | 0.364 | 0.224 | 0.640 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.029 [0.001, 0.056] p=0.030 | -0.029 [-0.057, 0.001] p=0.060 | -0.010 [-0.020, 0.001] p=0.060 | -0.023 [-0.046, -0.002] p=0.020 |

## valid_plurality

499 prompts; base accuracy 0.635; automatic failures 0; capped prompts 91; unparsed traces 221.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.757 | 0.241 | 0.209 | 0.606 |
| B1 | 0.786 | 0.212 | 0.199 | 0.583 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.029 [-0.002, 0.055] p=0.060 | -0.029 [-0.060, -0.003] p=0.030 | -0.010 [-0.019, 0.001] p=0.090 | -0.023 [-0.046, 0.003] p=0.070 |

## cap_free_valid_plurality

408 prompts; base accuracy 0.674; automatic failures 0; capped prompts 0; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.775 | 0.222 | 0.209 | 0.606 |
| B1 | 0.801 | 0.197 | 0.200 | 0.585 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.026 [-0.007, 0.058] p=0.150 | -0.026 [-0.059, 0.009] p=0.100 | -0.009 [-0.019, 0.005] p=0.220 | -0.021 [-0.049, 0.006] p=0.150 |

## cap_free_full_population

408 prompts; base accuracy 0.674; automatic failures 0; capped prompts 0; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.775 | 0.222 | 0.209 | 0.606 |
| B1 | 0.801 | 0.197 | 0.200 | 0.585 |
| dumb_cap_count | 0.633 | 0.365 | 0.220 | 0.632 |
| dumb_unparsed_count | 0.633 | 0.365 | 0.220 | 0.632 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.026 [-0.007, 0.058] p=0.150 | -0.026 [-0.059, 0.009] p=0.100 | -0.009 [-0.019, 0.005] p=0.220 | -0.021 [-0.049, 0.006] p=0.150 |

## all_eight_parseable

411 prompts; base accuracy 0.672; automatic failures 0; capped prompts 3; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.775 | 0.223 | 0.209 | 0.605 |
| B1 | 0.801 | 0.196 | 0.200 | 0.583 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.026 [-0.014, 0.066] p=0.110 | -0.026 [-0.063, 0.002] p=0.100 | -0.009 [-0.021, 0.004] p=0.120 | -0.022 [-0.049, 0.004] p=0.110 |
