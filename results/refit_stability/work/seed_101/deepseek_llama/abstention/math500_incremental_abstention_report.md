# Incremental abstention analysis — deepseek_llama / math500 (L24)

Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.

`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.

## full_population

500 prompts; base accuracy 0.634; automatic failures 1; capped prompts 92; unparsed traces 229.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.751 | 0.247 | 0.211 | 0.610 |
| B1 | 0.786 | 0.212 | 0.202 | 0.589 |
| dumb_cap_count | 0.640 | 0.358 | 0.224 | 0.641 |
| dumb_unparsed_count | 0.636 | 0.362 | 0.225 | 0.643 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.035 [0.001, 0.068] p=0.040 | -0.035 [-0.072, -0.007] p=0.020 | -0.009 [-0.018, 0.001] p=0.120 | -0.021 [-0.042, 0.003] p=0.090 |

## valid_plurality

499 prompts; base accuracy 0.635; automatic failures 0; capped prompts 91; unparsed traces 221.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.751 | 0.247 | 0.211 | 0.611 |
| B1 | 0.786 | 0.212 | 0.203 | 0.589 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.035 [0.001, 0.068] p=0.050 | -0.035 [-0.067, -0.001] p=0.030 | -0.009 [-0.018, 0.002] p=0.120 | -0.022 [-0.047, 0.001] p=0.070 |

## cap_free_valid_plurality

408 prompts; base accuracy 0.674; automatic failures 0; capped prompts 0; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.755 | 0.243 | 0.213 | 0.616 |
| B1 | 0.795 | 0.202 | 0.205 | 0.598 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.040 [0.011, 0.070] p=0.010 | -0.040 [-0.073, -0.002] p=0.020 | -0.008 [-0.018, 0.002] p=0.150 | -0.018 [-0.045, 0.012] p=0.210 |

## cap_free_full_population

408 prompts; base accuracy 0.674; automatic failures 0; capped prompts 0; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.755 | 0.243 | 0.213 | 0.616 |
| B1 | 0.795 | 0.202 | 0.205 | 0.598 |
| dumb_cap_count | 0.633 | 0.365 | 0.221 | 0.633 |
| dumb_unparsed_count | 0.633 | 0.365 | 0.221 | 0.633 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.040 [0.011, 0.070] p=0.010 | -0.040 [-0.073, -0.002] p=0.020 | -0.008 [-0.018, 0.002] p=0.150 | -0.018 [-0.045, 0.012] p=0.210 |

## all_eight_parseable

411 prompts; base accuracy 0.672; automatic failures 0; capped prompts 3; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.755 | 0.242 | 0.213 | 0.614 |
| B1 | 0.796 | 0.202 | 0.204 | 0.596 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.041 [0.004, 0.081] p=0.020 | -0.041 [-0.080, -0.006] p=0.040 | -0.008 [-0.019, 0.003] p=0.190 | -0.019 [-0.046, 0.007] p=0.210 |
