# Peer baseline and cost ladder

Every rung scored against what it costs at decision time. `B1` adds `rmd_tail_q20` to `B0` and buys **no** extra generations -- it reads hidden states of the eight the target already produced -- so it is free at the margin and no peer rung is cost matched to it.

Two readouts of the same purchased generations:

* **`graded`** -- the fraction of the peer's samples that were *correct*. Needs the gold answer, so it is **not computable at decision time**. It is an upper bound on what any peer method could deliver, not a baseline.
* **`agree`** -- the fraction of the peer's samples that returned the *target's own answer*. No gold needed; this is the peer ensemble a reviewer could actually deploy.

AURC, lower is better. A negative `B1 - rung` favours `B1`.

Headline population: `full_population`. Ladder sizes: 1, 2, 4, 8 siblings per peer.

## 1. Floors and the two free rungs

AURC does not bottom out at zero. Where a rung approaches the oracle floor, a delta can no longer separate "this is better" from "there was nothing left to remove".

| Model | n | Base acc. | Oracle AURC | `B0` | `B1` | B0 headroom |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 500 | 0.620 | 0.0836 | 0.2299 | 0.1779 | 0.1463 |
| deepseek | 500 | 0.750 | 0.0342 | 0.1519 | 0.1280 | 0.1177 |
| deepseek_llama | 500 | 0.634 | 0.0771 | 0.2471 | 0.2117 | 0.1700 |

## 2. The ladder in cost order

`Extra` is the margin over `B0` per prompt. `AURC` is the mean over 25 independent re-draws of which siblings were bought (sub-sampled rungs only; the full-cache rungs are deterministic). `removed` is the fraction of `B0`'s headroom the rung takes out -- at 1.00 the rung is on the oracle floor.

| Model | Rung | Kind | Extra calls | Extra tokens | AURC | across-draw range | removed |
|---|---|---|---:|---:|---:|---|---:|
| qwen | `B0` | none | 0 | 0 | 0.2299 | -- | 0.00 |
| qwen | `B1` | none | 0 | 0 | 0.1779 | -- | 0.36 |
| qwen | `B0_agree_deepseek_llama_m1` | agree | 1 | 3049 | 0.1475 | [0.1381, 0.1536] | 0.56 |
| qwen | `B0_agree_deepseek_m1` | agree | 1 | 3101 | 0.1372 | [0.1299, 0.1473] | 0.63 |
| qwen | `B0_graded_deepseek_llama_m1` | graded | 1 | 3049 | 0.1509 | [0.1389, 0.1586] | 0.54 |
| qwen | `B0_graded_deepseek_m1` | graded | 1 | 3101 | 0.1234 | [0.1171, 0.1311] | 0.73 |
| qwen | `B0_agree_both_m1` | agree | 2 | 6150 | 0.1324 | [0.1260, 0.1383] | 0.67 |
| qwen | `B0_agree_deepseek_llama_m2` | agree | 2 | 6099 | 0.1370 | [0.1268, 0.1448] | 0.64 |
| qwen | `B0_agree_deepseek_m2` | agree | 2 | 6202 | 0.1371 | [0.1271, 0.1470] | 0.63 |
| qwen | `B0_graded_both_m1` | graded | 2 | 6150 | 0.1223 | [0.1162, 0.1298] | 0.74 |
| qwen | `B0_graded_deepseek_llama_m2` | graded | 2 | 6099 | 0.1417 | [0.1371, 0.1490] | 0.60 |
| qwen | `B0_graded_deepseek_m2` | graded | 2 | 6202 | 0.1179 | [0.1126, 0.1223] | 0.77 |
| qwen | `B0_agree_both_m2` | agree | 4 | 12301 | 0.1266 | [0.1140, 0.1350] | 0.71 |
| qwen | `B0_agree_deepseek_llama_m4` | agree | 4 | 12197 | 0.1312 | [0.1262, 0.1392] | 0.67 |
| qwen | `B0_agree_deepseek_m4` | agree | 4 | 12404 | 0.1327 | [0.1265, 0.1419] | 0.66 |
| qwen | `B0_graded_both_m2` | graded | 4 | 12301 | 0.1188 | [0.1144, 0.1236] | 0.76 |
| qwen | `B0_graded_deepseek_llama_m4` | graded | 4 | 12197 | 0.1398 | [0.1374, 0.1422] | 0.62 |
| qwen | `B0_graded_deepseek_m4` | graded | 4 | 12404 | 0.1149 | [0.1131, 0.1185] | 0.79 |
| qwen | `B0_agree_both_m4` | agree | 8 | 24601 | 0.1242 | [0.1145, 0.1327] | 0.72 |
| qwen | `B0_agree_deepseek_llama_m8` | agree | 8 | 24394 | 0.1275 | -- | 0.70 |
| qwen | `B0_agree_deepseek_m8` | agree | 8 | 24808 | 0.1295 | -- | 0.69 |
| qwen | `B0_graded_both_m4` | graded | 8 | 24601 | 0.1171 | [0.1154, 0.1190] | 0.77 |
| qwen | `B0_graded_deepseek_llama_m8` | graded | 8 | 24394 | 0.1385 | -- | 0.62 |
| qwen | `B0_graded_deepseek_m8` | graded | 8 | 24808 | 0.1135 | -- | 0.80 |
| qwen | `B0_agree_both_m8` | agree | 16 | 49202 | 0.1240 | -- | 0.72 |
| qwen | `B0_graded_both_m8` | graded | 16 | 49202 | 0.1159 | -- | 0.78 |
| deepseek | `B0` | none | 0 | 0 | 0.1519 | -- | 0.00 |
| deepseek | `B1` | none | 0 | 0 | 0.1280 | -- | 0.20 |
| deepseek | `B0_agree_deepseek_llama_m1` | agree | 1 | 3049 | 0.1565 | [0.1514, 0.1628] | -0.04 |
| deepseek | `B0_agree_qwen_m1` | agree | 1 | 556 | 0.1212 | [0.1116, 0.1408] | 0.26 |
| deepseek | `B0_graded_deepseek_llama_m1` | graded | 1 | 3049 | 0.0748 | [0.0663, 0.0833] | 0.66 |
| deepseek | `B0_graded_qwen_m1` | graded | 1 | 556 | 0.0797 | [0.0637, 0.0940] | 0.61 |
| deepseek | `B0_agree_both_m1` | agree | 2 | 3606 | 0.1209 | [0.0998, 0.1515] | 0.26 |
| deepseek | `B0_agree_deepseek_llama_m2` | agree | 2 | 6099 | 0.1564 | [0.1510, 0.1616] | -0.04 |
| deepseek | `B0_agree_qwen_m2` | agree | 2 | 1112 | 0.1176 | [0.1117, 0.1265] | 0.29 |
| deepseek | `B0_graded_both_m1` | graded | 2 | 3606 | 0.0531 | [0.0466, 0.0600] | 0.84 |
| deepseek | `B0_graded_deepseek_llama_m2` | graded | 2 | 6099 | 0.0649 | [0.0612, 0.0691] | 0.74 |
| deepseek | `B0_graded_qwen_m2` | graded | 2 | 1112 | 0.0660 | [0.0564, 0.0769] | 0.73 |
| deepseek | `B0_agree_both_m2` | agree | 4 | 7211 | 0.1127 | [0.1031, 0.1248] | 0.33 |
| deepseek | `B0_agree_deepseek_llama_m4` | agree | 4 | 12197 | 0.1567 | [0.1532, 0.1606] | -0.04 |
| deepseek | `B0_agree_qwen_m4` | agree | 4 | 2225 | 0.1204 | [0.1148, 0.1248] | 0.27 |
| deepseek | `B0_graded_both_m2` | graded | 4 | 7211 | 0.0464 | [0.0435, 0.0521] | 0.90 |
| deepseek | `B0_graded_deepseek_llama_m4` | graded | 4 | 12197 | 0.0614 | [0.0592, 0.0656] | 0.77 |
| deepseek | `B0_graded_qwen_m4` | graded | 4 | 2225 | 0.0568 | [0.0502, 0.0643] | 0.81 |
| deepseek | `B0_agree_both_m4` | agree | 8 | 14422 | 0.1120 | [0.1051, 0.1222] | 0.34 |
| deepseek | `B0_agree_deepseek_llama_m8` | agree | 8 | 24394 | 0.1569 | -- | -0.04 |
| deepseek | `B0_agree_qwen_m8` | agree | 8 | 4450 | 0.1215 | -- | 0.26 |
| deepseek | `B0_graded_both_m4` | graded | 8 | 14422 | 0.0434 | [0.0416, 0.0488] | 0.92 |
| deepseek | `B0_graded_deepseek_llama_m8` | graded | 8 | 24394 | 0.0597 | -- | 0.78 |
| deepseek | `B0_graded_qwen_m8` | graded | 8 | 4450 | 0.0498 | -- | 0.87 |
| deepseek | `B0_agree_both_m8` | agree | 16 | 28844 | 0.1125 | -- | 0.33 |
| deepseek | `B0_graded_both_m8` | graded | 16 | 28844 | 0.0411 | -- | 0.94 |
| deepseek_llama | `B0` | none | 0 | 0 | 0.2471 | -- | 0.00 |
| deepseek_llama | `B1` | none | 0 | 0 | 0.2117 | -- | 0.21 |
| deepseek_llama | `B0_agree_deepseek_m1` | agree | 1 | 3101 | 0.2129 | [0.1768, 0.2291] | 0.20 |
| deepseek_llama | `B0_agree_qwen_m1` | agree | 1 | 556 | 0.1485 | [0.1405, 0.1668] | 0.58 |
| deepseek_llama | `B0_graded_deepseek_m1` | graded | 1 | 3101 | 0.1659 | [0.1528, 0.1741] | 0.48 |
| deepseek_llama | `B0_graded_qwen_m1` | graded | 1 | 556 | 0.1929 | [0.1770, 0.2162] | 0.32 |
| deepseek_llama | `B0_agree_both_m1` | agree | 2 | 3657 | 0.1397 | [0.1302, 0.1543] | 0.63 |
| deepseek_llama | `B0_agree_deepseek_m2` | agree | 2 | 6202 | 0.2166 | [0.1980, 0.2278] | 0.18 |
| deepseek_llama | `B0_agree_qwen_m2` | agree | 2 | 1112 | 0.1423 | [0.1356, 0.1502] | 0.62 |
| deepseek_llama | `B0_graded_both_m1` | graded | 2 | 3657 | 0.1669 | [0.1558, 0.1773] | 0.47 |
| deepseek_llama | `B0_graded_deepseek_m2` | graded | 2 | 6202 | 0.1618 | [0.1471, 0.1758] | 0.50 |
| deepseek_llama | `B0_graded_qwen_m2` | graded | 2 | 1112 | 0.1892 | [0.1762, 0.1989] | 0.34 |
| deepseek_llama | `B0_agree_both_m2` | agree | 4 | 7314 | 0.1378 | [0.1291, 0.1445] | 0.64 |
| deepseek_llama | `B0_agree_deepseek_m4` | agree | 4 | 12404 | 0.2083 | [0.1960, 0.2280] | 0.23 |
| deepseek_llama | `B0_agree_qwen_m4` | agree | 4 | 2225 | 0.1406 | [0.1370, 0.1484] | 0.63 |
| deepseek_llama | `B0_graded_both_m2` | graded | 4 | 7314 | 0.1644 | [0.1539, 0.1707] | 0.49 |
| deepseek_llama | `B0_graded_deepseek_m4` | graded | 4 | 12404 | 0.1625 | [0.1538, 0.1735] | 0.50 |
| deepseek_llama | `B0_graded_qwen_m4` | graded | 4 | 2225 | 0.1879 | [0.1796, 0.2001] | 0.35 |
| deepseek_llama | `B0_agree_both_m4` | agree | 8 | 14629 | 0.1369 | [0.1316, 0.1408] | 0.65 |
| deepseek_llama | `B0_agree_deepseek_m8` | agree | 8 | 24808 | 0.2065 | -- | 0.24 |
| deepseek_llama | `B0_agree_qwen_m8` | agree | 8 | 4450 | 0.1392 | -- | 0.63 |
| deepseek_llama | `B0_graded_both_m4` | graded | 8 | 14629 | 0.1659 | [0.1577, 0.1733] | 0.48 |
| deepseek_llama | `B0_graded_deepseek_m8` | graded | 8 | 24808 | 0.1597 | -- | 0.51 |
| deepseek_llama | `B0_graded_qwen_m8` | graded | 8 | 4450 | 0.1865 | -- | 0.36 |
| deepseek_llama | `B0_agree_both_m8` | agree | 16 | 29258 | 0.1376 | -- | 0.64 |
| deepseek_llama | `B0_graded_both_m8` | graded | 16 | 29258 | 0.1624 | -- | 0.50 |

## 3. Does a bought rung beat the free one?

`B1 - rung` on AURC. Negative favours `B1`, the rung that costs no extra generations. The interval is the frozen prompt bootstrap on the median draw; `sign stable` reports whether every draw agreed on the direction, which is the separate question of whether the verdict depends on which siblings you happened to buy.

| Model | Rung | Kind | Extra calls | B1 - rung | 95% CI | excludes 0 | sign stable | winner |
|---|---|---|---:|---:|---|:--:|:--:|---|
| qwen | `B0` | none | 0 | -0.0520 | [-0.0842, -0.0162] | yes | -- | B1 |
| qwen | `B0_agree_deepseek_llama_m1` | agree | 1 | 0.0312 | [-0.0031, +0.0610] | no | yes | tie |
| qwen | `B0_agree_deepseek_m1` | agree | 1 | 0.0413 | [+0.0173, +0.0683] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m1` | graded | 1 | 0.0264 | [-0.0061, +0.0551] | no | yes | tie |
| qwen | `B0_graded_deepseek_m1` | graded | 1 | 0.0547 | [+0.0242, +0.0854] | yes | yes | peer |
| qwen | `B0_agree_both_m1` | agree | 2 | 0.0456 | [+0.0195, +0.0841] | yes | yes | peer |
| qwen | `B0_agree_deepseek_llama_m2` | agree | 2 | 0.0400 | [+0.0058, +0.0738] | yes | yes | peer |
| qwen | `B0_agree_deepseek_m2` | agree | 2 | 0.0414 | [+0.0093, +0.0737] | yes | yes | peer |
| qwen | `B0_graded_both_m1` | graded | 2 | 0.0568 | [+0.0313, +0.0872] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m2` | graded | 2 | 0.0364 | [+0.0033, +0.0668] | yes | yes | peer |
| qwen | `B0_graded_deepseek_m2` | graded | 2 | 0.0602 | [+0.0300, +0.0871] | yes | yes | peer |
| qwen | `B0_agree_both_m2` | agree | 4 | 0.0493 | [+0.0224, +0.0855] | yes | yes | peer |
| qwen | `B0_agree_deepseek_llama_m4` | agree | 4 | 0.0468 | [+0.0148, +0.0796] | yes | yes | peer |
| qwen | `B0_agree_deepseek_m4` | agree | 4 | 0.0452 | [+0.0125, +0.0781] | yes | yes | peer |
| qwen | `B0_graded_both_m2` | graded | 4 | 0.0597 | [+0.0310, +0.0911] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m4` | graded | 4 | 0.0381 | [+0.0045, +0.0688] | yes | yes | peer |
| qwen | `B0_graded_deepseek_m4` | graded | 4 | 0.0633 | [+0.0336, +0.0908] | yes | yes | peer |
| qwen | `B0_agree_both_m4` | agree | 8 | 0.0522 | [+0.0241, +0.0891] | yes | yes | peer |
| qwen | `B0_agree_deepseek_llama_m8` | agree | 8 | 0.0505 | [+0.0199, +0.0791] | yes | -- | peer |
| qwen | `B0_agree_deepseek_m8` | agree | 8 | 0.0485 | [+0.0122, +0.0829] | yes | -- | peer |
| qwen | `B0_graded_both_m4` | graded | 8 | 0.0610 | [+0.0331, +0.0914] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m8` | graded | 8 | 0.0394 | [+0.0036, +0.0697] | yes | -- | peer |
| qwen | `B0_graded_deepseek_m8` | graded | 8 | 0.0644 | [+0.0335, +0.0927] | yes | -- | peer |
| qwen | `B0_agree_both_m8` | agree | 16 | 0.0540 | [+0.0241, +0.0921] | yes | -- | peer |
| qwen | `B0_graded_both_m8` | graded | 16 | 0.0621 | [+0.0347, +0.0931] | yes | -- | peer |
| deepseek | `B0` | none | 0 | -0.0240 | [-0.0524, +0.0024] | no | -- | tie |
| deepseek | `B0_agree_deepseek_llama_m1` | agree | 1 | -0.0284 | [-0.0644, -0.0021] | yes | yes | B1 |
| deepseek | `B0_agree_qwen_m1` | agree | 1 | 0.0075 | [-0.0281, +0.0425] | no | no | tie |
| deepseek | `B0_graded_deepseek_llama_m1` | graded | 1 | 0.0534 | [+0.0284, +0.0805] | yes | yes | peer |
| deepseek | `B0_graded_qwen_m1` | graded | 1 | 0.0463 | [+0.0143, +0.0842] | yes | yes | peer |
| deepseek | `B0_agree_both_m1` | agree | 2 | 0.0076 | [-0.0269, +0.0376] | no | no | tie |
| deepseek | `B0_agree_deepseek_llama_m2` | agree | 2 | -0.0287 | [-0.0652, +0.0023] | no | yes | tie |
| deepseek | `B0_agree_qwen_m2` | agree | 2 | 0.0098 | [-0.0262, +0.0459] | no | yes | tie |
| deepseek | `B0_graded_both_m1` | graded | 2 | 0.0751 | [+0.0478, +0.1040] | yes | yes | peer |
| deepseek | `B0_graded_deepseek_llama_m2` | graded | 2 | 0.0631 | [+0.0382, +0.0884] | yes | yes | peer |
| deepseek | `B0_graded_qwen_m2` | graded | 2 | 0.0618 | [+0.0324, +0.0967] | yes | yes | peer |
| deepseek | `B0_agree_both_m2` | agree | 4 | 0.0172 | [-0.0120, +0.0462] | no | yes | tie |
| deepseek | `B0_agree_deepseek_llama_m4` | agree | 4 | -0.0289 | [-0.0662, -0.0009] | yes | yes | B1 |
| deepseek | `B0_agree_qwen_m4` | agree | 4 | 0.0073 | [-0.0315, +0.0401] | no | yes | tie |
| deepseek | `B0_graded_both_m2` | graded | 4 | 0.0822 | [+0.0562, +0.1126] | yes | yes | peer |
| deepseek | `B0_graded_deepseek_llama_m4` | graded | 4 | 0.0672 | [+0.0424, +0.0930] | yes | yes | peer |
| deepseek | `B0_graded_qwen_m4` | graded | 4 | 0.0731 | [+0.0469, +0.1015] | yes | yes | peer |
| deepseek | `B0_agree_both_m4` | agree | 8 | 0.0162 | [-0.0149, +0.0473] | no | yes | tie |
| deepseek | `B0_agree_deepseek_llama_m8` | agree | 8 | -0.0289 | [-0.0657, -0.0013] | yes | -- | B1 |
| deepseek | `B0_agree_qwen_m8` | agree | 8 | 0.0065 | [-0.0306, +0.0453] | no | -- | tie |
| deepseek | `B0_graded_both_m4` | graded | 8 | 0.0850 | [+0.0590, +0.1152] | yes | yes | peer |
| deepseek | `B0_graded_deepseek_llama_m8` | graded | 8 | 0.0683 | [+0.0430, +0.0957] | yes | -- | peer |
| deepseek | `B0_graded_qwen_m8` | graded | 8 | 0.0782 | [+0.0518, +0.1099] | yes | -- | peer |
| deepseek | `B0_agree_both_m8` | agree | 16 | 0.0155 | [-0.0187, +0.0467] | no | -- | tie |
| deepseek | `B0_graded_both_m8` | graded | 16 | 0.0869 | [+0.0622, +0.1176] | yes | -- | peer |
| deepseek_llama | `B0` | none | 0 | -0.0354 | [-0.0716, -0.0074] | yes | -- | B1 |
| deepseek_llama | `B0_agree_deepseek_m1` | agree | 1 | -0.0038 | [-0.0481, +0.0364] | no | no | tie |
| deepseek_llama | `B0_agree_qwen_m1` | agree | 1 | 0.0632 | [+0.0308, +0.1001] | yes | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m1` | graded | 1 | 0.0446 | [+0.0153, +0.0768] | yes | yes | peer |
| deepseek_llama | `B0_graded_qwen_m1` | graded | 1 | 0.0192 | [-0.0109, +0.0563] | no | no | tie |
| deepseek_llama | `B0_agree_both_m1` | agree | 2 | 0.0732 | [+0.0438, +0.1088] | yes | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m2` | agree | 2 | -0.0064 | [-0.0557, +0.0388] | no | no | tie |
| deepseek_llama | `B0_agree_qwen_m2` | agree | 2 | 0.0700 | [+0.0389, +0.1038] | yes | yes | peer |
| deepseek_llama | `B0_graded_both_m1` | graded | 2 | 0.0445 | [+0.0149, +0.0796] | yes | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m2` | graded | 2 | 0.0494 | [+0.0221, +0.0796] | yes | yes | peer |
| deepseek_llama | `B0_graded_qwen_m2` | graded | 2 | 0.0216 | [-0.0126, +0.0608] | no | yes | tie |
| deepseek_llama | `B0_agree_both_m2` | agree | 4 | 0.0743 | [+0.0420, +0.1122] | yes | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m4` | agree | 4 | 0.0044 | [-0.0404, +0.0498] | no | no | tie |
| deepseek_llama | `B0_agree_qwen_m4` | agree | 4 | 0.0712 | [+0.0387, +0.1087] | yes | yes | peer |
| deepseek_llama | `B0_graded_both_m2` | graded | 4 | 0.0468 | [+0.0146, +0.0816] | yes | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m4` | graded | 4 | 0.0506 | [+0.0242, +0.0791] | yes | yes | peer |
| deepseek_llama | `B0_graded_qwen_m4` | graded | 4 | 0.0240 | [-0.0113, +0.0602] | no | yes | tie |
| deepseek_llama | `B0_agree_both_m4` | agree | 8 | 0.0748 | [+0.0383, +0.1138] | yes | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m8` | agree | 8 | 0.0052 | [-0.0415, +0.0444] | no | -- | tie |
| deepseek_llama | `B0_agree_qwen_m8` | agree | 8 | 0.0726 | [+0.0402, +0.1110] | yes | -- | peer |
| deepseek_llama | `B0_graded_both_m4` | graded | 8 | 0.0459 | [+0.0106, +0.0821] | yes | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m8` | graded | 8 | 0.0520 | [+0.0202, +0.0821] | yes | -- | peer |
| deepseek_llama | `B0_graded_qwen_m8` | graded | 8 | 0.0252 | [-0.0098, +0.0675] | no | -- | tie |
| deepseek_llama | `B0_agree_both_m8` | agree | 16 | 0.0741 | [+0.0397, +0.1158] | yes | -- | peer |
| deepseek_llama | `B0_graded_both_m8` | graded | 16 | 0.0493 | [+0.0138, +0.0845] | yes | -- | peer |

## 4. The one-extra-generation question

The cheapest thing on the ladder that is not already paid for is one sample from one peer. If it beats `B1`, a cheap peer wins outright. If it does not, `B1` wins at strictly lower cost. Only the `agree` rows are a fair answer to that question -- the `graded` rows need the gold answer and are reported as the bound they are.

| Model | Rung | Kind | Deployable | B1 - rung | 95% CI | sign stable | winner |
|---|---|---|:--:|---:|---|:--:|---|
| qwen | `B0_agree_deepseek_llama_m1` | agree | yes | 0.0312 | [-0.0031, +0.0610] | yes | tie |
| qwen | `B0_agree_deepseek_m1` | agree | yes | 0.0413 | [+0.0173, +0.0683] | yes | peer |
| qwen | `B0_graded_deepseek_llama_m1` | graded | no | 0.0264 | [-0.0061, +0.0551] | yes | tie |
| qwen | `B0_graded_deepseek_m1` | graded | no | 0.0547 | [+0.0242, +0.0854] | yes | peer |
| deepseek | `B0_agree_deepseek_llama_m1` | agree | yes | -0.0284 | [-0.0644, -0.0021] | yes | B1 |
| deepseek | `B0_agree_qwen_m1` | agree | yes | 0.0075 | [-0.0281, +0.0425] | no | tie |
| deepseek | `B0_graded_deepseek_llama_m1` | graded | no | 0.0534 | [+0.0284, +0.0805] | yes | peer |
| deepseek | `B0_graded_qwen_m1` | graded | no | 0.0463 | [+0.0143, +0.0842] | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m1` | agree | yes | -0.0038 | [-0.0481, +0.0364] | no | tie |
| deepseek_llama | `B0_agree_qwen_m1` | agree | yes | 0.0632 | [+0.0308, +0.1001] | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m1` | graded | no | 0.0446 | [+0.0153, +0.0768] | yes | peer |
| deepseek_llama | `B0_graded_qwen_m1` | graded | no | 0.0192 | [-0.0109, +0.0563] | no | tie |

## 5. Saturated comparisons

Rungs that have removed at least 90% of `B0`'s headroom. On these the delta against `B1` is not evidence about which readout is better; there is almost nothing left for either to remove.

* **qwen** -- headroom 0.1463 above a floor of 0.0836; no rung saturated
* **deepseek** -- headroom 0.1177 above a floor of 0.0342; 2 rung(s) saturated: `B0_graded_both_m4`, `B0_graded_both_m8`
* **deepseek_llama** -- headroom 0.1700 above a floor of 0.0771; no rung saturated

## What the cost model does not charge for

`B1` needs the target's hidden states retained and a Mahalanobis readout fitted over them. That is real work and real memory; it is not a generation, and it does not scale with the number of models you are willing to run. Charging it as tokens would be the mirror image of the error this rung exists to correct, so it is named here and left uncosted rather than silently folded in.

The `graded` rungs are not charged for the gold answer they consume, because it cannot be bought at decision time at any price. They bound the peer family from above and are reported for that reason only.
