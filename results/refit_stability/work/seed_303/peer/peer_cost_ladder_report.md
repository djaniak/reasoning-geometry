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
| qwen | 500 | 0.620 | 0.0836 | 0.2357 | 0.1810 | 0.1521 |
| deepseek | 500 | 0.750 | 0.0342 | 0.1743 | 0.1336 | 0.1401 |
| deepseek_llama | 500 | 0.634 | 0.0771 | 0.2512 | 0.2028 | 0.1742 |

## 2. The ladder in cost order

`Extra` is the margin over `B0` per prompt. `AURC` is the mean over 25 independent re-draws of which siblings were bought (sub-sampled rungs only; the full-cache rungs are deterministic). `removed` is the fraction of `B0`'s headroom the rung takes out -- at 1.00 the rung is on the oracle floor.

| Model | Rung | Kind | Extra calls | Extra tokens | AURC | across-draw range | removed |
|---|---|---|---:|---:|---:|---|---:|
| qwen | `B0` | none | 0 | 0 | 0.2357 | -- | 0.00 |
| qwen | `B1` | none | 0 | 0 | 0.1810 | -- | 0.36 |
| qwen | `B0_agree_deepseek_llama_m1` | agree | 1 | 3049 | 0.1476 | [0.1334, 0.1568] | 0.58 |
| qwen | `B0_agree_deepseek_m1` | agree | 1 | 3101 | 0.1370 | [0.1231, 0.1545] | 0.65 |
| qwen | `B0_graded_deepseek_llama_m1` | graded | 1 | 3049 | 0.1533 | [0.1446, 0.1694] | 0.54 |
| qwen | `B0_graded_deepseek_m1` | graded | 1 | 3101 | 0.1219 | [0.1144, 0.1311] | 0.75 |
| qwen | `B0_agree_both_m1` | agree | 2 | 6150 | 0.1277 | [0.1184, 0.1396] | 0.71 |
| qwen | `B0_agree_deepseek_llama_m2` | agree | 2 | 6099 | 0.1361 | [0.1289, 0.1459] | 0.65 |
| qwen | `B0_agree_deepseek_m2` | agree | 2 | 6202 | 0.1323 | [0.1191, 0.1512] | 0.68 |
| qwen | `B0_graded_both_m1` | graded | 2 | 6150 | 0.1219 | [0.1157, 0.1290] | 0.75 |
| qwen | `B0_graded_deepseek_llama_m2` | graded | 2 | 6099 | 0.1439 | [0.1400, 0.1503] | 0.60 |
| qwen | `B0_graded_deepseek_m2` | graded | 2 | 6202 | 0.1179 | [0.1142, 0.1240] | 0.77 |
| qwen | `B0_agree_both_m2` | agree | 4 | 12301 | 0.1223 | [0.1109, 0.1394] | 0.75 |
| qwen | `B0_agree_deepseek_llama_m4` | agree | 4 | 12197 | 0.1299 | [0.1238, 0.1385] | 0.70 |
| qwen | `B0_agree_deepseek_m4` | agree | 4 | 12404 | 0.1281 | [0.1191, 0.1491] | 0.71 |
| qwen | `B0_graded_both_m2` | graded | 4 | 12301 | 0.1190 | [0.1165, 0.1239] | 0.77 |
| qwen | `B0_graded_deepseek_llama_m4` | graded | 4 | 12197 | 0.1404 | [0.1365, 0.1432] | 0.63 |
| qwen | `B0_graded_deepseek_m4` | graded | 4 | 12404 | 0.1149 | [0.1134, 0.1179] | 0.79 |
| qwen | `B0_agree_both_m4` | agree | 8 | 24601 | 0.1163 | [0.1109, 0.1221] | 0.79 |
| qwen | `B0_agree_deepseek_llama_m8` | agree | 8 | 24394 | 0.1264 | -- | 0.72 |
| qwen | `B0_agree_deepseek_m8` | agree | 8 | 24808 | 0.1260 | -- | 0.72 |
| qwen | `B0_graded_both_m4` | graded | 8 | 24601 | 0.1173 | [0.1155, 0.1190] | 0.78 |
| qwen | `B0_graded_deepseek_llama_m8` | graded | 8 | 24394 | 0.1390 | -- | 0.64 |
| qwen | `B0_graded_deepseek_m8` | graded | 8 | 24808 | 0.1138 | -- | 0.80 |
| qwen | `B0_agree_both_m8` | agree | 16 | 49202 | 0.1151 | -- | 0.79 |
| qwen | `B0_graded_both_m8` | graded | 16 | 49202 | 0.1163 | -- | 0.79 |
| deepseek | `B0` | none | 0 | 0 | 0.1743 | -- | 0.00 |
| deepseek | `B1` | none | 0 | 0 | 0.1336 | -- | 0.29 |
| deepseek | `B0_agree_deepseek_llama_m1` | agree | 1 | 3049 | 0.1765 | [0.1617, 0.1882] | -0.02 |
| deepseek | `B0_agree_qwen_m1` | agree | 1 | 556 | 0.1310 | [0.1202, 0.1425] | 0.31 |
| deepseek | `B0_graded_deepseek_llama_m1` | graded | 1 | 3049 | 0.0767 | [0.0664, 0.0904] | 0.70 |
| deepseek | `B0_graded_qwen_m1` | graded | 1 | 556 | 0.0772 | [0.0647, 0.0892] | 0.69 |
| deepseek | `B0_agree_both_m1` | agree | 2 | 3606 | 0.1229 | [0.1077, 0.1388] | 0.37 |
| deepseek | `B0_agree_deepseek_llama_m2` | agree | 2 | 6099 | 0.1773 | [0.1740, 0.1823] | -0.02 |
| deepseek | `B0_agree_qwen_m2` | agree | 2 | 1112 | 0.1252 | [0.1187, 0.1299] | 0.35 |
| deepseek | `B0_graded_both_m1` | graded | 2 | 3606 | 0.0533 | [0.0469, 0.0651] | 0.86 |
| deepseek | `B0_graded_deepseek_llama_m2` | graded | 2 | 6099 | 0.0647 | [0.0611, 0.0705] | 0.78 |
| deepseek | `B0_graded_qwen_m2` | graded | 2 | 1112 | 0.0643 | [0.0557, 0.0773] | 0.79 |
| deepseek | `B0_agree_both_m2` | agree | 4 | 7211 | 0.1168 | [0.1033, 0.1271] | 0.41 |
| deepseek | `B0_agree_deepseek_llama_m4` | agree | 4 | 12197 | 0.1763 | [0.1703, 0.1823] | -0.01 |
| deepseek | `B0_agree_qwen_m4` | agree | 4 | 2225 | 0.1271 | [0.1243, 0.1322] | 0.34 |
| deepseek | `B0_graded_both_m2` | graded | 4 | 7211 | 0.0463 | [0.0420, 0.0498] | 0.91 |
| deepseek | `B0_graded_deepseek_llama_m4` | graded | 4 | 12197 | 0.0605 | [0.0580, 0.0634] | 0.81 |
| deepseek | `B0_graded_qwen_m4` | graded | 4 | 2225 | 0.0559 | [0.0516, 0.0640] | 0.85 |
| deepseek | `B0_agree_both_m4` | agree | 8 | 14422 | 0.1191 | [0.1098, 0.1298] | 0.39 |
| deepseek | `B0_agree_deepseek_llama_m8` | agree | 8 | 24394 | 0.1757 | -- | -0.01 |
| deepseek | `B0_agree_qwen_m8` | agree | 8 | 4450 | 0.1268 | -- | 0.34 |
| deepseek | `B0_graded_both_m4` | graded | 8 | 14422 | 0.0428 | [0.0398, 0.0478] | 0.94 |
| deepseek | `B0_graded_deepseek_llama_m8` | graded | 8 | 24394 | 0.0586 | -- | 0.83 |
| deepseek | `B0_graded_qwen_m8` | graded | 8 | 4450 | 0.0512 | -- | 0.88 |
| deepseek | `B0_agree_both_m8` | agree | 16 | 28844 | 0.1198 | -- | 0.39 |
| deepseek | `B0_graded_both_m8` | graded | 16 | 28844 | 0.0406 | -- | 0.95 |
| deepseek_llama | `B0` | none | 0 | 0 | 0.2512 | -- | 0.00 |
| deepseek_llama | `B1` | none | 0 | 0 | 0.2028 | -- | 0.28 |
| deepseek_llama | `B0_agree_deepseek_m1` | agree | 1 | 3101 | 0.2151 | [0.2044, 0.2245] | 0.21 |
| deepseek_llama | `B0_agree_qwen_m1` | agree | 1 | 556 | 0.1449 | [0.1362, 0.1571] | 0.61 |
| deepseek_llama | `B0_graded_deepseek_m1` | graded | 1 | 3101 | 0.1749 | [0.1635, 0.1982] | 0.44 |
| deepseek_llama | `B0_graded_qwen_m1` | graded | 1 | 556 | 0.2021 | [0.1928, 0.2128] | 0.28 |
| deepseek_llama | `B0_agree_both_m1` | agree | 2 | 3657 | 0.1375 | [0.1256, 0.1510] | 0.65 |
| deepseek_llama | `B0_agree_deepseek_m2` | agree | 2 | 6202 | 0.2190 | [0.2086, 0.2304] | 0.19 |
| deepseek_llama | `B0_agree_qwen_m2` | agree | 2 | 1112 | 0.1396 | [0.1307, 0.1470] | 0.64 |
| deepseek_llama | `B0_graded_both_m1` | graded | 2 | 3657 | 0.1776 | [0.1685, 0.1902] | 0.42 |
| deepseek_llama | `B0_graded_deepseek_m2` | graded | 2 | 6202 | 0.1681 | [0.1541, 0.1841] | 0.48 |
| deepseek_llama | `B0_graded_qwen_m2` | graded | 2 | 1112 | 0.1973 | [0.1882, 0.2057] | 0.31 |
| deepseek_llama | `B0_agree_both_m2` | agree | 4 | 7314 | 0.1360 | [0.1256, 0.1450] | 0.66 |
| deepseek_llama | `B0_agree_deepseek_m4` | agree | 4 | 12404 | 0.2204 | [0.2147, 0.2265] | 0.18 |
| deepseek_llama | `B0_agree_qwen_m4` | agree | 4 | 2225 | 0.1383 | [0.1331, 0.1437] | 0.65 |
| deepseek_llama | `B0_graded_both_m2` | graded | 4 | 7314 | 0.1723 | [0.1598, 0.1884] | 0.45 |
| deepseek_llama | `B0_graded_deepseek_m4` | graded | 4 | 12404 | 0.1679 | [0.1585, 0.1793] | 0.48 |
| deepseek_llama | `B0_graded_qwen_m4` | graded | 4 | 2225 | 0.1910 | [0.1838, 0.2019] | 0.35 |
| deepseek_llama | `B0_agree_both_m4` | agree | 8 | 14629 | 0.1385 | [0.1303, 0.1450] | 0.65 |
| deepseek_llama | `B0_agree_deepseek_m8` | agree | 8 | 24808 | 0.2200 | -- | 0.18 |
| deepseek_llama | `B0_agree_qwen_m8` | agree | 8 | 4450 | 0.1354 | -- | 0.67 |
| deepseek_llama | `B0_graded_both_m4` | graded | 8 | 14629 | 0.1709 | [0.1636, 0.1795] | 0.46 |
| deepseek_llama | `B0_graded_deepseek_m8` | graded | 8 | 24808 | 0.1642 | -- | 0.50 |
| deepseek_llama | `B0_graded_qwen_m8` | graded | 8 | 4450 | 0.1894 | -- | 0.35 |
| deepseek_llama | `B0_agree_both_m8` | agree | 16 | 29258 | 0.1362 | -- | 0.66 |
| deepseek_llama | `B0_graded_both_m8` | graded | 16 | 29258 | 0.1690 | -- | 0.47 |

## 3. Does a bought rung beat the free one?

`B1 - rung` on AURC. Negative favours `B1`, the rung that costs no extra generations. The interval is the frozen prompt bootstrap on the median draw; `sign stable` reports whether every draw agreed on the direction, which is the separate question of whether the verdict depends on which siblings you happened to buy.

| Model | Rung | Kind | Extra calls | B1 - rung | 95% CI | excludes 0 | sign stable | winner |
|---|---|---|---:|---:|---|:--:|:--:|---|
| qwen | `B0` | none | 0 | -0.0546 | [-0.0964, -0.0139] | yes | -- | B1 |
| qwen | `B0_agree_deepseek_llama_m1` | agree | 1 | 0.0332 | [-0.0003, +0.0695] | no | yes | tie |
| qwen | `B0_agree_deepseek_m1` | agree | 1 | 0.0443 | [+0.0143, +0.0795] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m1` | graded | 1 | 0.0275 | [-0.0021, +0.0641] | no | yes | tie |
| qwen | `B0_graded_deepseek_m1` | graded | 1 | 0.0593 | [+0.0368, +0.0953] | yes | yes | peer |
| qwen | `B0_agree_both_m1` | agree | 2 | 0.0538 | [+0.0236, +0.0946] | yes | yes | peer |
| qwen | `B0_agree_deepseek_llama_m2` | agree | 2 | 0.0450 | [+0.0160, +0.0765] | yes | yes | peer |
| qwen | `B0_agree_deepseek_m2` | agree | 2 | 0.0488 | [+0.0201, +0.0861] | yes | yes | peer |
| qwen | `B0_graded_both_m1` | graded | 2 | 0.0594 | [+0.0254, +0.0965] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m2` | graded | 2 | 0.0379 | [+0.0050, +0.0712] | yes | yes | peer |
| qwen | `B0_graded_deepseek_m2` | graded | 2 | 0.0639 | [+0.0411, +0.0980] | yes | yes | peer |
| qwen | `B0_agree_both_m2` | agree | 4 | 0.0585 | [+0.0293, +0.0959] | yes | yes | peer |
| qwen | `B0_agree_deepseek_llama_m4` | agree | 4 | 0.0521 | [+0.0243, +0.0844] | yes | yes | peer |
| qwen | `B0_agree_deepseek_m4` | agree | 4 | 0.0557 | [+0.0224, +0.0969] | yes | yes | peer |
| qwen | `B0_graded_both_m2` | graded | 4 | 0.0626 | [+0.0300, +0.0967] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m4` | graded | 4 | 0.0406 | [+0.0097, +0.0736] | yes | yes | peer |
| qwen | `B0_graded_deepseek_m4` | graded | 4 | 0.0663 | [+0.0432, +0.1011] | yes | yes | peer |
| qwen | `B0_agree_both_m4` | agree | 8 | 0.0645 | [+0.0373, +0.1034] | yes | yes | peer |
| qwen | `B0_agree_deepseek_llama_m8` | agree | 8 | 0.0547 | [+0.0235, +0.0887] | yes | -- | peer |
| qwen | `B0_agree_deepseek_m8` | agree | 8 | 0.0550 | [+0.0217, +0.0965] | yes | -- | peer |
| qwen | `B0_graded_both_m4` | graded | 8 | 0.0639 | [+0.0312, +0.0994] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m8` | graded | 8 | 0.0420 | [+0.0097, +0.0751] | yes | -- | peer |
| qwen | `B0_graded_deepseek_m8` | graded | 8 | 0.0673 | [+0.0452, +0.1016] | yes | -- | peer |
| qwen | `B0_agree_both_m8` | agree | 16 | 0.0660 | [+0.0335, +0.1046] | yes | -- | peer |
| qwen | `B0_graded_both_m8` | graded | 16 | 0.0648 | [+0.0319, +0.0999] | yes | -- | peer |
| deepseek | `B0` | none | 0 | -0.0407 | [-0.0767, -0.0148] | yes | -- | B1 |
| deepseek | `B0_agree_deepseek_llama_m1` | agree | 1 | -0.0432 | [-0.0764, -0.0161] | yes | yes | B1 |
| deepseek | `B0_agree_qwen_m1` | agree | 1 | 0.0039 | [-0.0284, +0.0402] | no | no | tie |
| deepseek | `B0_graded_deepseek_llama_m1` | graded | 1 | 0.0562 | [+0.0312, +0.0912] | yes | yes | peer |
| deepseek | `B0_graded_qwen_m1` | graded | 1 | 0.0563 | [+0.0159, +0.0946] | yes | yes | peer |
| deepseek | `B0_agree_both_m1` | agree | 2 | 0.0093 | [-0.0206, +0.0394] | no | no | tie |
| deepseek | `B0_agree_deepseek_llama_m2` | agree | 2 | -0.0434 | [-0.0732, -0.0179] | yes | yes | B1 |
| deepseek | `B0_agree_qwen_m2` | agree | 2 | 0.0083 | [-0.0328, +0.0472] | no | yes | tie |
| deepseek | `B0_graded_both_m1` | graded | 2 | 0.0812 | [+0.0515, +0.1118] | yes | yes | peer |
| deepseek | `B0_graded_deepseek_llama_m2` | graded | 2 | 0.0689 | [+0.0407, +0.0981] | yes | yes | peer |
| deepseek | `B0_graded_qwen_m2` | graded | 2 | 0.0702 | [+0.0348, +0.1037] | yes | yes | peer |
| deepseek | `B0_agree_both_m2` | agree | 4 | 0.0161 | [-0.0204, +0.0527] | no | yes | tie |
| deepseek | `B0_agree_deepseek_llama_m4` | agree | 4 | -0.0431 | [-0.0750, -0.0164] | yes | yes | B1 |
| deepseek | `B0_agree_qwen_m4` | agree | 4 | 0.0067 | [-0.0323, +0.0453] | no | yes | tie |
| deepseek | `B0_graded_both_m2` | graded | 4 | 0.0875 | [+0.0580, +0.1169] | yes | yes | peer |
| deepseek | `B0_graded_deepseek_llama_m4` | graded | 4 | 0.0731 | [+0.0439, +0.1037] | yes | yes | peer |
| deepseek | `B0_graded_qwen_m4` | graded | 4 | 0.0780 | [+0.0486, +0.1091] | yes | yes | peer |
| deepseek | `B0_agree_both_m4` | agree | 8 | 0.0138 | [-0.0249, +0.0517] | no | yes | tie |
| deepseek | `B0_agree_deepseek_llama_m8` | agree | 8 | -0.0421 | [-0.0721, -0.0174] | yes | -- | B1 |
| deepseek | `B0_agree_qwen_m8` | agree | 8 | 0.0068 | [-0.0318, +0.0465] | no | -- | tie |
| deepseek | `B0_graded_both_m4` | graded | 8 | 0.0910 | [+0.0610, +0.1202] | yes | yes | peer |
| deepseek | `B0_graded_deepseek_llama_m8` | graded | 8 | 0.0750 | [+0.0453, +0.1051] | yes | -- | peer |
| deepseek | `B0_graded_qwen_m8` | graded | 8 | 0.0824 | [+0.0530, +0.1121] | yes | -- | peer |
| deepseek | `B0_agree_both_m8` | agree | 16 | 0.0137 | [-0.0240, +0.0514] | no | -- | tie |
| deepseek | `B0_graded_both_m8` | graded | 16 | 0.0930 | [+0.0628, +0.1231] | yes | -- | peer |
| deepseek_llama | `B0` | none | 0 | -0.0485 | [-0.0768, -0.0214] | yes | -- | B1 |
| deepseek_llama | `B0_agree_deepseek_m1` | agree | 1 | -0.0132 | [-0.0496, +0.0194] | no | yes | tie |
| deepseek_llama | `B0_agree_qwen_m1` | agree | 1 | 0.0583 | [+0.0271, +0.0927] | yes | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m1` | graded | 1 | 0.0282 | [-0.0158, +0.0623] | no | yes | tie |
| deepseek_llama | `B0_graded_qwen_m1` | graded | 1 | -0.0002 | [-0.0371, +0.0372] | no | no | tie |
| deepseek_llama | `B0_agree_both_m1` | agree | 2 | 0.0657 | [+0.0335, +0.1003] | yes | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m2` | agree | 2 | -0.0166 | [-0.0526, +0.0173] | no | yes | tie |
| deepseek_llama | `B0_agree_qwen_m2` | agree | 2 | 0.0631 | [+0.0320, +0.0960] | yes | yes | peer |
| deepseek_llama | `B0_graded_both_m1` | graded | 2 | 0.0262 | [-0.0103, +0.0563] | no | yes | tie |
| deepseek_llama | `B0_graded_deepseek_m2` | graded | 2 | 0.0343 | [-0.0014, +0.0711] | no | yes | tie |
| deepseek_llama | `B0_graded_qwen_m2` | graded | 2 | 0.0056 | [-0.0348, +0.0412] | no | no | tie |
| deepseek_llama | `B0_agree_both_m2` | agree | 4 | 0.0661 | [+0.0366, +0.0974] | yes | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m4` | agree | 4 | -0.0174 | [-0.0525, +0.0211] | no | yes | tie |
| deepseek_llama | `B0_agree_qwen_m4` | agree | 4 | 0.0646 | [+0.0374, +0.0969] | yes | yes | peer |
| deepseek_llama | `B0_graded_both_m2` | graded | 4 | 0.0315 | [-0.0037, +0.0628] | no | yes | tie |
| deepseek_llama | `B0_graded_deepseek_m4` | graded | 4 | 0.0359 | [+0.0029, +0.0683] | yes | yes | peer |
| deepseek_llama | `B0_graded_qwen_m4` | graded | 4 | 0.0120 | [-0.0246, +0.0504] | no | yes | tie |
| deepseek_llama | `B0_agree_both_m4` | agree | 8 | 0.0632 | [+0.0300, +0.0979] | yes | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m8` | agree | 8 | -0.0172 | [-0.0544, +0.0195] | no | -- | tie |
| deepseek_llama | `B0_agree_qwen_m8` | agree | 8 | 0.0674 | [+0.0386, +0.0996] | yes | -- | peer |
| deepseek_llama | `B0_graded_both_m4` | graded | 8 | 0.0322 | [-0.0063, +0.0626] | no | yes | tie |
| deepseek_llama | `B0_graded_deepseek_m8` | graded | 8 | 0.0386 | [+0.0086, +0.0702] | yes | -- | peer |
| deepseek_llama | `B0_graded_qwen_m8` | graded | 8 | 0.0133 | [-0.0239, +0.0503] | no | -- | tie |
| deepseek_llama | `B0_agree_both_m8` | agree | 16 | 0.0666 | [+0.0362, +0.1015] | yes | -- | peer |
| deepseek_llama | `B0_graded_both_m8` | graded | 16 | 0.0338 | [-0.0020, +0.0651] | no | -- | tie |

## 4. The one-extra-generation question

The cheapest thing on the ladder that is not already paid for is one sample from one peer. If it beats `B1`, a cheap peer wins outright. If it does not, `B1` wins at strictly lower cost. Only the `agree` rows are a fair answer to that question -- the `graded` rows need the gold answer and are reported as the bound they are.

| Model | Rung | Kind | Deployable | B1 - rung | 95% CI | sign stable | winner |
|---|---|---|:--:|---:|---|:--:|---|
| qwen | `B0_agree_deepseek_llama_m1` | agree | yes | 0.0332 | [-0.0003, +0.0695] | yes | tie |
| qwen | `B0_agree_deepseek_m1` | agree | yes | 0.0443 | [+0.0143, +0.0795] | yes | peer |
| qwen | `B0_graded_deepseek_llama_m1` | graded | no | 0.0275 | [-0.0021, +0.0641] | yes | tie |
| qwen | `B0_graded_deepseek_m1` | graded | no | 0.0593 | [+0.0368, +0.0953] | yes | peer |
| deepseek | `B0_agree_deepseek_llama_m1` | agree | yes | -0.0432 | [-0.0764, -0.0161] | yes | B1 |
| deepseek | `B0_agree_qwen_m1` | agree | yes | 0.0039 | [-0.0284, +0.0402] | no | tie |
| deepseek | `B0_graded_deepseek_llama_m1` | graded | no | 0.0562 | [+0.0312, +0.0912] | yes | peer |
| deepseek | `B0_graded_qwen_m1` | graded | no | 0.0563 | [+0.0159, +0.0946] | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m1` | agree | yes | -0.0132 | [-0.0496, +0.0194] | yes | tie |
| deepseek_llama | `B0_agree_qwen_m1` | agree | yes | 0.0583 | [+0.0271, +0.0927] | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m1` | graded | no | 0.0282 | [-0.0158, +0.0623] | yes | tie |
| deepseek_llama | `B0_graded_qwen_m1` | graded | no | -0.0002 | [-0.0371, +0.0372] | no | tie |

## 5. Saturated comparisons

Rungs that have removed at least 90% of `B0`'s headroom. On these the delta against `B1` is not evidence about which readout is better; there is almost nothing left for either to remove.

* **qwen** -- headroom 0.1521 above a floor of 0.0836; no rung saturated
* **deepseek** -- headroom 0.1401 above a floor of 0.0342; 3 rung(s) saturated: `B0_graded_both_m2`, `B0_graded_both_m4`, `B0_graded_both_m8`
* **deepseek_llama** -- headroom 0.1742 above a floor of 0.0771; no rung saturated

## What the cost model does not charge for

`B1` needs the target's hidden states retained and a Mahalanobis readout fitted over them. That is real work and real memory; it is not a generation, and it does not scale with the number of models you are willing to run. Charging it as tokens would be the mirror image of the error this rung exists to correct, so it is named here and left uncosted rather than silently folded in.

The `graded` rungs are not charged for the gold answer they consume, because it cannot be bought at decision time at any price. They bound the peer family from above and are reported for that reason only.
