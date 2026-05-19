# Training Time Across Three Seeds (extracted from log)

Source: `/home/yuchen/projects/FedQHD/logs/main_table_3seed_variance_nohup.log`

Per-run wall-clock seconds; mean ± sample std across 3 seeds (42, 43, 44).

| Env | Condition | Method | s42 | s43 | s44 | mean (s) | std (s) |
|---|---|---|---:|---:|---:|---:|---:|
| CartPole | homo | distillation_dqn | 772.03 | 702.70 | 758.52 | 744.42 | 36.75 |
| CartPole | homo | oracle_dqn | 537.47 | 556.03 | 516.80 | 536.77 | 19.62 |
| CartPole | hetero | distillation_dqn_hetero | 693.29 | 526.45 | 635.63 | 618.46 | 84.74 |
| CartPole | hetero | oracle_dqn_hetero | 3605.62 | 4036.49 | 3899.59 | 3847.23 | 220.15 |
| CartPole | hetero | oracle_qhd_hetero | 320.12 | 388.62 | 390.54 | 366.43 | 40.11 |
| Acrobot | homo | distillation_dqn | 1288.91 | 1289.60 | 1302.74 | 1293.75 | 7.79 |
| Acrobot | homo | fedavg_dqn | 1393.14 | 1397.84 | 1320.34 | 1370.44 | 43.45 |
| Acrobot | homo | fedqhd_homo | 332.16 | 337.28 | 333.97 | 334.47 | 2.60 |
| Acrobot | homo | independent_qhd | 583.52 | 576.74 | 574.73 | 578.33 | 4.61 |
| Acrobot | homo | oracle_dqn | 851.31 | 839.19 | 1038.46 | 909.65 | 111.71 |
| Acrobot | homo | oracle_qhd | 293.67 | 295.16 | 295.46 | 294.76 | 0.96 |
| Acrobot | homo | truncate_fedavg_qhd | 331.79 | 333.67 | 334.13 | 333.20 | 1.24 |
| Acrobot | hetero | distillation_dqn_hetero | 1253.93 | 1024.22 | 1276.25 | 1184.80 | 139.51 |
| Acrobot | hetero | fedqhd_hetero | 111.67 | 111.65 | 110.74 | 111.35 | 0.53 |
| Acrobot | hetero | independent_qhd_hetero | 298.16 | 298.94 | 299.61 | 298.90 | 0.73 |
| Acrobot | hetero | oracle_dqn_hetero | 4104.89 | 4823.31 | 4980.16 | 4636.12 | 466.70 |
| Acrobot | hetero | oracle_qhd_hetero | 371.75 | 378.24 | 376.80 | 375.60 | 3.41 |
| Acrobot | hetero | truncate_fedavg_qhd_hetero | 142.67 | 113.16 | 117.50 | 124.44 | 15.93 |
| LunarLander | homo | distillation_dqn | 2247.11 | 1777.04 | 1611.08 | 1878.41 | 329.91 |
| LunarLander | homo | fedavg_dqn | 2205.82 | 2174.81 | 2250.24 | 2210.29 | 37.91 |
| LunarLander | homo | fedqhd_homo | 928.51 | 941.29 | 920.66 | 930.15 | 10.41 |
| LunarLander | homo | independent_qhd | 169.95 | 169.34 | 168.74 | 169.34 | 0.61 |
| LunarLander | homo | oracle_dqn | 1492.47 | 1424.90 | 1565.64 | 1494.34 | 70.39 |
| LunarLander | homo | oracle_qhd | 927.25 | 899.99 | 930.07 | 919.10 | 16.61 |
| LunarLander | homo | truncate_fedavg_qhd | 940.55 | 933.24 | 922.46 | 932.08 | 9.10 |
| LunarLander | hetero | independent_qhd_hetero | 49.48 | 49.64 | 49.53 | 49.55 | 0.08 |
| LunarLander | hetero | truncate_fedavg_qhd_hetero | 168.74 | 172.36 | 180.34 | 173.81 | 5.93 |

**Incomplete runs (start logged, no completion):**
- LunarLander hetero distillation_dqn_hetero seed=42 (line 56157)
- LunarLander hetero fedqhd_hetero seed=42 (line 56803)
