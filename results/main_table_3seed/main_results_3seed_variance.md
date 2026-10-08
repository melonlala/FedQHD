# Main Results With 3-Seed Variance

Final average reward over the last 100 episodes, reported as mean ± sample std across seeds. The JSON summary also includes sample variance.

| Method | CartPole Homo | CartPole Hetero | Acrobot Homo | Acrobot Hetero | LunarLander Homo | LunarLander Hetero | MountainCar Homo | MountainCar Hetero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Indep. QHD | 26.07 ± 1.08 | 23.97 ± 1.11 | -306.08 ± 17.10 | -495.36 ± 0.36 | -187.32 ± 6.41 | -174.59 ± 5.60 | -199.59 ± 0.43 | -200.00 ± 0.00 |
| Trunc. FedAvg-QHD | 230.82 ± 85.69 | 112.64 ± 26.07 | -106.49 ± 1.83 | -280.77 ± 52.79 | 65.00 ± 5.05 | -24.75 ± 9.70 | -139.90 ± 0.93 | -157.11 ± 0.98 |
| FedAvg-DQN | 120.83 ± 8.97 | -- | -85.27 ± 2.45 | -- | 200.90 ± 13.96 | -- | -146.75 ± 5.94 | -- |
| Distill. FedDQN | 169.91 ± 8.75 | 149.39 ± 12.75 | -87.54 ± 0.55 | -89.22 ± 2.55 | 122.45 ± 53.17 | 140.72 ± 21.71 | -152.33 ± 6.65 | -158.19 ± 4.26 |
| FedQHD | 466.30 ± 22.30 | 351.08 ± 61.14 | -105.02 ± 3.34 | -102.53 ± 0.69 | 224.12 ± 11.79 | 67.61 ± 88.34 | -196.50 ± 2.36 | -161.21 ± 3.40 |
| Oracle QHD† | 373.23 ± 80.86 | 458.54 ± 52.12 | -93.61 ± 0.62 | -101.18 ± 1.51 | 87.62 ± 6.18 | 216.54 ± 28.96 | -139.52 ± 2.80 | -143.64 ± 9.67 |
| Oracle DQN† | 139.07 ± 10.48 | 204.35 ± 16.91 | -75.30 ± 0.35 | -79.83 ± 2.04 | 174.80 ± 31.92 | 184.76 ± 3.41 | -122.01 ± 4.80 | -125.31 ± 5.04 |
