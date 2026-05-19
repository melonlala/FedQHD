# FedQHD — Federated Q-Learning with Hyperdimensional Approximation

Reference implementation accompanying the FedQHD paper. The repository implements
federated reinforcement learning using Random-Fourier-Feature (RFF) hyperdimensional
encoders and anchor-based aggregation, together with the DQN baselines and ablation
studies used in the paper.

## Repository layout

```
FedQHD/
├── agent/                       # RL agents
│   ├── qhd_agent.py             # QHD agent (RFF encoder + linear value head)
│   ├── dqn_agent.py             # DQN baseline
│   ├── fedavg.py                # FedAvg coordinator
│   └── ... (sarsa, q_learning, linear_sarsa variants)
├── env/                         # Environment wrappers
│   ├── utils.py                 # CartPole, Acrobot, LunarLander, MountainCar,
│   │                            # GridWorld, CliffWalking, Taxi, Pong, Freeway
│   ├── bounds.py                # Per-env state bounds for RFF normalisation
│   └── gridworld.py
├── scripts/                     # Top-level runnable entry points
│   ├── experiments_runner.py    # All training functions (Q1/Q2 methods)
│   ├── run_single_method.py     # CLI: run one method, write result+params+summary
│   ├── ablation_experiments.py  # A1/A2/A3 ablations
│   ├── scalability_experiments.py  # Vary N ∈ {5,10,20,50}
│   ├── train.py                 # Generic training loops used by main.py
│   └── run_{CartPole,Acrobot,LunarLander,MountainCar}.sh
├── utils/                       # Helper utilities
│   ├── hparam_search.py         # Grid search over (lr, agg, decay, rff_gamma)
│   ├── rerun_main_table_with_variance.py  # Multi-seed main table generator
│   ├── plot_scalability_bar.py
│   └── utils.py                 # Plotting helpers
├── main.py                      # Legacy single-method entry point
└── results/                     # JSON results, figures, summaries
```

## Installation

```bash
pip install numpy matplotlib gymnasium tqdm torch

# Optional, for Atari Pong / Freeway:
pip install ale-py "gymnasium[atari]" "gymnasium[accept-rom-license]"
```

## Methods implemented

Training functions live in `scripts/experiments_runner.py`.

| Method key (CLI) | Function | Notes |
|---|---|---|
| `independent_qhd` | `train_independent_qhd` | No federation, lower bound. |
| `fedqhd_homo` | `train_fedqhd_homogeneous` | Algorithm 1 — shared RFF encoder, direct parameter averaging. |
| `oracle_qhd` | `train_oracle_qhd(use_heterogeneous=False)` | Per-episode federation over a shared encoder, upper bound. |
| `fedavg_dqn` | `train_fedavg_dqn` | FedAvg over a 2-layer MLP DQN. |
| `oracle_dqn` | `train_oracle_dqn` | DQN with per-episode federation. |
| `distillation_dqn` | `train_distillation_dqn(heterogeneous=False)` | Policy-distillation baseline. |
| `truncate_fedavg_qhd` | `train_truncate_fedavg_qhd(heterogeneous=False)` | Naive pad/truncate averaging. |
| `independent_qhd_hetero` | `train_independent_qhd(heterogeneous=True)` | Heterogeneous encoder dims/bandwidths, no federation. |
| `fedqhd_hetero` | `train_fedqhd_heterogeneous` | Algorithm 2 — anchor-based ridge aggregation. |
| `oracle_qhd_hetero` | `train_oracle_qhd(use_heterogeneous=True)` | N per-client encoders, data pooling. |
| `oracle_dqn_hetero` | `train_oracle_dqn(use_heterogeneous=True)` | DQN oracle counterpart. |
| `truncate_fedavg_qhd_hetero` | `train_truncate_fedavg_qhd(heterogeneous=True)` | Naive hetero baseline. |
| `distillation_dqn_hetero` | `train_distillation_dqn(heterogeneous=True)` | Hetero distillation baseline. |

## Quick start — single method

`scripts/run_single_method.py` runs one method, writes
`results/<env>/q{1,2}_*/<Method_Name>.json`, merges the entry into
`summary.json`, and saves a `params_<method>.json` for traceability.

```bash
cd scripts

# Q1 (homogeneous encoders)
python run_single_method.py --method fedqhd_homo --env CartPole --question q1 \
    --episodes 600 --agent_num 5 --qhd_lr 0.1 --qhd_agg_interval 50

# Q2 (heterogeneous encoders, anchor aggregation)
python run_single_method.py --method fedqhd_hetero --env LunarLander --question q2 \
    --episodes 600 --agent_num 5 --qhd_lr 0.5 --qhd_agg_interval 25 \
    --rff_gamma 0.5 --qhd_exploration_decay 0.900

# DQN baseline
python run_single_method.py --method fedavg_dqn --env Acrobot --question q1 \
    --dqn_lr 0.001 --dqn_agg_interval 25
```

Selected CLI flags (see `--help` for the full list):

| Flag | Default | Purpose |
|---|---|---|
| `--method` | `fedqhd_homo` | Method key from the table above. |
| `--env` | `CartPole` | One of `CartPole / Acrobot / LunarLander / MountainCar / GridWorld / CliffWalking / Taxi / Pong / Freeway`. |
| `--question` | `q2` | `q1` (homogeneous) or `q2` (heterogeneous) — only controls output subdirectory. |
| `--episodes` | 600 | Training episodes. |
| `--agent_num` | 5 | Number of federated clients. |
| `--runs` | 1 | Independent runs to average. |
| `--qhd_lr / --qhd_agg_interval / --qhd_discount / --qhd_exploration_{rate,decay,min}` | — | QHD hyperparameters. |
| `--dqn_lr / --dqn_agg_interval / --dqn_discount / --dqn_exploration_{rate,decay,min}` | — | DQN hyperparameters. |
| `--hyperdimension` | 10000 | RFF dimension `D` for homogeneous QHD. |
| `--rff_gamma` | 1.0 | RFF bandwidth. |
| `--anchor_set_size` | 200 | `m` for heterogeneous anchor aggregation. |
| `--hetero_dims` | `500,1000,2000,5000` | Per-client encoder dims for Q2 methods. |

## Reproducing the paper experiments

Per-environment shell scripts wrap the parameter choices used in the paper. They
launch the relevant methods in parallel and log to `logs/`.

```bash
cd scripts
bash run_CartPole.sh
bash run_Acrobot.sh
bash run_LunarLander.sh
bash run_MountainCar.sh
```

To regenerate the main-results table with multiple seeds (and standard deviations):

```bash
python utils/rerun_main_table_with_variance.py
```

This reads existing `results/<Env>/<question>/params_*.json` files when present and
reuses the same hyperparameters.

## Scalability (Q4)

```bash
cd scripts
python scalability_experiments.py
# or via the legacy runner:
python run_experiments.py --experiment scalability --env CartPole \
    --client_counts "5,10,20,50" --episodes 600
```

Plot the bar summary:

```bash
python utils/plot_scalability_bar.py
```

## Ablations (A1 / A2 / A3)

`scripts/ablation_experiments.py` computes the federation-gap metrics described in
the paper (Q-error and policy-value gap) using a high-D reference QHD agent for
Q* estimation.

```bash
cd scripts

# A1 — vary encoder dimension D (m = 4D)
python ablation_experiments.py --metric all --which A1 \
    --d_grid 16 32 64 128 256 512 1024 2048 \
    --qhd_lr 0.2 --eps_decay 0.9 --rl_discount 0.9 --episodes 600 --lc_window 20

# A2 — fix D, vary anchor count m
python ablation_experiments.py --metric all --which A2 \
    --d_fixed 512 --m_grid 51 102 256 512 1024 2048 \
    --qhd_lr 0.2 --eps_decay 0.9 --rl_discount 0.9 --episodes 600 --lc_window 20

# A3 — vary ridge regularisation λ (LunarLander shows the natural U-shape)
python ablation_experiments.py --env LunarLander --which A3 --metric Q_error \
    --episodes 60 --d_fixed 1024 --m_fixed 256 --num_seeds 3 \
    --lambda_grid 0.001 0.01 0.1 1 10 100 1000 10000 \
    --qhd_lr 0.1 --eps_decay 0.990 --ref_eps_decay 0.995
```

Outputs land in `results/ablation/` and `results/figures/`.

## Hyperparameter search

```bash
python utils/hparam_search.py --method_type qhd    --env CartPole --episodes 500
python utils/hparam_search.py --method_type dqn    --env CartPole --episodes 500
python utils/hparam_search.py --method_type hetero --env LunarLander --episodes 100 \
    --eps_decay_values 0.900,0.990,0.995
```

`HETERO_GRID` sweeps `lr ∈ {0.1, 0.2, 0.5}`, `agg ∈ {10, 25}`, and
`exploration_decay ∈ {0.990, 0.995, 0.999}` by default.

## Output structure

```
results/
└── <Environment>/
    ├── q1_homogeneous/
    │   ├── <Method_Name>.json        # reward_history, success_history,
    │   │                              # training_time, projection_residuals, ...
    │   ├── params_<method_key>.json   # exact CLI hyperparameters
    │   └── summary.json               # cross-method roll-up
    └── q2_heterogeneous/
        └── ...
```

Each result JSON conforms to the `ExperimentResults` schema defined in
`scripts/experiments_runner.py`.

## Supported environments

| Env | State dim | Actions | Notes |
|---|---|---|---|
| CartPole-v1 | 4 | 2 | |
| Acrobot-v1 | 6 | 3 | |
| LunarLander-v3 | 8 | 4 | Box2D. |
| MountainCar-v0 | 2 | 3 | Sparse reward; use `qhd_agg_interval=50`. |
| GridWorld | 2 | 4 | Custom (`env/gridworld.py`). |
| CliffWalking | 4×12 | 4 | |
| Taxi-v3 | 4 (continuous) | 6 | |
| Pong / Freeway | 128 (RAM) | 6 / 3 | Requires `ale-py`. |

## Citation

If you use this code, please cite the FedQHD paper (see `main.tex` / `main.bib`).
