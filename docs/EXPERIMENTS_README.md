# FedQHD Experiments Guide

This document describes how to run all experiments mentioned in the paper (sections/experiments.tex).

## Quick Start

### Install Dependencies

```bash
pip install numpy matplotlib gymnasium tqdm
pip install ale-py  # For Atari environments (Pong, Freeway)
```

### Run Basic Experiments

```bash
# Q1: Homogeneous Encoders Comparison
python run_experiments.py --experiment q1 --env CartPole --episodes 600 --agent_num 5

# Q2: Heterogeneous Encoders
python run_experiments.py --experiment q2 --env CartPole --episodes 600 --agent_num 5

# Q4: Scalability Analysis
python run_experiments.py --experiment scalability --env CartPole --episodes 600 --client_counts "5,10,20,50"

# Run all experiments
python run_experiments.py --experiment all --env CartPole --episodes 600 --agent_num 5
```

## Experiment Types

### Q1: Performance Comparison - Homogeneous Encoders

**Research Question:** Does FedQHD achieve effective knowledge aggregation compared to independent learning and existing federated RL methods?

**Baselines:**
- Independent QHD: Single agent without federation (lower bound)
- Oracle QHD: Centralized training with pooled data (upper bound)
- FedAvg-DQN: Federated deep Q-learning
- **FedQHD (Ours)**: Homogeneous encoder federation

**Command:**
```bash
python run_experiments.py \
    --experiment q1 \
    --env CartPole \
    --episodes 600 \
    --agent_num 5 \
    --aggregation_interval 50 \
    --hyperdimension 10000 \
    --runs 3  # Run 3 times and average
```

**Outputs:**
- Learning curves comparison (Figure: learning_curves_comparison.pdf)
- Training time table (computation_cost_table.tex)
- Performance summary (performance_summary_table.tex)

### Q2: Heterogeneous Encoders and Anchor-Based Aggregation

**Research Question:** How does FedQHD handle heterogeneous encoders via anchor-based aggregation?

**Methodology:** Clients use different encoder dimensions (1000, 5000, 10000, 50000) and bandwidths. Server uses anchor-based aggregation (Eq. 212 in methodology.tex).

**Command:**
```bash
python run_experiments.py \
    --experiment q2 \
    --env CartPole \
    --episodes 600 \
    --agent_num 5 \
    --aggregation_interval 50 \
    --runs 3
```

**Outputs:**
- Homogeneous vs. Heterogeneous comparison (hetero_comparison.pdf)
- Projection residual analysis (projection_residual_analysis.pdf)
- Performance degradation metrics

### Q4: Scalability Analysis

**Research Question:** How does FedQHD scale with the number of clients?

**Methodology:** Vary N ∈ {5, 10, 20, 50} and measure performance improvement and convergence speed.

**Command:**
```bash
python run_experiments.py \
    --experiment scalability \
    --env CartPole \
    --episodes 600 \
    --client_counts "5,10,20,50" \
    --aggregation_interval 50
```

**Outputs:**
- Performance vs. N (scalability_performance.pdf)
- Improvement vs. N (scalability_improvement.pdf)
- Training time vs. N (scalability_time.pdf)
- Learning curves for each N (scalability_learning_curves.pdf)

## Supported Environments

### Classic Control
- **CartPole-v1**: 4D state, 2 actions
- **Acrobot-v1**: 6D state, 3 actions
- **MountainCar-v0**: 2D state, 3 actions
- **LunarLander-v3**: 8D state, 4 actions

### Grid Worlds
- **Taxi-v3**: Discrete state space (500 states)
- **CliffWalking**: 4x12 grid

### Atari (RAM-based)
- **Pong**: 128D RAM state, 6 actions
- **Freeway**: 128D RAM state, 3 actions

## Configuration Parameters

### Agent Parameters
- `--hyperdimension`: HD vector dimension (default: 10000)
- `--learning_rate`: Learning rate (default: 0.01)
- `--discount_factor`: Gamma for TD learning (default: 0.99)
- `--exploration_rate`: Initial epsilon (default: 1.0)
- `--exploration_decay`: Epsilon decay rate (default: 0.9995)
- `--rff_gamma`: RFF kernel bandwidth (default: 1.0)

### Federation Parameters
- `--agent_num`: Number of clients (default: 5)
- `--aggregation_interval`: Episodes between aggregations (default: 50)

### Experiment Parameters
- `--episodes`: Training episodes (default: 600)
- `--runs`: Number of independent runs for averaging (default: 1)
- `--output_dir`: Results directory (default: results/)

## Example Experiment Scripts

### Paper Figure Generation

```bash
# Generate all figures for CartPole
python run_experiments.py --experiment all --env CartPole --episodes 1000 --runs 5 --visualize

# Generate all figures for Acrobot
python run_experiments.py --experiment all --env Acrobot --episodes 1000 --runs 5 --visualize

# Generate all figures for LunarLander
python run_experiments.py --experiment all --env LunarLander --episodes 1000 --runs 5 --visualize

# Generate all figures for Atari Pong
python run_experiments.py --experiment all --env Pong --episodes 2000 --runs 5 --visualize
```

### Ablation Studies

```bash
# Vary aggregation interval
for interval in 10 25 50 100; do
    python run_experiments.py \
        --experiment q1 \
        --env CartPole \
        --episodes 600 \
        --aggregation_interval $interval \
        --output_dir results/ablation_interval
done

# Vary hyperdimension
for dim in 1000 5000 10000 50000; do
    python run_experiments.py \
        --experiment q1 \
        --env CartPole \
        --episodes 600 \
        --hyperdimension $dim \
        --output_dir results/ablation_dimension
done

# Vary number of clients
for N in 2 5 10 20; do
    python run_experiments.py \
        --experiment q1 \
        --env CartPole \
        --episodes 600 \
        --agent_num $N \
        --output_dir results/ablation_clients
done
```

## Output Structure

Results are saved in the following structure:

```
results/
├── CartPole/
│   ├── q1_homogeneous/
│   │   ├── Independent_QHD.json
│   │   ├── FedQHD_(Homogeneous).json
│   │   ├── Oracle_QHD.json
│   │   ├── FedAvg-DQN.json
│   │   ├── summary.json
│   │   ├── learning_curves_comparison.pdf
│   │   ├── computation_cost_table.tex
│   │   └── performance_summary_table.tex
│   ├── q2_heterogeneous/
│   │   ├── FedQHD_(Heterogeneous).json
│   │   ├── hetero_comparison.pdf
│   │   └── projection_residual_analysis.pdf
│   └── scalability/
│       ├── N_5/
│       ├── N_10/
│       ├── N_20/
│       ├── N_50/
│       ├── scalability_summary.json
│       ├── scalability_performance.pdf
│       ├── scalability_improvement.pdf
│       └── scalability_time.pdf
├── Acrobot/
│   └── ...
└── LunarLander/
    └── ...
```

## Visualization

### Generate Plots from Saved Results

```bash
# Visualize results after running experiments
python visualize_results.py --results_dir results/CartPole/q1_homogeneous

# Or enable automatic visualization
python run_experiments.py --experiment q1 --env CartPole --episodes 600 --visualize
```

## Implementation Details

### FedQHD Algorithm

**Homogeneous Encoders (Algorithm 1 in methodology.tex):**
- All clients share the same RFF encoder parameters
- Direct parameter averaging: W^glob = Σ(π_i * W_i)
- Implemented in `experiments_runner.py::train_fedqhd_homogeneous()`

**Heterogeneous Encoders (Algorithm 2 in methodology.tex):**
1. Each client i computes anchor Q-values: Q^ref_i = X_i * W_i
2. Server averages: Q^glob_ref = Σ(π_i * Q^ref_i)
3. Server compiles back: W^glob_i = (X_i^H X_i + λI)^-1 X_i^H Q^glob_ref
4. Clients update with W^glob_i
- Implemented in `experiments_runner.py::train_fedqhd_heterogeneous()`

### Key Files

- `experiments_runner.py`: Core experiment implementations
- `scalability_experiments.py`: Scalability analysis
- `visualize_results.py`: Plot generation
- `run_experiments.py`: Master experiment runner
- `agent/hd_sarsa_agent.py`: QHD agent implementation
- `agent/fedavg.py`: Federation coordinator
- `env/utils.py`: Environment wrappers

## Troubleshooting

### Out of Memory
If you encounter OOM errors with large hyperdimensions:
```bash
# Use smaller dimension
python run_experiments.py --experiment q1 --env CartPole --hyperdimension 5000

# Or reduce number of agents
python run_experiments.py --experiment q1 --env CartPole --agent_num 3
```

### Slow Training
For faster experiments during development:
```bash
# Reduce episodes
python run_experiments.py --experiment q1 --env CartPole --episodes 300

# Reduce runs
python run_experiments.py --experiment q1 --env CartPole --runs 1
```

### Atari Dependencies
If Atari environments fail:
```bash
pip install ale-py
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{fedqhd2025,
  title={FedQHD: Federated Reinforcement Learning via Hyperdimensional Computing},
  author={Your Name},
  booktitle={Conference},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email].
