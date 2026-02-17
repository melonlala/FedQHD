# Baseline Implementation Verification

## Baselines to Check:

### Q1: Homogeneous Encoders (5 baselines)
1. **Independent QHD** - Single agent, no federation
2. **FedQHD (Homogeneous)** - Federated QHD with shared encoder
3. **Oracle QHD** - Centralized pooled data (upper bound)
4. **Oracle DQN** - Centralized DQN with pooled data
5. **FedAvg-DQN** - Federated DQN

### Q2: Heterogeneous Encoders (3 methods)
1. **FedQHD (Heterogeneous)** - Anchor-based aggregation
2. **Truncate/Pad FedAvg-QHD** - Naive padding/truncation
3. **Knowledge Distillation FedQHD** - Policy distillation

## Key Questions to Verify:

### 1. Independent QHD ✅
- [x] Uses QHDAgent (Q-learning)? ✅ YES
- [x] Single agent, no aggregation? ✅ YES
- [x] Proper Q-learning update? ✅ YES (line 153: q_target = reward + gamma * max(next_q_values))

### 2. Oracle QHD ✅
- [x] Uses QHDAgent? ✅ YES
- [x] Single centralized agent? ✅ YES
- [x] Trains on data from ALL clients? ✅ YES (iterates through all envs)
- [x] Proper Q-learning update? ✅ YES (same as Independent QHD)

### 3. FedQHD (Homogeneous) ✅
- [x] Uses QHDAgent for all clients? ✅ YES (via FedAvgAgent with agent_type='qhd')
- [x] Same encoder dimension for all? ✅ YES (args.hyperdimension)
- [x] Parameter averaging: W^glob = Σ(π_i * W_i)? ✅ YES (fedavg.py:128-132)
- [x] Aggregates every K episodes? ✅ YES (args.aggregation_interval)
- [x] Distributes global model back? ✅ YES (fedavg.py:152)

### 4. Oracle DQN ✅
- [x] Uses DQNAgent? ✅ YES
- [x] Single centralized agent? ✅ YES
- [x] Trains on data from ALL clients? ✅ YES (iterates through all envs)
- [x] Proper DQN update with replay buffer? ✅ YES (dqn_agent.py:125-131)

### 5. FedAvg-DQN ✅
- [x] Uses DQNAgent for all clients? ✅ YES (via FedAvgAgent with agent_type='dqn')
- [x] FedAvg aggregation of neural network weights? ✅ YES (fedavg.py:110-119)
- [x] Aggregates every K episodes? ✅ YES (args.aggregation_interval)

### 6. FedQHD (Heterogeneous) ✅
- [x] Different encoder dimensions per client? ✅ YES (D_i ∈ {1000, 5000, 10000, 50000})
- [x] Uses anchor states? ✅ YES (m=200 anchor states)
- [x] Function-space aggregation: Q^glob = Σ(π_i * Q^ref_i)? ✅ YES (line 423)
- [x] Ridge regression to map back: W^glob_i = (X_i^T X_i + λI)^-1 X_i^T Q^glob? ✅ YES (line 430-434)
- [x] Tracks projection residual? ✅ YES (line 439-442)

### 7. Truncate/Pad FedAvg-QHD ✅
- [x] Different dimensions per client? ✅ YES (D_i ∈ {1000, 5000, 10000, 50000})
- [x] Pads shorter vectors with zeros? ✅ YES (line 711-715)
- [x] Truncates longer vectors? ✅ YES (line 717-719)
- [x] Averages in common dimension? ✅ YES (line 722)
- [x] Maps back to original dimensions? ✅ YES (line 725-732)

### 8. Knowledge Distillation ✅
- [x] Different dimensions per client? ✅ YES (D_i ∈ {1000, 5000, 10000, 50000})
- [x] Computes ensemble Q-values on distillation set? ✅ YES (line 841-848)
- [x] Distills ensemble to each client? ✅ YES (line 850-860)
- [x] Uses gradient-based distillation? ✅ YES (minimizes ||Q_pred - Q_target||^2)
