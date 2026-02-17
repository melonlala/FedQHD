"""Quick test for new baseline implementations"""
import sys
import argparse
from experiments_runner import train_oracle_dqn, train_truncate_fedavg_qhd, train_distillation_fedqhd

# Create minimal args
class Args:
    def __init__(self):
        self.env = 'CartPole'
        self.agent_num = 2
        self.learning_rate = 0.01
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.exploration_min = 0.01
        self.hyperdimension = 1000
        self.rff_gamma = 1.0
        self.aggregation_interval = 10

args = Args()
episodes = 20  # Short test

print("="*60)
print("Testing New Baselines")
print("="*60)

try:
    print("\n1. Testing Oracle DQN...")
    result1 = train_oracle_dqn(episodes, args)
    print(f"✓ Oracle DQN works! Final reward: {result1.final_avg_reward:.2f}")
except Exception as e:
    print(f"✗ Oracle DQN failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n2. Testing Truncate FedAvg-QHD...")
    result2 = train_truncate_fedavg_qhd(episodes, args)
    print(f"✓ Truncate FedAvg-QHD works! Final reward: {result2.final_avg_reward:.2f}")
except Exception as e:
    print(f"✗ Truncate FedAvg-QHD failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n3. Testing Distillation FedQHD...")
    result3 = train_distillation_fedqhd(episodes, args)
    print(f"✓ Distillation FedQHD works! Final reward: {result3.final_avg_reward:.2f}")
except Exception as e:
    print(f"✗ Distillation FedQHD failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All new baselines tested!")
print("="*60)
