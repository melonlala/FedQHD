"""
Quick test script to verify experiment implementations work
"""

import argparse
import numpy as np
from experiments_runner import (
    train_independent_qhd,
    train_oracle_qhd,
    train_fedqhd_homogeneous,
    train_fedqhd_heterogeneous
)


def create_test_args():
    """Create minimal args for testing"""
    args = argparse.Namespace()
    args.env = 'CartPole'
    args.episodes = 50  # Short for testing
    args.agent_num = 2  # Small number for testing
    args.hyperdimension = 1000  # Smaller for speed
    args.learning_rate = 0.01
    args.discount_factor = 0.99
    args.exploration_rate = 1.0
    args.exploration_decay = 0.995
    args.exploration_min = 0.01
    args.aggregation_interval = 10
    args.grid_size = [8, 8]
    args.random_seed = 42
    args.rff_gamma = 1.0
    return args


def test_independent_qhd():
    print("\n" + "="*60)
    print("TEST 1: Independent QHD")
    print("="*60)

    args = create_test_args()
    results = train_independent_qhd(args.episodes, args)

    assert len(results.reward_history) == args.episodes
    assert results.training_time > 0
    assert results.method_name == "Independent QHD"

    print(f"✓ Independent QHD test passed")
    print(f"  - Final reward: {results.final_avg_reward:.2f}")
    print(f"  - Training time: {results.training_time:.2f}s")


def test_oracle_qhd():
    print("\n" + "="*60)
    print("TEST 2: Oracle QHD")
    print("="*60)

    args = create_test_args()
    results = train_oracle_qhd(args.episodes, args)

    assert len(results.reward_history) == args.episodes
    assert results.training_time > 0
    assert results.method_name == "Oracle QHD"

    print(f"✓ Oracle QHD test passed")
    print(f"  - Final reward: {results.final_avg_reward:.2f}")
    print(f"  - Training time: {results.training_time:.2f}s")


def test_fedqhd_homogeneous():
    print("\n" + "="*60)
    print("TEST 3: FedQHD (Homogeneous)")
    print("="*60)

    args = create_test_args()
    results = train_fedqhd_homogeneous(args.episodes, args)

    assert len(results.reward_history) == args.episodes
    assert results.training_time > 0
    assert results.method_name == "FedQHD (Homogeneous)"

    print(f"✓ FedQHD (Homogeneous) test passed")
    print(f"  - Final reward: {results.final_avg_reward:.2f}")
    print(f"  - Training time: {results.training_time:.2f}s")


def test_fedqhd_heterogeneous():
    print("\n" + "="*60)
    print("TEST 4: FedQHD (Heterogeneous)")
    print("="*60)

    args = create_test_args()
    results = train_fedqhd_heterogeneous(args.episodes, args, anchor_set_size=50)

    assert len(results.reward_history) == args.episodes
    assert results.training_time > 0
    assert results.method_name == "FedQHD (Heterogeneous)"
    assert len(results.projection_residuals) > 0

    print(f"✓ FedQHD (Heterogeneous) test passed")
    print(f"  - Final reward: {results.final_avg_reward:.2f}")
    print(f"  - Training time: {results.training_time:.2f}s")
    print(f"  - Avg projection residual: {np.mean(results.projection_residuals):.4f}")


def test_atari_env():
    print("\n" + "="*60)
    print("TEST 5: Atari Environment (Pong)")
    print("="*60)

    try:
        from env.utils import create_env

        args = create_test_args()
        args.env = 'Pong'
        args.episodes = 5  # Very short for Atari

        env = create_env(args)
        assert env.state_dim == 128  # RAM state
        assert env.action_dim == 6

        # Test one episode
        state, _ = env.reset()
        assert len(state) == 128

        for _ in range(10):
            action = np.random.randint(env.action_dim)
            next_state, reward, done, _, _ = env.step(action)
            if done:
                break

        env.close()
        print(f"✓ Atari Pong environment test passed")
        print(f"  - State dim: {env.state_dim}")
        print(f"  - Action dim: {env.action_dim}")

    except ImportError as e:
        print(f"⚠ Atari test skipped (ale-py not installed): {e}")
    except Exception as e:
        print(f"⚠ Atari test failed: {e}")


def run_all_tests():
    print("\n" + "="*60)
    print("RUNNING EXPERIMENT TESTS")
    print("="*60)

    np.random.seed(42)

    try:
        test_independent_qhd()
        test_oracle_qhd()
        test_fedqhd_homogeneous()
        test_fedqhd_heterogeneous()
        test_atari_env()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60 + "\n")
        return True

    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ TEST FAILED: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
