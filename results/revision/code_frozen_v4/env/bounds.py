state_bounds = {
    'LunarLander': [
                [-1.0, 1.0], [-0.2, 1.4], [-1.5, 1.5], [-2.0, 0.5],
                [-0.5, 0.5], [-1.0, 1.0], [0.0, 1.0],  [0.0, 1.0]
            ],
    'MountainCar': [
                [-1.2, 0.6], [-0.07, 0.07]
            ],
    'CartPole': [
                [-2.4, 2.4], [-3.0, 3.0], [-0.5, 0.5], [-3.5, 3.5]
            ],
    'CliffWalking': [
                [0, 11], [0, 3]
            ],
    # Acrobot-v1 observation: [cos θ1, sin θ1, cos θ2, sin θ2, θ1_dot, θ2_dot]
    # (fixed 2026-10-07: cos θ2 / sin θ2 previously got the ±4π / ±9π velocity bounds)
    'Acrobot': [
                [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],
                [-12.57, 12.57], [-28.27, 28.27]
            ],
    'GridWorld': [
                [0, 64], [0, 3]
            ],
    'Taxi': [
                [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
            ],
    'Pong': [[0.0, 1.0]] * 128,  # RAM state (128 bytes normalized to [0,1])
    'Freeway': [[0.0, 1.0]] * 128  # RAM state (128 bytes normalized to [0,1])
}