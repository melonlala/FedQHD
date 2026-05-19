from env.utils import create_env
from env.bounds import state_bounds
from agent.q_learning_agent import QLearningAgent
from agent.dqn_agent import DQNAgent
from agent.qhd_agent import QHDAgent
from agent.sarsa_agent import SARSAAgent
from agent.linear_sarsa_agent import LinearSARSAAgent
from agent.hd_sarsa_agent import HDSARSAAgent
import numpy as np
import tqdm
from agent.fedavg import FedAvgAgent
from plot_utils import visualize_reward_history
def train_single_agent(episodes, args):
    """Train a single agent with early stopping when model vector converges
    
    Stops training when model vectors change less than threshold for 10 consecutive episodes.
    """
    env= create_env(args)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    if args.agent_type == 'q_learning':
        agent = QLearningAgent(
            state_space=env.state_space,
            action_dim=action_dim,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            exploration_rate=args.exploration_rate,
            exploration_decay=args.exploration_decay,
            exploration_min=args.exploration_min
        )
    elif args.agent_type == 'dqn':
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            exploration_rate=args.exploration_rate,
            exploration_decay=args.exploration_decay,
            exploration_min=args.exploration_min,
            batch_size=64,
            target_update_freq=100
        )
    elif args.agent_type == 'qhd':
        agent = QHDAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            exploration_rate=args.exploration_rate,
            exploration_decay=args.exploration_decay,
            exploration_min=args.exploration_min,
            state_bounds=state_bounds[args.env]
        )
    elif args.agent_type == 'sarsa':
        agent = SARSAAgent(
            grid_size=args.grid_size,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            exploration_rate=args.exploration_rate,
            exploration_decay=args.exploration_decay,
            exploration_min=args.exploration_min
        )
    elif args.agent_type == 'linear_sarsa':
        agent = LinearSARSAAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            exploration_rate=args.exploration_rate,
            exploration_decay=args.exploration_decay,
            exploration_min=args.exploration_min,
            feature_type='rbf',
            state_bounds=state_bounds[args.env]
        )
    elif args.agent_type == 'hd_sarsa':
        agent = HDSARSAAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            exploration_rate=args.exploration_rate,
            exploration_decay=args.exploration_decay,
            exploration_min=args.exploration_min,
            state_bounds=state_bounds[args.env],
            hd_dim=args.hyperdimension
        )
    
    reward_history = []
    success_history = []
    
    # Early stopping: track model vector changes
    no_change_count = 0
    convergence_window = 20  # Stop if no change for 10 consecutive episodes

    for episode in tqdm.tqdm(range(episodes), desc="Training Single Agent"):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update_model(state, action, reward, next_state, done)
            
            state = next_state
            
            total_reward += reward

        agent.decay_exploration()
        success_history.append(1 if env.is_success(state) else 0)
        reward_history.append(total_reward)
        
        if total_reward==500:
            no_change_count += 1
        else:
            no_change_count = 0
        
        # Early stopping if converged
        if no_change_count >= convergence_window and episode >=1000:
            break

    return agent, reward_history, success_history

def train_fedavg(episodes, args):
    """
    Train multiple agents using Federated Averaging.
    
    Args:
        episodes: Total number of episodes
        args: Command line arguments
        aggregation_interval: How often to aggregate (every N episodes)
    """
    
    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_dim

    fed_agent = FedAvgAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        agent_type=args.agent_type,
        num_agents=num_agents,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        exploration_rate=args.exploration_rate,
        exploration_decay=args.exploration_decay,
        exploration_min=args.exploration_min,
        hd_dim=args.hyperdimension,
        state_bounds=state_bounds[args.env]
    )
    
    # Track rewards for each agent separately and globally
    agent_reward_histories = [[] for _ in range(num_agents)]
    agent_success_histories = [[] for _ in range(num_agents)]
    global_reward_history = []
    global_success_history = []
    
    print(f"Training {num_agents} agents with FedAvg...")
    print(f"Aggregation every {args.aggregation_interval} steps\n")

    agent_step_counts = [0] * num_agents  # Track steps for each agent
    # use hard synchronous step_based aggregation

    
    for episode in range(episodes):
        episode_rewards = []
        episode_successes = []
        
        # Each agent trains on their own environment
        for agent_id in range(num_agents):
            agent = fed_agent.get_agent(agent_id)
            env = envs[agent_id]
            state, _ = env.reset()
            done = False
            total_reward = 0
                
            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.update_model(state, action, reward, next_state, done)
                
                state = next_state
                
                total_reward += reward
                agent_step_counts[agent_id] += 1
                 

            # Record individual agent performance
            agent_reward_histories[agent_id].append(total_reward)
            agent_success_histories[agent_id].append(1 if env.is_success(state) else 0)
            episode_rewards.append(total_reward)
            episode_successes.append(1 if env.is_success(state) else 0)
        
        # if all(count >= aggregation_threshold for count in agent_step_counts):
        if (episode + 1) % args.aggregation_interval == 0:
            fed_agent.aggregate()
            fed_agent.distribute()
            agent_step_counts = [0] * num_agents
        
        global_reward_history.append(np.mean(episode_rewards))
        global_success_history.append(np.mean(episode_successes))
        
        # Decay exploration for all agents
        fed_agent.decay_exploration()
    
    return fed_agent, global_reward_history, global_success_history, agent_reward_histories, agent_success_histories

def train_parallel(episodes, args, aggregation_threshold=10):
    """
    Train multiple agents in parallel to compare with FedAvg.
    """
    envs = [create_env(args) for _ in range(args.agent_num)]
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_dim
    num_agents = args.agent_num
    agents = []
    
    for _ in range(num_agents):
        if args.agent_type == 'q_learning':
            agent = QLearningAgent(
                state_space=env.state_space,
                action_dim=action_dim,
                learning_rate=args.learning_rate,
                discount_factor=args.discount_factor,
                exploration_rate=args.exploration_rate,
                exploration_decay=args.exploration_decay,
                exploration_min=args.exploration_min
            )
        # Add other agent types as needed
        elif args.agent_type == 'linear_sarsa':
            agent = LinearSARSAAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=args.learning_rate,
                discount_factor=args.discount_factor,
                exploration_rate=args.exploration_rate,
                exploration_decay=args.exploration_decay,
                exploration_min=args.exploration_min,
                feature_type='rbf',
                state_bounds=state_bounds[args.env]
            )
        elif args.agent_type == 'sarsa':
            agent = SARSAAgent(
                grid_size=args.grid_size,
                learning_rate=args.learning_rate,
                discount_factor=args.discount_factor,
                exploration_rate=args.exploration_rate,
                exploration_decay=args.exploration_decay,
                exploration_min=args.exploration_min
            )
        elif args.agent_type == 'hd_sarsa':
            agent = HDSARSAAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=args.learning_rate,
                discount_factor=args.discount_factor,
                exploration_rate=args.exploration_rate,
                exploration_decay=args.exploration_decay,
                exploration_min=args.exploration_min,
                state_bounds=state_bounds[args.env]
            )
        elif args.agent_type == 'dqn':
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=args.learning_rate,
                discount_factor=args.discount_factor,
                exploration_rate=args.exploration_rate,
                exploration_decay=args.exploration_decay,
                exploration_min=args.exploration_min,
                batch_size=64,
                target_update_freq=100
            )
        elif args.agent_type == 'qhd':
            agent = QHDAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=args.learning_rate,
                discount_factor=args.discount_factor,
                exploration_rate=args.exploration_rate,
                exploration_decay=args.exploration_decay,
                exploration_min=args.exploration_min,
                state_bounds=state_bounds[args.env]
            )
        agents.append(agent)
    
    # Track rewards for each agent separately and globally
    agent_reward_histories = [[] for _ in range(num_agents)]
    agent_success_histories = [[] for _ in range(num_agents)]
    global_reward_history = []
    global_success_history = []
    
    print(f"Training {num_agents} agents in parallel...\n")
    
    for episode in range(episodes):
        episode_rewards = []
        episode_successes = []
        
        for agent_id, agent in enumerate(agents):
            env = envs[agent_id]
            state, _ = env.reset()
            done = False
            total_reward = 0
                
            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.update_model(state, action, reward, next_state, done)
                
                state = next_state
                
                total_reward += reward
                 
            # Record individual agent performance
            agent_reward_histories[agent_id].append(total_reward)
            agent_success_histories[agent_id].append(1 if env.is_success(state) else 0)
            episode_rewards.append(total_reward)
            episode_successes.append(1 if env.is_success(state) else 0)
        
        global_reward_history.append(np.mean(episode_rewards))
        global_success_history.append(np.mean(episode_successes))
        
        # Decay exploration for all agents
        for agent in agents:
            agent.decay_exploration()
    
    return  global_reward_history, global_success_history, agent_reward_histories, agent_success_histories

def train(episodes, args):
    """Main training dispatcher"""
    if args.agent_num == 1:
        # agent, global_rewards, global_success = train_single_agent(episodes, args)
        # return agent, global_rewards, global_success, [global_rewards], [global_success]
        agent, global_rewards, global_success, agent_rewards, agent_success = train_fedavg(
            episodes, args, args.aggregation_interval
        )
        return agent, global_rewards, global_success, agent_rewards, agent_success
    elif args.agent_num > 1:
        agent, global_rewards, global_success, agent_rewards, agent_success = train_fedavg(
            episodes, args, args.aggregation_interval
        )
        return agent, global_rewards, global_success, agent_rewards, agent_success

def train_fedavg_over_Q(episodes, args, single_agent_model_vector, encoder_rff_parameters):
    """output the MSE loss of FedAvg over Q-values, using 2000-episode trained single agent model to approximate true Q-values
    
    The reference single agent model is used to compute "true" Q-values for states encountered during training.
    """
    
    # Train FedAvg agents with MSE calculation
    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_dim

    fed_agent = FedAvgAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        agent_type=args.agent_type,
        num_agents=num_agents,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        exploration_rate=args.exploration_rate,
        exploration_decay=args.exploration_decay,
        exploration_min=args.exploration_min,
        hd_dim=args.hyperdimension,
        state_bounds=state_bounds[args.env]
    )
    
    mse_losses = []
    
    print(f"Training {num_agents} agents with FedAvg...")
    print(f"Aggregation every {args.aggregation_interval} steps\n")
    
    # tqdm for progress bar
    for episode in tqdm.tqdm(range(episodes), desc=f"Training {args.agent_num} {args.hyperdimension}-dimensional FedAvg agents with MSE Calculation"):
    
        episodes_mse_losses = np.zeros(num_agents)
        
        # Each agent trains on their own environment
        for agent_id in range(num_agents):
            agent = fed_agent.get_agent(agent_id)
            env = envs[agent_id]
            state, _ = env.reset()
            done = False
            total_reward = 0
                
            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.update_model(state, action, reward, next_state, done)
                
                # Calculate MSE by comparing Q-values for current state
                single_state_hv = np.dot(encoder_rff_parameters['omega'], agent._normalize_state(state)) + encoder_rff_parameters['b']
                single_state_hv = encoder_rff_parameters['scale_factor'] * np.cos(single_state_hv)

                true_Q = np.dot(single_agent_model_vector, single_state_hv)
                estimated_Q = agent.get_all_q_values(state)
                
                mse = np.mean((estimated_Q - true_Q) ** 2)
                episodes_mse_losses[agent_id] += mse
                
                state = next_state
        
        mse_losses.append(np.mean(episodes_mse_losses))
        episodes_mse_losses = np.zeros(num_agents)
        
        # if all(count >= aggregation_threshold for count in agent_step_counts):
        if (episode + 1) % args.aggregation_interval == 0:
            fed_agent.aggregate()
            fed_agent.distribute()
            agent_step_counts = [0] * num_agents
        
        # Decay exploration for all agents
        fed_agent.decay_exploration()

        if (episode + 1) % 10 == 0:
            avg_mse = np.mean(mse_losses[-100:]) if len(mse_losses) > 0 else 0
            print(f"Episode {episode + 1}/{episodes}, Recent MSE: {avg_mse:.6f}")
    return mse_losses

def train_fedavg_over_V(episodes, args):
    """output the MSE loss of FedAvg over V-values, using 2000-episode trained single agent model to approximate true V-values
    Values are computed as the averaged reward of current policy over num_traj trajectories given.
    The reference model is used to compute "true" V-values for states encountered during training.
    """
    
    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_dim

    fed_agent = FedAvgAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        agent_type=args.agent_type,
        num_agents=num_agents,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        exploration_rate=args.exploration_rate,
        exploration_decay=args.exploration_decay,
        exploration_min=args.exploration_min,
        hd_dim=args.hyperdimension,
        state_bounds=state_bounds[args.env]
    )
    
    v = []

    # OPTIMIZATION: Only compute V-values every N episodes to speed up training
    v_eval_interval = getattr(args, 'v_eval_interval', 10)  # Default: evaluate every 10 episodes

    print(f"Training {num_agents} agents with FedAvg...")
    print(f"Aggregation every {args.aggregation_interval} steps")
    print(f"V-value evaluation every {v_eval_interval} episodes (for speed)\n")

    # tqdm for progress bar
    for episode in tqdm.tqdm(range(episodes), desc=f"Training {args.agent_num} {args.hyperdimension}-dimensional FedAvg agents with MSE Calculation"):
        # Each agent trains on their own environment
        for agent_id in range(num_agents):
            agent = fed_agent.get_agent(agent_id)
            env = envs[agent_id]
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update_model(state, action, reward, next_state, done)
                state = next_state

            # OPTIMIZATION: Only estimate V-values every v_eval_interval episodes
            # This significantly speeds up training while maintaining good tracking
            if episode % v_eval_interval == 0 or episode == episodes - 1:
                v_estimates = []
                for _ in range(args.num_traj):
                    state, _ = env.reset()
                    done = False
                    traj_reward = 0
                    step_count = 0
                    while not done:
                        # Use greedy action selection (no exploration) for V-value estimation
                        q_values = agent.get_all_q_values(state)
                        action = np.argmax(q_values)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        traj_reward += reward
                        step_count += 1
                        state = next_state
                    v_estimates.append(traj_reward)

                v.append(np.mean(v_estimates))
            else:
                # Use last computed V-value for episodes where we don't evaluate
                if len(v) > 0:
                    v.append(v[-1])
        
        # if all(count >= aggregation_threshold for count in agent_step_counts):
        if (episode + 1) % args.aggregation_interval == 0:
            fed_agent.aggregate()
            fed_agent.distribute()
            agent_step_counts = [0] * num_agents
        
        # Decay exploration for all agents
        fed_agent.decay_exploration()

        if (episode + 1) % 10 == 0:
            avg_v = np.mean(v[-100:]) if len(v) > 0 else 0
            print(f"Episode {episode + 1}/{episodes}, Recent V: {avg_v:.6f}")
    return v
    

def multi_run_train(episodes, args):
    """Run multiple training sessions and average results"""
    runs = args.runs
    all_global_rewards = []
    all_global_success = []
    all_agent_rewards = []
    all_agent_success = []

    for run in range(runs):
        print(f"Starting run {run + 1}/{runs}")
        _, global_rewards, global_success, agent_rewards, agent_success = train(episodes, args)
        all_global_rewards.append(global_rewards)
        all_global_success.append(global_success)
        all_agent_rewards.append(agent_rewards)
        all_agent_success.append(agent_success)

    # Average results across runs
    avg_global_rewards = np.mean(all_global_rewards, axis=0)
    avg_global_success = np.mean(all_global_success, axis=0)
    avg_agent_rewards = [np.mean(agent_rewards, axis=0) for agent_rewards in zip(*all_agent_rewards)]
    avg_agent_success = [np.mean(agent_success, axis=0) for agent_success in zip(*all_agent_success)]

    return None, avg_global_rewards, avg_global_success, avg_agent_rewards, avg_agent_success
