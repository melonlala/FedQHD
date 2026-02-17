import numpy as np
import matplotlib.pyplot as plt
import os
def visualize_reward_history(agent_num, rewards, args, title=None):
    """Plot reward history"""
    num_agent = agent_num
    avg_num = args.avg_window
    agent_type = args.agent_type
    output_file = f"{title}.png" if title else 'reward_history.png'
    output_path = os.path.join(args.output_dir, output_file)
    if avg_num > 1:
        reward_per = [np.mean(rewards[i:i+avg_num]) 
                     for i in range(0, len(rewards), avg_num)]
    else:
        reward_per = rewards
    
    # plot x as original episode numbers
    x = np.arange(len(reward_per)) * avg_num if avg_num > 1 else np.arange(len(reward_per))

    
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.plot(x, reward_per)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Reward History over {num_agent} {agent_type} Agents : {len(rewards)} Episodes')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reward history to {output_path}")


def visualize_success_history(success_history, avg_num, output_path='success_history.png'):
    """Plot success rate history"""
    if avg_num > 1:
        success_rate = [np.mean(success_history[i:i+avg_num]) 
                       for i in range(0, len(success_history), avg_num)]
    else:
        success_rate = success_history
    
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.plot(success_rate)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate over {len(success_history)} Episodes')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved success history to {output_path}")

def plot_combined_rewards(args, rewards_data, avg_window, output_dir):
    """Plot the global reward for each configuration on a single figure for comparison."""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of configurations
    num_configs = len(rewards_data)

    # calculate the shared y-axis range and x-axis range
    all_rewards = [reward for rewards in rewards_data.values() for reward in rewards[0]]
    y_range = (min(all_rewards), max(all_rewards) + 5)
    x_range = max(len(rewards[0]) for rewards in rewards_data.values())

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Create a single figure for all global rewards
    plt.figure(figsize=(8, 6))

    # plot subfigures for each configuration
    for config_idx, (num_agents, reward) in enumerate(sorted(rewards_data.items())):
        # Extract the global rewards
        global_rewards = reward[0]
        
        # Add smoothing if requested
        if avg_window > 1: 
            smoothed_global = []
            for i in range(0, len(global_rewards), avg_window):
                window_data = global_rewards[i:i+avg_window]
                smoothed_global.append(np.mean(window_data))
            global_rewards = smoothed_global
        
        # Calculate x-axis data (Episode numbers)
        x_data = np.arange(0, len(global_rewards) * avg_window, avg_window) if avg_window > 1 else np.arange(len(global_rewards))

        # subplot global rewards
        plt.subplot(num_configs, 1, config_idx + 1)
        plt.plot(x_data, global_rewards, color=colors[config_idx % len(colors)], linewidth=2)
        plt.ylim(y_range)
        plt.title(f'{num_agents} {args.agent_type} Agent{"s" if num_agents > 1 else ""}', fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.grid(True, alpha=0.3)
    # Adjust layout and save the figure
    plt.tight_layout()
    save_path = f"{output_dir}/global_reward_comparison_combined.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved global rewards comparison to {save_path}")

def calculate_convergence_episode(success_history, threshold=0.8, window=50):
    """
    Calculate when an agent converges (achieves consistent success).
    
    Args:
        success_history: List of success values (0 or 1) for each episode
        threshold: Success rate threshold to consider converged (default 0.8 = 80%)
        window: Rolling window size to check for convergence
    
    Returns:
        Episode number when convergence achieved, or None if never converged
    """
    if len(success_history) < window:
        return None
    
    for i in range(window, len(success_history) + 1):
        window_success_rate = np.mean(success_history[i-window:i])
        if window_success_rate >= threshold:
            return i  # Return the episode where convergence was achieved
    
    return None  # Never converged


def visualize_individual_agents(agent_reward_histories, agent_success_histories, 
                                avg_num, output_dir):
    """
    Plot individual agent performance alongside global average.
    Creates two plots: one for rewards and one for success rates.
    """
    num_agents = len(agent_reward_histories)
    # Define colors for different agents
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    # ========== Plot Individual Rewards ==========
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each agent's reward curve
    for agent_id in range(num_agents):
        rewards = agent_reward_histories[agent_id]
        if avg_num > 1:
            rewards_smoothed = [np.mean(rewards[i:i+avg_num]) 
                            for i in range(0, len(rewards), avg_num)]
        else:
            rewards_smoothed = rewards
        
        ax.plot(rewards_smoothed, label=f'Agent {agent_id + 1}', 
            color=colors[agent_id], alpha=0.6, linewidth=1.5)
    
    # Plot global average (thicker line)
    global_avg = [np.mean([agent_reward_histories[j][i] 
                        for j in range(num_agents)]) 
                for i in range(len(agent_reward_histories[0]))]
    
    if avg_num > 1:
        global_avg_smoothed = [np.mean(global_avg[i:i+avg_num]) 
                            for i in range(0, len(global_avg), avg_num)]
    else:
        global_avg_smoothed = global_avg
    
    ax.plot(global_avg_smoothed, label='Global Average', 
        color='black', linewidth=3, linestyle='--')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title(f'Individual Agent Rewards ({num_agents} Agents)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    reward_path = f"{output_dir}/{num_agents}_individual_rewards.png"
    plt.savefig(reward_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved individual agent rewards to {reward_path}")
    
    # ========== Plot Individual Success Rates ==========
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each agent's success rate
    for agent_id in range(num_agents):
        success = agent_success_histories[agent_id]
        if avg_num > 1:
            success_smoothed = [np.mean(success[i:i+avg_num]) 
                            for i in range(0, len(success), avg_num)]
        else:
            success_smoothed = success
        
        ax.plot(success_smoothed, label=f'Agent {agent_id + 1}', 
            color=colors[agent_id], alpha=0.6, linewidth=1.5)
    
    # Plot global average (thicker line)
    global_success = [np.mean([agent_success_histories[j][i] 
                            for j in range(num_agents)]) 
                    for i in range(len(agent_success_histories[0]))]
    
    if avg_num > 1:
        global_success_smoothed = [np.mean(global_success[i:i+avg_num]) 
                                for i in range(0, len(global_success), avg_num)]
    else:
        global_success_smoothed = global_success
    
    ax.plot(global_success_smoothed, label='Global Average', 
        color='black', linewidth=3, linestyle='--')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title(f'Individual Agent Success Rates ({num_agents} Agents)', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    success_path = f"{output_dir}/{num_agents}_individual_success_rates.png"
    plt.savefig(success_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved individual agent success rates to {success_path}")
    
    return


def visualize_convergence_analysis(convergence_data, output_dir):
    """
    Plot convergence episode vs number of agents.
    
    Args:
        convergence_data: Dict with format {num_agents: [conv_ep1, conv_ep2, ...]}
        output_dir: Directory to save the plot
    """
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 7))
    
    agent_numbers = sorted(convergence_data.keys())
    avg_convergence = []
    std_convergence = []
    
    for num_agents in agent_numbers:
        episodes = convergence_data[num_agents]
        # Filter out None values (agents that didn't converge)
        valid_episodes = [ep for ep in episodes if ep is not None]
        
        if valid_episodes:
            avg_convergence.append(np.mean(valid_episodes))
            std_convergence.append(np.std(valid_episodes))
        else:
            avg_convergence.append(None)
            std_convergence.append(None)
    
    # Filter out None values for plotting
    valid_data = [(n, avg, std) for n, avg, std in zip(agent_numbers, avg_convergence, std_convergence) 
                  if avg is not None]
    
    if valid_data:
        valid_agent_nums, valid_avgs, valid_stds = zip(*valid_data)
        
        # Plot with error bars
        ax.errorbar(valid_agent_nums, valid_avgs, yerr=valid_stds, 
                   marker='o', markersize=8, linewidth=2, capsize=5,
                   label='Avg Convergence ± Std')
        
        # Plot individual agent convergence as scatter points
        for num_agents in agent_numbers:
            episodes = convergence_data[num_agents]
            valid_eps = [ep for ep in episodes if ep is not None]
            if valid_eps:
                ax.scatter([num_agents] * len(valid_eps), valid_eps, 
                          alpha=0.4, s=50, color='gray', zorder=1)
    
    ax.set_xlabel('Number of Agents', fontsize=14, fontweight='bold')
    ax.set_ylabel('Convergence Episode', fontsize=14, fontweight='bold')
    ax.set_title('Agent Convergence Speed vs Number of Agents', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Set integer x-axis
    ax.set_xticks(agent_numbers)
    
    convergence_path = f"{output_dir}/convergence_analysis.png"
    plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved convergence analysis to {convergence_path}")

def visualize_multi_agent_comparison(rewards_data, success_data, avg_window, output_dir):
    """
    Compare multiple multi-agent configurations in terms of rewards and success rates.
    
    Args:
        rewards_data: Dict with format {num_agents: [reward_episode1, reward_episode2, ...]}
        success_data: Dict with format {num_agents: [success_episode1, success_episode2, ...]}
        avg_num: Number of episodes to average over for smoothing
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Global Rewards Comparison
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(rewards_data)))
    
    for i, (num_agents, reward_histories) in enumerate(sorted(rewards_data.items())):
        # rewards_data[num_agents] contains list of reward histories
        # For global comparison, we want the first (and likely only) history
        global_rewards = reward_histories[0]
        
        # Apply smoothing if requested
        if avg_window > 1:
            smoothed_rewards = []
            for j in range(0, len(global_rewards), avg_window):
                window_data = global_rewards[j:j+avg_window]
                smoothed_rewards.append(np.mean(window_data))
            plot_data = smoothed_rewards
            x_data = np.arange(0, len(global_rewards), avg_window)
        else:
            plot_data = global_rewards
            x_data = np.arange(len(global_rewards))
        
        # Plot the curve
        label = f'{num_agents} Agent{"s" if num_agents > 1 else ""}'
        plt.plot(x_data, plot_data, color=colors[i], linewidth=2, 
                label=label, alpha=0.8)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Global Reward Comparison Across Different Agent Numbers', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    reward_comparison_path = os.path.join(output_dir, 'global_rewards_comparison.png')
    plt.savefig(reward_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved global rewards comparison to {reward_comparison_path}")

def plot_parallel_vs_fedavg_comparison(parallel_rewards, fedavg_rewards, 
                                       num_agents, avg_window, agent_type, output_dir):
    """
    Create a 2-subplot comparison: Parallel (top) vs FedAvg (bottom)
    
    Args:
        parallel_rewards: Reward history for parallel training
        fedavg_rewards: Reward history for fedavg training
        num_agents: Number of agents
        avg_window: Window size for smoothing
        agent_type: Type of agent
        output_dir: Directory to save the plot
    """
    # Apply smoothing if requested
    def smooth_data(data, window):
        if window > 1:
            smoothed = []
            for i in range(0, len(data), window):
                window_data = data[i:i+window]
                smoothed.append(np.mean(window_data))
            return smoothed
        return data
    
    parallel_smoothed = smooth_data(parallel_rewards, avg_window)
    fedavg_smoothed = smooth_data(fedavg_rewards, avg_window)
    
    # Calculate x-axis
    x_parallel = np.arange(0, len(parallel_rewards), avg_window if avg_window > 1 else 1)[:len(parallel_smoothed)]
    x_fedavg = np.arange(0, len(fedavg_rewards), avg_window if avg_window > 1 else 1)[:len(fedavg_smoothed)]
    
    # Create single figure
    plt.figure(figsize=(12, 8))
    
    # Plot original data with low alpha (background)
    plt.plot(np.arange(len(parallel_rewards)), parallel_rewards, 
             color='#1f77b4', linewidth=1, alpha=0.2, label='_nolegend_')
    plt.plot(np.arange(len(fedavg_rewards)), fedavg_rewards, 
             color='#ff7f0e', linewidth=1, alpha=0.2, label='_nolegend_')
    
    # Plot smoothed data (main curves)
    plt.plot(x_parallel, parallel_smoothed, color='#1f77b4', linewidth=3, 
             label='Parallel Training', marker='o', markersize=3, markevery=10)
    plt.plot(x_fedavg, fedavg_smoothed, color='#ff7f0e', linewidth=3, 
             label='FedAvg Training', marker='s', markersize=3, markevery=10)
    
    # Styling
    plt.xlabel('Episode', fontsize=14, fontweight='bold')
    plt.ylabel('Average Reward', fontsize=14, fontweight='bold')
    plt.title(f'Training Comparison: Parallel vs FedAvg - {num_agents} {agent_type} Agents', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
    

    # Save the plot
    save_path = os.path.join(output_dir, 'reward_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"Saved comparison plot to {save_path}")
    print(f"{'='*60}")

    # save npy file
    # Save numpy arrays for easy loading
    np.save(os.path.join(output_dir, f'parallel_rewards_{num_agents}agents.npy'), 
            np.array(parallel_rewards))
    np.save(os.path.join(output_dir, f'fedavg_rewards_{num_agents}agents.npy'), 
            np.array(fedavg_rewards))
    


def plot_agent_num_ablation(v_data, avg_window=1, output_dir=None, 
                            agent_type='hd_sarsa', env="CartPole", hdim=10000):
    """Plot MSE loss comparison for different number of agents on a single figure."""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate the shared y-axis range and x-axis range
    all_losses = [mse for losses in v_data.values() for mse in np.array(losses).flatten()]
    y_range = (min(all_losses), max(all_losses) * 1.1)
    x_range = max(len(np.array(losses).flatten()) for losses in v_data.values())

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Create a single figure with all curves on the same plot
    plt.figure(figsize=(10, 6))

    # Plot all agent configurations on the same figure
    for idx, num_agents in enumerate(sorted(v_data.keys())):
        losses = np.array(v_data[num_agents]).flatten()
        
        # Add smoothing if requested
        if avg_window > 1:
            smoothed_losses = []
            for i in range(0, len(losses) - avg_window + 1, avg_window):
                window_data = losses[i:i+avg_window]
                smoothed_losses.append(np.mean(window_data))
            losses = smoothed_losses
        
        # Calculate x-axis data (Episode/Round numbers)
        x_data = np.arange(len(losses)) * avg_window if avg_window > 1 else np.arange(len(losses))

        # Plot MSE losses on same plot
        label = f'N = {num_agents} Agent{"s" if num_agents > 1 else ""}'
        plt.plot(x_data, losses, color=colors[idx % len(colors)], linewidth=2, label=label)
    
    plt.ylim(y_range)
    plt.title(f'Agent Number Ablation - {agent_type} (D={hdim}) in {env}', 
              fontsize=14, fontweight='bold')
    # plt.yscale('log')
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('V-value (log scale)', fontsize=12)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'agent_num_ablation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved agent number ablation plot to {save_path}")

def plot_hdim_comparison(v_loss_data, avg_window=1, output_dir=None, 
                         agent_type='hd_sarsa', env="CartPole", num_agents=10):
    """Plot V-value comparison for different hyperdimension D values"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    D_values = sorted(v_loss_data.keys())
    
    for idx, D in enumerate(D_values):
        losses = np.array(v_loss_data[D]).flatten()
        base_color = colors[idx % len(colors)]

        x_raw = np.arange(len(losses))
        ax1.plot(x_raw, losses, color=base_color, linewidth=1, alpha=0.2, label='_nolegend_')

        if avg_window > 1:
            smoothed = [np.mean(losses[i:i+avg_window])
                       for i in range(0, len(losses)-avg_window+1, avg_window)]
            x_data = np.arange(len(smoothed)) * avg_window
        else:
            smoothed = losses
            x_data = x_raw

        ax1.plot(x_data, smoothed, color=base_color,
                linewidth=2.5, label=f'D = {D}')
    
    # ax1.set_yscale('log')
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('V-loss', fontsize=12)
    ax1.set_title(f'V-loss Convergence - {num_agents} {agent_type} Agents in {env}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'hdim_ablation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved D ablation plot to {save_path}")
    
    