"""
Task 3: Q-Learning Visualization and Export Functions
======================================================
Provides visualization and export utilities for Q-learning results,
including training history plots, policy export, and Q-value analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os


def action_tensor_to_markdown(policy_det, title=""):
    """
    Convert a 5×5 action policy matrix to a markdown table.
    
    Args:
        policy_det: 5×5 array where each element is action index (0-3)
        title: Optional title to include
        
    Returns:
        str: Markdown formatted table
    """
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', -1: 'GOAL', 4: 'OBS'}
    
    lines = []
    if title:
        lines.append(f"### {title}\n")
    
    # Header row
    lines.append("| Y\\X | 0 | 1 | 2 | 3 | 4 |")
    lines.append("|-----|---|---|---|---|---|")
    
    # Data rows (reversed for visual clarity)
    for y in range(4, -1, -1):
        row_data = []
        for x in range(5):
            action_idx = int(policy_det[(x, y)])
            if action_idx == -1:
                cell = "GOAL"
            elif action_idx == 4:
                cell = "OBS"
            else:
                cell = action_names.get(action_idx, f"A{action_idx}")
            row_data.append(cell)
        
        line = f"| {y} | " + " | ".join(row_data) + " |"
        lines.append(line)
    
    return "\n".join(lines) + "\n"


def plot_training_history(training_history, algorithm_name="Q-Learning"):
    """
    Create a 4-subplot visualization of training convergence.
    
    Subplots:
    [0,0] Episode rewards with 100-episode moving average
    [0,1] Reward distribution histogram with mean line
    [1,0] Unique state-action pairs explored per episode
    [1,1] Cumulative training reward over all episodes
    
    Args:
        training_history: Dict with keys:
            - 'episode_rewards': List of rewards per episode
            - 'episode_visited_pairs': List of unique (state, action) counts
        algorithm_name: Algorithm name for title
        
    Returns:
        None (saves plot)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{algorithm_name} Training Convergence (5000 Episodes)', 
                 fontsize=16, fontweight='bold')
    
    # Extract data
    episode_rewards = training_history['episode_rewards']
    episode_visited_pairs = training_history['episode_visited_pairs']
    
    num_episodes = len(episode_rewards)
    episodes = np.arange(num_episodes)
    
    # [0,0] Episode rewards with moving average
    axes[0, 0].plot(episodes, episode_rewards, 'b-', alpha=0.5, linewidth=1, label='Episode Reward')
    
    # Moving average (100-episode window)
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        moving_avg_episodes = np.arange(window-1, num_episodes)
        axes[0, 0].plot(moving_avg_episodes, moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title(f'Episode Rewards (Mean: {np.mean(episode_rewards):.2f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # [0,1] Reward distribution histogram
    axes[0, 1].hist(episode_rewards, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    mean_reward = np.mean(episode_rewards)
    axes[0, 1].axvline(mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
    axes[0, 1].set_xlabel('Reward Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Reward Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # [1,0] Unique state-action pairs explored
    axes[1, 0].fill_between(episodes, episode_visited_pairs, alpha=0.5, color='green', label='Visited Pairs')
    axes[1, 0].plot(episodes, episode_visited_pairs, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Unique State-Action Pairs per Episode (Max: {max(episode_visited_pairs)})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # [1,1] Cumulative reward
    cumulative_reward = np.cumsum(episode_rewards)
    axes[1, 1].plot(episodes, cumulative_reward, 'purple', linewidth=2, label='Cumulative Reward')
    axes[1, 1].fill_between(episodes, cumulative_reward, alpha=0.3, color='purple')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].set_title(f'Cumulative Training Reward (Total: {cumulative_reward[-1]:.0f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_action_tensor_json(Q_values, policy_array, algorithm_name="QLearning"):
    """
    Export the policy to JSON format.
    
    Args:
        Q_values: Dict of Q-values (not used in export, for consistency)
        policy_array: dict {(x, y): int} of action indices
        algorithm_name: Algorithm name for file naming
        
    Returns:
        str: Path to saved JSON file
    """
    # Convert dict to 2D list in [y][x] format (standard row-major)
    policy_list = [[int(policy_array.get((x, y), 0)) for x in range(5)] for y in range(5)]
    
    output_dir = "./visualization/"
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"{algorithm_name}_Learned_action_tensor.json")
    with open(filepath, 'w') as f:
        json.dump(policy_list, f, indent=2)
    
    return filepath


def save_q_values(Q_values, algorithm_name="QLearning"):
    """
    Export Q-values to JSON format.
    
    Args:
        Q_values: Dict with keys ((x, y), action) containing Q-values
        algorithm_name: Algorithm name for file naming
        
    Returns:
        str: Path to saved JSON file
    """
    output_dir = "./visualization/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to serializable format (state as string key)
    q_dict = {}
    for (state, action), value in Q_values.items():
        state_key = f"({state[0]},{state[1]})"
        if state_key not in q_dict:
            q_dict[state_key] = {}
        q_dict[state_key][action] = float(value)
    
    filepath = os.path.join(output_dir, f"{algorithm_name}_Q_values.json")
    with open(filepath, 'w') as f:
        json.dump(q_dict, f, indent=2)
    
    return filepath


def print_policy(policy_matrix):
    """
    Print policy to console in readable format.
    
    Args:
        policy_matrix: 5×5 array of action indices
    """
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', -1: 'GOAL', 4: 'OBS'}
    
    print("\nPolicy Table:")
    print("=" * 40)
    for y in range(4, -1, -1):
        row = []
        for x in range(5):
            action_idx = int(policy_matrix[(x, y)])
            if action_idx == -1:
                cell = "GOAL"
            elif action_idx == 4:
                cell = "OBS"
            else:
                cell = action_names.get(action_idx, f"A{action_idx}")
            row.append(f"{cell:5}")
        print(f"Y={y}: " + " ".join(row))
    print("       " + "".join([f"X={x:1}    " for x in range(5)]))
    print("=" * 40)


def plot_q_value_history(q_snapshots):
    """
    Plot Q-value evolution over training for all 25 states.

    Layout: 5×5 grid of subplots matching grid coordinates
    (row 0 = y=4 at top, row 4 = y=0 at bottom; col = x).
    Each subplot shows 4 lines for actions u/d/l/r.
    Goal state and obstacle states are annotated instead of plotted.

    Args:
        q_snapshots: list of (episode, snapshot_dict) recorded every 100 episodes
            snapshot_dict: {(x,y): {'u': v, 'd': v, 'l': v, 'r': v}}

    Returns:
        matplotlib.figure.Figure
    """
    import scene_map as map0

    ACTIONS = ['u', 'd', 'l', 'r']
    ACTION_LABELS = {'u': 'Up', 'd': 'Down', 'l': 'Left', 'r': 'Right'}
    ACTION_COLORS = {'u': '#1f77b4', 'd': '#ff7f0e', 'l': '#2ca02c', 'r': '#d62728'}

    episodes = [ep for ep, _ in q_snapshots]

    fig, axes = plt.subplots(5, 5, figsize=(22, 18), sharex=False)
    fig.suptitle(
        f'Q-Value Update History — sampled every 100 episodes\n'
        f'({len(episodes)} snapshots, {episodes[-1]} total episodes)',
        fontsize=14, fontweight='bold'
    )

    for row in range(5):
        y = 4 - row           # row 0 → y=4, row 4 → y=0
        for col in range(5):
            x = col           # col 0 → x=0
            ax = axes[row, col]
            ax.set_title(f'({x},{y})', fontsize=8, pad=2)

            if (x, y) == map0.end_point:
                ax.text(0.5, 0.5, 'GOAL', ha='center', va='center',
                        fontsize=14, fontweight='bold', color='green',
                        transform=ax.transAxes)
                ax.set_facecolor('#e8f5e9')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            if (x, y) in map0.road_blocking:
                ax.text(0.5, 0.5, 'OBS', ha='center', va='center',
                        fontsize=14, fontweight='bold', color='gray',
                        transform=ax.transAxes)
                ax.set_facecolor('#f5f5f5')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Build per-action series from snapshots
            for action in ACTIONS:
                q_series = [snap[(x, y)][action] for _, snap in q_snapshots]
                ax.plot(episodes, q_series,
                        color=ACTION_COLORS[action],
                        linewidth=1.0,
                        label=ACTION_LABELS[action])

            ax.axhline(0, color='black', linewidth=0.4, linestyle='--', alpha=0.4)
            ax.set_xlim(episodes[0], episodes[-1])
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)

    # Shared axis labels via figure text
    fig.text(0.5, 0.01, 'Episode', ha='center', fontsize=11)
    fig.text(0.01, 0.5, 'Q-Value', va='center', rotation='vertical', fontsize=11)

    # Single legend for the whole figure
    handles = [
        plt.Line2D([0], [0], color=ACTION_COLORS[a], linewidth=2, label=ACTION_LABELS[a])
        for a in ACTIONS
    ]
    fig.legend(handles=handles, loc='lower right', fontsize=10,
               title='Action', title_fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.99, 0.02))

    plt.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])
    return fig
