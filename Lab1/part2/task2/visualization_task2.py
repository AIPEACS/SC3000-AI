"""
Task 2: Visualization and Plotting Functions
=============================================
This module contains all plotting, visualization, and JSON export functions
for Task 2 with training history tracking.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import scene_map as map0
from agent_task2 import GAMMA, ACTIONS, reward_calc


# ==================== DIRECTORY SETUP ====================

VIS_DIR = os.path.join(os.path.dirname(__file__), "visualization")
os.makedirs(VIS_DIR, exist_ok=True)


# ==================== HELPER FUNCTIONS ====================

def action_to_symbol(action_idx):
    """
    Convert action index to visual symbol.
    
    Args:
        action_idx: Index of action (0=up, 1=down, 2=left, 3=right)
        
    Returns:
        str: Unicode arrow symbol representing the action
    """
    symbols = {0: '↑', 1: '↓', 2: '←', 3: '→', -1: 'G'}
    return symbols.get(action_idx, '?')


# ==================== PRINTING FUNCTIONS ====================

def print_policy(policy_det, title="Learned Policy"):
    """
    Display the deterministic policy as a grid with action symbols.
    
    Args:
        policy_det: 5x5 array of action indices
        title: Title for display
    """
    print(f"\n{title}:")
    print("-" * 50)
    # Print in visual order: top to bottom (y descending), left to right (x ascending)
    for y in range(4, -1, -1):
        row = []
        for x in range(5):
            if (x, y) in map0.road_blocking:
                row.append(" [X] ")
            else:
                row.append(f"  {action_to_symbol(policy_det[x, y])}  ")
        print(" ".join(row))
    print()


def action_tensor_to_markdown(policy_det, title="Learned Policy"):
    """
    Convert action tensor (5x5 policy matrix) to markdown table.
    
    Args:
        policy_det: 5x5 array of action indices
        title: Title for the markdown table
        
    Returns:
        str: Markdown formatted 5x5 table
    """
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', -1: 'GOAL'}
    
    markdown = f"## {title}\n\n"
    markdown += "| Y\\X | 0 | 1 | 2 | 3 | 4 |\n"
    markdown += "|-----|---|---|---|---|----|\n"
    
    # Print in visual order: top to bottom (y descending), left to right (x ascending)
    for y in range(4, -1, -1):
        row = [f"| {y} |"]
        for x in range(5):
            if (x, y) in map0.road_blocking:
                row.append(" OBS |")
            else:
                action_idx = int(policy_det[x, y])
                action_name = action_names.get(action_idx, '?')
                row.append(f" {action_name} |")
        markdown += "".join(row) + "\n"
    
    return markdown


# ==================== PLOTTING FUNCTIONS ====================

def plot_training_history(training_history, algorithm_name="MonteCarlo"):
    """
    Create training history plots showing convergence.
    
    Args:
        training_history: List of dicts with 'episode', 'reward', 'visited_pairs'
        algorithm_name: Name of the algorithm for file naming
    """
    if not training_history:
        print("No training history to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Task 2: {algorithm_name} - Training History', 
                fontsize=16, fontweight='bold')
    
    episodes = [h['episode'] for h in training_history]
    rewards = [h['reward'] for h in training_history]
    visited_pairs = [h['visited_pairs'] for h in training_history]
    
    # Plot 1: Episode Reward (Line Plot)
    ax = axes[0, 0]
    ax.plot(episodes, rewards, 'b-', linewidth=1.5, alpha=0.7)
    # Add moving average
    window = min(50, len(rewards) // 2)
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(rewards)+1), moving_avg, 'r-', linewidth=2.5, label=f'Moving Avg ({window})')
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Reward', fontsize=11, fontweight='bold')
    ax.set_title('Episode Rewards Over Training', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 2: Reward Distribution (Histogram)
    ax = axes[0, 1]
    ax.hist(rewards, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax.set_xlabel('Total Reward', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    # Plot 3: Visited State-Action Pairs
    ax = axes[1, 0]
    ax.plot(episodes, visited_pairs, 'g-o', linewidth=1.5, markersize=3, alpha=0.7)
    ax.fill_between(episodes, visited_pairs, alpha=0.15, color='green')
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Unique State-Action Pairs', fontsize=11, fontweight='bold')
    ax.set_title('Exploration: Visited Pairs per Episode', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative Reward
    ax = axes[1, 1]
    cumulative_rewards = np.cumsum(rewards)
    ax.plot(episodes, cumulative_rewards, 'purple', linewidth=2)
    ax.fill_between(episodes, cumulative_rewards, alpha=0.2, color='purple')
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Training Reward', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(VIS_DIR, f'{algorithm_name}_training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_path}")
    plt.close()


# ==================== JSON EXPORT FUNCTIONS ====================

def policy_to_action_tensor(policy_det, title="Action Tensor"):
    """
    Convert deterministic policy to action tensor (5x5 matrix of action indices).
    
    Args:
        policy_det: 5x5 array of action indices
        title: Title for the tensor
        
    Returns:
        dict: JSON-serializable action tensor representation
    """
    action_tensor = {
        "title": title,
        "description": "5x5 matrix of action indices for each state",
        "action_indices": {
            "0": "UP",
            "1": "DOWN",
            "2": "LEFT",
            "3": "RIGHT",
            "-1": "GOAL"
        },
        "action_tensor": []
    }
    
    # Convert 5x5 array to list of lists
    for x in range(5):
        row = []
        for y in range(5):
            if (x, y) == map0.end_point:
                row.append(-1)  # Mark goal position
            else:
                row.append(int(policy_det[x, y]))
        action_tensor["action_tensor"].append(row)
    
    return action_tensor


def save_action_tensor_json(policy_det, algorithm_name="Learned_Policy"):
    """
    Save the action tensor to a JSON file.
    
    Args:
        policy_det: Deterministic policy (5x5 array)
        algorithm_name: Name for the output file
        
    Returns:
        str: Path to saved JSON file
    """
    tensor_json = policy_to_action_tensor(policy_det, f"{algorithm_name} - Action Tensor")
    
    filename = os.path.join(VIS_DIR, f"{algorithm_name}_action_tensor.json")
    with open(filename, 'w') as f:
        json.dump(tensor_json, f, indent=2)
    
    print(f"✓ Saved action tensor to: {filename}")
    
    # Also print to console
    print(f"\n{algorithm_name} Action Tensor (JSON):")
    print("-" * 60)
    print(json.dumps(tensor_json, indent=2))
    
    return filename


def save_q_values(q_state_action, algorithm_name="Q_values"):
    """
    Save Q-values to a JSON file.
    
    Args:
        q_state_action: 5x5x4 array of Q-values
        algorithm_name: Name for the output file
        
    Returns:
        str: Path to saved JSON file
    """
    q_dict = {
        "title": f"{algorithm_name}",
        "description": "Q-values for state-action pairs",
        "shape": [5, 5, 4],
        "action_map": {"0": "UP", "1": "DOWN", "2": "LEFT", "3": "RIGHT"},
        "q_values": {}
    }
    
    for x in range(5):
        for y in range(5):
            state_key = f"({x},{y})"
            q_dict["q_values"][state_key] = [
                float(q_state_action[x, y, 0]),  # UP
                float(q_state_action[x, y, 1]),  # DOWN
                float(q_state_action[x, y, 2]),  # LEFT
                float(q_state_action[x, y, 3])   # RIGHT
            ]
    
    filename = os.path.join(VIS_DIR, f"{algorithm_name}.json")
    with open(filename, 'w') as f:
        json.dump(q_dict, f, indent=2)
    
    print(f"✓ Saved Q-values to: {filename}")
    
    return filename


def save_policy_json(policy_det, algorithm_name="Learned_Policy"):
    """
    Save the policy to a JSON file.
    
    Args:
        policy_det: Deterministic policy (5x5 array)
        algorithm_name: Name for the output file
        
    Returns:
        str: Path to saved JSON file
    """
    policy_json = {
        "title": f"{algorithm_name} - Policy",
        "policy_map": {
            "0": "UP",
            "1": "DOWN",
            "2": "LEFT",
            "3": "RIGHT"
        },
        "policy": {}
    }
    
    for x in range(5):
        for y in range(5):
            if (x, y) == map0.end_point:
                action_str = "GOAL"
            else:
                action_idx = int(policy_det[x, y])
                action_str = ["UP", "DOWN", "LEFT", "RIGHT"][action_idx]
            
            state_key = f"({x},{y})"
            policy_json["policy"][state_key] = action_str
    
    filename = os.path.join(VIS_DIR, f"{algorithm_name}_policy.json")
    with open(filename, 'w') as f:
        json.dump(policy_json, f, indent=2)
    
    print(f"✓ Saved policy to: {filename}")
    
    return filename
