"""
Task 1: Visualization and Plotting Functions
==============================================
This module contains all plotting, visualization, and JSON export functions
for Task 1 visualization outputs.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import scene_map as map0
from agent import GAMMA, ACTIONS, reward_calc


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

def print_value_function(V, title="Value Function"):
    """
    Display the value function as a grid.
    Values are shown with 2 decimal places.
    Grid is displayed with (0,0) at bottom-left and (4,4) at top-right.
    
    Args:
        V: 5x5 value function array
        title: Title for display
    """
    print(f"\n{title}:")
    print("-" * 50)
    # Print in visual order: top to bottom (y descending), left to right (x ascending)
    for y in range(4, -1, -1):
        row = []
        for x in range(5):
            if (x, y) in map0.road_blocking:
                row.append("  [X]  ")
            else:
                row.append(f"{V[(x, y)]:7.2f}")
        print(" ".join(row))
    print()


def print_policy(policy_det, title="Optimal Policy"):
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
                row.append(f"  {action_to_symbol(policy_det[(x, y)])}  ")
        print(" ".join(row))
    print()


def action_tensor_to_markdown(policy_det, title="Optimal Policy"):
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
                action_idx = int(policy_det[(x, y)])
                action_name = action_names.get(action_idx, '?')
                row.append(f" {action_name} |")
        markdown += "".join(row) + "\n"
    
    return markdown


# ==================== PLOTTING FUNCTIONS ====================
def plot_and_save_results(epoch_metrics_vi, epoch_metrics_pi):
    """
    Create training epoch analysis plots showing convergence.
    
    Args:
        epoch_metrics_vi: List of max value changes per epoch for VI
        epoch_metrics_pi: List of max value changes per epoch for PI
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Task 1: Planning Algorithms - Training Convergence Analysis', 
                fontsize=16, fontweight='bold')
    
    epochs_vi = list(range(1, len(epoch_metrics_vi) + 1))
    epochs_pi = list(range(1, len(epoch_metrics_pi) + 1))
    
    # Plot 1: Bellman Residual per Epoch (Line Plot)
    ax = axes[0, 0]
    ax.semilogy(epochs_vi, epoch_metrics_vi, 'b-o', linewidth=2.5, markersize=7, label='Value Iteration', alpha=0.8)
    ax.semilogy(epochs_pi, epoch_metrics_pi, 'r-s', linewidth=2.5, markersize=7, label='Policy Iteration', alpha=0.8)
    ax.set_xlabel('Training Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Max Value Change (Bellman Residual)', fontsize=11, fontweight='bold')
    ax.set_title('Convergence: Max Value Change per Epoch (Log Scale)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='upper right')
    
    # Plot 2: Bellman Residual per Epoch (Bar Comparison)
    ax = axes[0, 1]
    width = 0.35
    x_pos_vi = np.array(epochs_vi) - width/2
    x_pos_pi = np.array(epochs_pi) + width/2
    ax.bar(x_pos_vi, epoch_metrics_vi, width, label='Value Iteration', alpha=0.8, color='steelblue', edgecolor='black')
    ax.bar(x_pos_pi, epoch_metrics_pi, width, label='Policy Iteration', alpha=0.8, color='salmon', edgecolor='black')
    ax.set_xlabel('Training Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Max Value Change', fontsize=11, fontweight='bold')
    ax.set_title('Bellman Residual: Algorithm Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    # Plot 3: Value Iteration - Training Progress
    ax = axes[1, 0]
    ax.semilogy(epochs_vi, epoch_metrics_vi, 'b-o', linewidth=2, markersize=6, alpha=0.8)
    ax.fill_between(epochs_vi, epoch_metrics_vi, alpha=0.2, color='blue')
    ax.set_xlabel('Training Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Max Value Change (Log Scale)', fontsize=11, fontweight='bold')
    ax.set_title(f'Value Iteration - Convergence ({len(epochs_vi)} epochs)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add convergence stats
    final_change_vi = epoch_metrics_vi[-1] if epoch_metrics_vi else 0
    ax.text(0.98, 0.97, f'Final Change: {final_change_vi:.2e}', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 4: Policy Iteration - Training Progress
    ax = axes[1, 1]
    ax.semilogy(epochs_pi, epoch_metrics_pi, 'r-s', linewidth=2, markersize=6, alpha=0.8)
    ax.fill_between(epochs_pi, epoch_metrics_pi, alpha=0.2, color='red')
    ax.set_xlabel('Training Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Max Value Change (Log Scale)', fontsize=11, fontweight='bold')
    ax.set_title(f'Policy Iteration - Convergence ({len(epochs_pi)} epochs)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add convergence stats
    final_change_pi = epoch_metrics_pi[-1] if epoch_metrics_pi else 0
    ax.text(0.98, 0.97, f'Final Change: {final_change_pi:.2e}', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plot_path = os.path.join(VIS_DIR, 'task1_visualization.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_path}")
    plt.close()


# ==================== JSON EXPORT FUNCTIONS ====================

def policy_to_json_matrix(policy_det):
    """
    Convert the deterministic policy to a JSON-serializable matrix format.
    Includes both action names and symbols for clarity.
    
    Args:
        policy_det: 5x5 array of action indices
        
    Returns:
        dict: JSON-serializable policy matrix with all details
    """
    policy_dict = {
        "policy_type": "deterministic",
        "action_map": {
            "0": "up",
            "1": "down",
            "2": "left",
            "3": "right"
        },
        "grid": []
    }
    
    # Convert to visual grid format (y descending)
    for y in range(4, -1, -1):
        row = []
        for x in range(5):
            if (x, y) in map0.road_blocking:
                row.append({
                    "position": [x, y],
                    "state": "obstacle",
                    "action": None
                })
            elif (x, y) == map0.end_point:
                row.append({
                    "position": [x, y],
                    "state": "goal",
                    "action": None
                })
            else:
                action_idx = int(policy_det[(x, y)])
                action_name = ACTIONS[action_idx]
                action_symbol = action_to_symbol(action_idx)
                row.append({
                    "position": [x, y],
                    "state": "regular",
                    "action": action_name,
                    "action_code": action_idx,
                    "action_symbol": action_symbol
                })
        policy_dict["grid"].append(row)
    
    return policy_dict


def policy_to_json_simple(policy_det, title):
    """
    Convert deterministic policy to a simple JSON format.
    Maps state coordinates to action strings.
    
    Args:
        policy_det: 5x5 array of action indices
        title: Title for the policy
        
    Returns:
        dict: JSON-serializable policy representation
    """
    policy_json = {
        "title": title,
        "actions_map": {
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
                action_idx = int(policy_det[(x, y)])
                action_str = ["UP", "DOWN", "LEFT", "RIGHT"][action_idx]
            
            state_key = f"({x},{y})"
            policy_json["policy"][state_key] = action_str
    
    return policy_json


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
        "description": "5x5 matrix of optimal action indices for each state",
        "action_indices": {
            "0": "UP",
            "1": "DOWN",
            "2": "LEFT",
            "3": "RIGHT",
            "-1": "GOAL"
        },
        "action_tensor": []
    }
    
    # Convert to list of lists in [y][x] format (standard row-major)
    for y in range(5):
        row = []
        for x in range(5):
            row.append(int(policy_det[(x, y)]))
        action_tensor["action_tensor"].append(row)
    
    return action_tensor


def save_action_tensor_json(policy_det, algorithm_name="Optimal_Policy"):
    """
    Save the action tensor (5x5 policy matrix) to a JSON file.
    
    Args:
        policy_det: Deterministic policy (5x5 array)
        algorithm_name: Name for the output file
        
    Returns:
        str: Path to saved JSON file
    """
    tensor_json = policy_to_action_tensor(policy_det, f"{algorithm_name} - Action Tensor")
    
    # Save to file in VIS_DIR
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
        q_state_action: 5x5x4 numpy array of Q-values [x, y, action]
        algorithm_name: Name for the output file

    Returns:
        str: Path to saved JSON file
    """
    q_dict = {
        "title": algorithm_name,
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
                float(q_state_action[x, y, 3]),  # RIGHT
            ]

    filename = os.path.join(VIS_DIR, f"{algorithm_name}.json")
    with open(filename, 'w') as f:
        json.dump(q_dict, f, indent=2)

    print(f"✓ Saved Q-values to: {filename}")
    return filename


def save_policy_json(policy_det, algorithm_name="Optimal_Policy", format_type="matrix"):
    """
    Save the policy to a JSON file in the visualization directory.
    
    Args:
        policy_det: Deterministic policy (5x5 array)
        algorithm_name: Name for the output file
        format_type: Either "matrix" (detailed) or "simple" (compact)
        
    Returns:
        str: Path to saved JSON file
    """
    # Choose format
    if format_type == "simple":
        policy_json = policy_to_json_simple(policy_det, f"{algorithm_name} - Optimal Policy")
    else:
        policy_json = policy_to_json_matrix(policy_det)
    
    # Save to file in VIS_DIR
    filename = os.path.join(VIS_DIR, f"{algorithm_name}_policy.json")
    with open(filename, 'w') as f:
        json.dump(policy_json, f, indent=2)
    
    print(f"✓ Saved policy to: {filename}")
    
    # Also print to console
    print(f"\n{algorithm_name} Policy (JSON):")
    print("-" * 60)
    print(json.dumps(policy_json, indent=2))
    
    return filename
