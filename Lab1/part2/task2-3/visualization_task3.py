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



