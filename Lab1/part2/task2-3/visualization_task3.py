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
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scene_map as map0


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
    lines.append("| X\\Y | 0 | 1 | 2 | 3 | 4 |")
    lines.append("|-----|---|---|---|---|---|")
    
    # Data rows: x descending (top = x=4), y ascending (left = y=0)
    for x in range(4, -1, -1):
        row_data = []
        for y in range(5):
            action_idx = int(policy_det[(x, y)])
            if action_idx == -1:
                cell = "GOAL"
            elif action_idx == 4:
                cell = "OBS"
            else:
                cell = action_names.get(action_idx, f"A{action_idx}")
            row_data.append(cell)
        
        line = f"| {x} | " + " | ".join(row_data) + " |"
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
    # Convert dict to 2D list in [x][y] format matching the axis convention
    policy_list = [[int(policy_array.get((x, y), 0)) for y in range(5)] for x in range(5)]
    
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
    for x in range(4, -1, -1):
        row = []
        for y in range(5):
            action_idx = int(policy_matrix[(x, y)])
            if action_idx == -1:
                cell = "GOAL"
            elif action_idx == 4:
                cell = "OBS"
            else:
                cell = action_names.get(action_idx, f"A{action_idx}")
            row.append(f"{cell:5}")
        print(f"X={x}: " + " ".join(row))
    print("       " + "".join([f"Y={y:1}    " for y in range(5)]))
    print("=" * 40)


# ==================== Q-VALUE HISTORY PLOT ====================

DEBUG_VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_visualization")
os.makedirs(DEBUG_VIS_DIR, exist_ok=True)


def plot_q_value_history(q_snapshots):
    """
    Plot Q-value evolution over training for all 25 states.

    Layout: 5×5 grid of subplots matching grid coordinates
    (row 0 = x=4 at top, row 4 = x=0 at bottom; col = y).
    Each subplot shows 4 lines for actions u/d/l/r.
    Goal state and obstacle states are annotated instead of plotted.

    Args:
        q_snapshots: list of (episode, snapshot_dict) recorded every 100 episodes
            snapshot_dict: {(x,y): {'u': v, 'd': v, 'l': v, 'r': v}}

    Returns:
        matplotlib.figure.Figure
    """
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
        x = 4 - row          # row 0 → x=4, row 4 → x=0
        for col in range(5):
            y = col           # col 0 → y=0
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

    fig.text(0.5, 0.01, 'Episode', ha='center', fontsize=11)
    fig.text(0.01, 0.5, 'Q-Value', va='center', rotation='vertical', fontsize=11)

    handles = [
        plt.Line2D([0], [0], color=ACTION_COLORS[a], linewidth=2, label=ACTION_LABELS[a])
        for a in ACTIONS
    ]
    fig.legend(handles=handles, loc='lower right', fontsize=10,
               title='Action', title_fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.99, 0.02))

    plt.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])
    return fig

