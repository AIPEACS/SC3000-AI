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
                row.append(f"{V[x, y]:7.2f}")
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
                row.append(f"  {action_to_symbol(policy_det[x, y])}  ")
        print(" ".join(row))
    print()


# ==================== PLOTTING FUNCTIONS ====================

def visualize_trajectory_and_rewards(path, name="Policy"):
    """
    Create a comprehensive visualization showing:
    1. Agent's trajectory on the grid (left)
    2. Cumulative rewards per step (right)
    
    Args:
        path: List of (x, y) positions visited
        name: Name of the policy for title
        
    Returns:
        matplotlib.figure.Figure: Generated figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # -------- Left plot: Grid with trajectory --------
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title(f'{name}\nAgent Trajectory on Grid')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    
    # Draw roadblocks
    for obstacle in map0.road_blocking:
        rect = patches.Rectangle((obstacle[0] - 0.4, obstacle[1] - 0.4), 0.8, 0.8,
                                 linewidth=2, edgecolor='red', facecolor='red', alpha=0.5)
        ax1.add_patch(rect)
        ax1.text(obstacle[0], obstacle[1], 'X', ha='center', va='center', 
                fontsize=12, fontweight='bold')
    
    # Draw start and goal
    start_circle = patches.Circle(map0.start_point, 0.15, color='green', zorder=5)
    ax1.add_patch(start_circle)
    ax1.text(map0.start_point[0] - 0.35, map0.start_point[1] - 0.35, 'S', 
            fontsize=10, fontweight='bold')
    
    goal_circle = patches.Circle(map0.end_point, 0.15, color='blue', zorder=5)
    ax1.add_patch(goal_circle)
    ax1.text(map0.end_point[0] + 0.25, map0.end_point[1] + 0.25, 'G', 
            fontsize=10, fontweight='bold')
    
    # Draw trajectory
    if len(path) > 1:
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.6, label='Path')
        ax1.scatter(x_coords[:-1], y_coords[:-1], color='cyan', s=50, alpha=0.7, zorder=4)
        ax1.scatter([x_coords[-1]], [y_coords[-1]], color='darkblue', s=100, marker='*', zorder=6)
    
    ax1.legend()
    
    # -------- Right plot: Cumulative rewards per step --------
    cumulative_rewards = []
    total = 0
    for position in path:
        r = reward_calc(position[0], position[1])
        total += r
        cumulative_rewards.append(total)
    
    steps = list(range(len(cumulative_rewards)))
    ax2.plot(steps, cumulative_rewards, 'go-', linewidth=2, markersize=8, label='Cumulative Reward')
    ax2.fill_between(steps, cumulative_rewards, alpha=0.3, color='green')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title(f'{name}\nCumulative Reward per Step (γ={GAMMA})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add annotations for key points
    if cumulative_rewards:
        ax2.text(len(cumulative_rewards) - 1, cumulative_rewards[-1], 
                f'  Final: {cumulative_rewards[-1]:.2f}', 
                fontsize=10, va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_and_save_results(path_vi, rewards_vi, path_pi, rewards_pi):
    """
    Create comprehensive comparison plots and save to file.
    Includes 4 subplots:
    1. Cumulative reward comparison
    2. Reward per step comparison
    3. Value Iteration trajectory
    4. Policy Iteration trajectory
    
    Args:
        path_vi: Path taken by Value Iteration policy
        rewards_vi: Rewards per step for VI policy
        path_pi: Path taken by Policy Iteration policy
        rewards_pi: Rewards per step for PI policy
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Task 1: Planning Algorithms - Value Iteration vs Policy Iteration', 
                fontsize=16, fontweight='bold')
    
    # Plot 1: Cumulative Reward over Steps
    ax = axes[0, 0]
    steps_vi = range(len(rewards_vi))
    steps_pi = range(len(rewards_pi))
    cumulative_vi = np.cumsum(rewards_vi)
    cumulative_pi = np.cumsum(rewards_pi)
    
    ax.plot(steps_vi, cumulative_vi, 'b-o', linewidth=2, markersize=6, label='Value Iteration')
    ax.plot(steps_pi, cumulative_pi, 'r-s', linewidth=2, markersize=6, label='Policy Iteration')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Cumulative Reward', fontsize=11)
    ax.set_title('Cumulative Reward per Step', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 2: Reward per Step
    ax = axes[0, 1]
    ax.bar(np.array(steps_vi) - 0.2, rewards_vi, width=0.4, label='Value Iteration', alpha=0.8)
    ax.bar(np.array(steps_pi) + 0.2, rewards_pi, width=0.4, label='Policy Iteration', alpha=0.8)
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title('Reward per Step', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    # Plot 3: Agent Path - Value Iteration
    ax = axes[1, 0]
    path_x_vi = [p[0] for p in path_vi]
    path_y_vi = [p[1] for p in path_vi]
    
    # Draw grid
    for i in range(5):
        ax.axhline(y=i-0.5, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(x=i-0.5, color='gray', linewidth=0.5, alpha=0.3)
    
    # Draw obstacles
    for idx, obs in enumerate(map0.road_blocking):
        ax.plot(obs[0], obs[1], 's', color='black', markersize=15, 
               label='Obstacle' if idx == 0 else '')
    
    # Draw start and goal
    ax.plot(map0.start_point[0], map0.start_point[1], 'go', markersize=12, label='Start')
    ax.plot(map0.end_point[0], map0.end_point[1], 'r*', markersize=20, label='Goal')
    
    # Draw path
    ax.plot(path_x_vi, path_y_vi, 'b-', linewidth=2, alpha=0.6)
    ax.plot(path_x_vi, path_y_vi, 'bo', markersize=5, alpha=0.6)
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel('X Coordinate', fontsize=11)
    ax.set_ylabel('Y Coordinate', fontsize=11)
    ax.set_title(f'Value Iteration Path ({len(path_vi)-1} steps)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    
    # Plot 4: Agent Path - Policy Iteration
    ax = axes[1, 1]
    path_x_pi = [p[0] for p in path_pi]
    path_y_pi = [p[1] for p in path_pi]
    
    # Draw grid
    for i in range(5):
        ax.axhline(y=i-0.5, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(x=i-0.5, color='gray', linewidth=0.5, alpha=0.3)
    
    # Draw obstacles
    for idx, obs in enumerate(map0.road_blocking):
        ax.plot(obs[0], obs[1], 's', color='black', markersize=15, 
               label='Obstacle' if idx == 0 else '')
    
    # Draw start and goal
    ax.plot(map0.start_point[0], map0.start_point[1], 'go', markersize=12, label='Start')
    ax.plot(map0.end_point[0], map0.end_point[1], 'r*', markersize=20, label='Goal')
    
    # Draw path
    ax.plot(path_x_pi, path_y_pi, 'r-', linewidth=2, alpha=0.6)
    ax.plot(path_x_pi, path_y_pi, 'ro', markersize=5, alpha=0.6)
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel('X Coordinate', fontsize=11)
    ax.set_ylabel('Y Coordinate', fontsize=11)
    ax.set_title(f'Policy Iteration Path ({len(path_pi)-1} steps)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    
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
                action_idx = int(policy_det[x, y])
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
                action_idx = int(policy_det[x, y])
                action_str = ["UP", "DOWN", "LEFT", "RIGHT"][action_idx]
            
            state_key = f"({x},{y})"
            policy_json["policy"][state_key] = action_str
    
    return policy_json


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
