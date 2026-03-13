"""
Task 2: Monte Carlo Learning with Unknown Transition Model
===========================================================
Main script for implementing Monte Carlo (MC) control using epsilon-greedy exploration.

This script:
1. Initializes Q-values and returns tracking
2. Implements Episode generation with stochastic environment
3. Runs Monte Carlo control (first-visit or every-visit)
4. Tracks training metrics per episode
5. Extracts learned policy and compares with Task 1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import numpy as np
import random
from agent_task2 import (
    STEP_COST, GOAL_REWARD, NET_GOAL_REWARD, GAMMA, 
    ACTIONS, ACTION_MAP, reward_calc, get_next_state_stochastic
)
import scene_map as map0
from visualization_task2 import (
    VIS_DIR, print_policy, action_tensor_to_markdown,
    save_policy_json, save_action_tensor_json, save_q_values
)


# ==================== INITIALIZATION ====================

def initialize_q_values():
    """
    Initialize Q(s,a) = 0 for all state-action pairs.
    
    Returns:
        dict: Q-values keyed by ((x, y), action)
    """
    Q = {}
    for x in range(5):
        for y in range(5):
            for action in ACTIONS:
                Q[((x, y), action)] = 0.0
    return Q


def initialize_returns():
    """
    Initialize returns tracking for each state-action pair.
    
    Returns:
        dict: Lists of returns keyed by ((x, y), action)
    """
    returns = {}
    for x in range(5):
        for y in range(5):
            for action in ACTIONS:
                returns[((x, y), action)] = []
    return returns


def initialize_visit_counts():
    """
    Initialize visit counts for each state-action pair.
    
    Returns:
        dict: Visit counts keyed by ((x, y), action)
    """
    visits = {}
    for x in range(5):
        for y in range(5):
            for action in ACTIONS:
                visits[((x, y), action)] = 0
    return visits


# ==================== EPISODE GENERATION ====================

def select_action_epsilon_greedy(x, y, Q, epsilon=0.1):
    """
    Select action using epsilon-greedy strategy.
    
    Args:
        x, y: Current state
        Q: Q-value dictionary
        epsilon: Exploration probability
        
    Returns:
        str: Selected action
    """
    # Initialize missing state-actions if needed (defensive)
    for action in ACTIONS:
        if ((x, y), action) not in Q:
            Q[((x, y), action)] = 0.0
    
    if random.random() < epsilon:
        # Explore: select random action
        return random.choice(ACTIONS)
    else:
        # Exploit: select best action
        q_values = [Q[((x, y), a)] for a in ACTIONS]
        best_action_idx = np.argmax(q_values)
        return ACTIONS[best_action_idx]


def generate_episode(Q, epsilon=0.1, max_steps=1000):
    """
    Generate one complete episode following epsilon-greedy policy.
    
    Args:
        Q: Q-value dictionary
        epsilon: Exploration probability
        max_steps: Maximum steps before timeout
        
    Returns:
        tuple: (episode, total_reward)
            - episode: List of (state, action, reward) tuples
            - total_reward: Total reward for the episode
    """
    x, y = map0.start_point
    episode = []
    total_reward = 0
    
    for step in range(max_steps):
        if (x, y) == map0.end_point:
            break
        
        # Select action using epsilon-greedy
        action = select_action_epsilon_greedy(x, y, Q, epsilon)
        
        # Take action in stochastic environment
        next_x, next_y = get_next_state_stochastic(x, y, action)
        
        # Calculate reward
        reward = reward_calc(next_x, next_y)
        
        # Record transition
        episode.append(((x, y), action, reward))
        total_reward += reward
        
        # Move to next state
        x, y = next_x, next_y
    
    return episode, total_reward


# ==================== MONTE CARLO CONTROL ====================

def monte_carlo_control(num_episodes=1000, epsilon=0.1):
    """
    Monte Carlo Control - First-visit MC prediction and control.
    
    Args:
        num_episodes: Number of episodes to train
        epsilon: Exploration probability
        
    Returns:
        tuple: (Q, policy, training_history)
            - Q: Learned Q-value dictionary
            - policy: 5x5 array of best actions
            - training_history: List of metrics per episode
    """
    print("=" * 60)
    print("MONTE CARLO CONTROL")
    print("=" * 60)
    
    Q = initialize_q_values()
    returns = initialize_returns()
    visit_counts = initialize_visit_counts()
    
    training_history = []  # Track metrics per episode
    q_snapshots = []       # Q-value snapshots every 100 episodes
    
    for episode_num in range(num_episodes):
        # Generate episode
        episode, episode_reward = generate_episode(Q, epsilon)
        
        # Track training metrics
        visited_pairs = set()
        
        # First pass (forward): identify first occurrence of each (state, action)
        first_visits = {}
        for t, (state, action, _) in enumerate(episode):
            if (state, action) not in first_visits:
                first_visits[(state, action)] = t
        
        # Process episode (backwards for efficient G computation)
        G = 0  # Return
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + GAMMA * G
            
            # First-visit: only update at the first occurrence (forward)
            if first_visits.get((state, action)) == t:
                visited_pairs.add((state, action))
                returns[(state, action)].append(G)
                visit_counts[(state, action)] += 1
                
                # Update Q-value as average of returns
                Q[(state, action)] = np.mean(returns[(state, action)])
        
        # Progress update every 500 episodes (plus first and last)
        if (episode_num + 1) % 500 == 0 or episode_num == 0 or episode_num == num_episodes - 1:
            visited_states = len(visited_pairs)
            print(f"Episode {episode_num + 1}/{num_episodes}: Reward={episode_reward:.2f}, Visited pairs={visited_states}")
        
        training_history.append({
            'episode': episode_num + 1,
            'reward': episode_reward,
            'visited_pairs': len(visited_pairs),
            'epsilon': epsilon
        })

        # Snapshot Q-values every 100 episodes
        if (episode_num + 1) % 100 == 0:
            snapshot = {
                (x, y): {a: Q[((x, y), a)] for a in ACTIONS}
                for x in range(5) for y in range(5)
            }
            q_snapshots.append((episode_num + 1, snapshot))
    
    # Extract deterministic policy from final Q-values
    policy = {}
    for x in range(5):
        for y in range(5):
            if (x, y) == map0.end_point:
                policy[(x, y)] = -1
            elif (x, y) in map0.road_blocking:
                policy[(x, y)] = 4
            else:
                q_values = [Q[((x, y), a)] for a in ACTIONS]
                policy[(x, y)] = int(np.argmax(q_values))
    
    print(f"\n✓ Training complete after {num_episodes} episodes\n")
    return Q, policy, training_history, q_snapshots


# ==================== POLICY EXTRACTION & COMPARISON ====================

def q_values_to_array(Q):
    """
    Convert Q-value dictionary to arrays for analysis.
    
    Returns:
        tuple: (q_state_action, v_state)
    """
    q_state_action = np.zeros((5, 5, 4))
    v_state = np.zeros((5, 5))
    
    for x in range(5):
        for y in range(5):
            for a_idx, action in enumerate(ACTIONS):
                q_state_action[x, y, a_idx] = Q[((x, y), action)]
            v_state[x, y] = np.max(q_state_action[x, y, :])
    
    return q_state_action, v_state


def compare_policies(policy_mc, policy_optimal):
    """
    Compare learned policy with optimal policy from Task 1.
    """
    differences = 0
    total_states = 0
    
    for x in range(5):
        for y in range(5):
            if (x, y) not in map0.road_blocking and (x, y) != map0.end_point:
                total_states += 1
                if policy_mc[(x, y)] != policy_optimal[(x, y)]:
                    differences += 1
    
    match_rate = ((total_states - differences) / total_states * 100) if total_states > 0 else 0
    
    print(f"Policy Comparison (MC vs Optimal):")
    print(f"  - Total states: {total_states}")
    print(f"  - Differences: {differences}")
    print(f"  - Match rate: {match_rate:.1f}%")
    print()


def _similarity_md_section(policy):
    """Build an MD section comparing policy vs VI optimal. Returns empty string on failure."""
    vi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'task2-1', 'visualization',
                           'task1_Optimal_action_tensor.json')
    try:
        with open(vi_path) as f:
            data = json.load(f)
        tensor = data["action_tensor"]
        optimal = {(x, y): int(tensor[y][x]) for y in range(5) for x in range(5)}
    except Exception as e:
        return f"\n## Similarity with Optimal Policy\n\n_Could not load VI optimal: {e}_\n"
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', -1: 'GOAL'}
    mismatches, matches, total = [], 0, 0
    for x in range(5):
        for y in range(5):
            if (x, y) in map0.road_blocking or (x, y) == map0.end_point:
                continue
            total += 1
            if policy[(x, y)] == optimal[(x, y)]:
                matches += 1
            else:
                mismatches.append(
                    f"({x},{y}) learned {action_names.get(policy[(x,y)],'?')}, "
                    f"optimal {action_names.get(optimal[(x,y)],'?')}"
                )
    pct = 100 * matches / total if total else 0
    lines = [
        "\n## Similarity with Optimal Policy",
        "",
        "- **Reference**: Value Iteration optimal policy (Task 2-1)",
        f"- **Evaluated states**: {total} (excludes obstacles and goal)",
        f"- **Matches**: {matches} / {total}",
        f"- **Similarity**: **{pct:.1f}%**",
    ]
    if mismatches:
        lines.append("- **Mismatched states**:")
        for m in mismatches:
            lines.append(f"  - {m}")
    else:
        lines.append("- All states match the optimal policy ✓")
    lines.append("")
    return "\n".join(lines) + "\n"


# ==================== MAIN EXECUTION ====================

def main():
    """
    Main execution: Train with Monte Carlo and compare with Task 1.
    """
    print("\n" + "=" * 60)
    print("TASK 2: MONTE CARLO LEARNING WITH UNKNOWN MODEL")
    print("=" * 60 + "\n")
    
    # -------- MONTE CARLO CONTROL --------
    Q, policy_mc, training_history, q_snapshots = monte_carlo_control(num_episodes=30000, epsilon=0.1)
    
    # -------- EXTRACT Q-VALUES --------
    q_state_action, v_state = q_values_to_array(Q)
    
    # -------- DISPLAY POLICY --------
    print("\n" + "=" * 60)
    print("LEARNED POLICY")
    print("=" * 60)
    print_policy(policy_mc, "Monte Carlo - Learned Policy")
    
    # -------- RESULTS EXPORT --------
    print("\n" + "=" * 60)
    print("RESULTS EXPORT")
    print("=" * 60)
    
    # Export action tensors to JSON
    print("\n Exporting action tensors to JSON format...")
    save_action_tensor_json(policy_mc, "MonteCarlo_Learned")
    
    # Export Q-values
    print("\n Exporting Q-values to JSON format...")
    save_q_values(q_state_action, "MonteCarlo_Q_values")
    
    # Export action tensors to Markdown
    print("\n Exporting action tensors to Markdown format...")
    md_mc = action_tensor_to_markdown(policy_mc, "Monte Carlo - Learned Policy")
    
    md_path = os.path.join(VIS_DIR, 'task2_policies.md')
    with open(md_path, 'w') as f:
        f.write("# Task 2: Monte Carlo Learning\n\n")
        f.write("## Problem\n")
        f.write("5x5 Grid World with stochastic transitions. Unknown environment model.\n")
        f.write("Stochastic transitions: 0.8 intended, 0.1 each perpendicular direction.\n")
        f.write("Epsilon-greedy exploration with epsilon=0.1.\n\n")
        f.write("## Learned Policy\n\n")
        f.write(md_mc + "\n")
        f.write("## Legend\n")
        f.write("- `UP` = Move up\n")
        f.write("- `DOWN` = Move down\n")
        f.write("- `LEFT` = Move left\n")
        f.write("- `RIGHT` = Move right\n")
        f.write("- `OBS` = Obstacle\n")
        f.write("- `GOAL` = Goal state (4,4)\n")
        f.write(_similarity_md_section(policy_mc))
    print(f"✓ Saved policies to: {md_path}")
    
    # Create summary JSON
    print("\n Creating summary report...")
    summary = {
        "task": "Task 2: Monte Carlo Learning",
        "algorithm": "First-Visit Monte Carlo Control",
        "num_episodes": len(training_history),
        "epsilon": 0.1,
        "discount_factor": float(GAMMA),
        "step_cost": STEP_COST,
        "goal_reward": GOAL_REWARD,
        "training": {
            "final_episode_reward": float(training_history[-1]['reward']) if training_history else 0,
            "avg_reward_last_100": float(np.mean([h['reward'] for h in training_history[-100:]])) if len(training_history) >= 100 else 0
        }
    }
    
    summary_path = os.path.join(VIS_DIR, 'task2_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to: {summary_path}")
    
    # Load Task 1 optimal policy for comparison
    try:
        task1_summary_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'task1', 'visualization', 'task1_summary.json'
        )
        if os.path.exists(task1_summary_path):
            with open(task1_summary_path, 'r') as f:
                task1_data = json.load(f)
            print("\n" + "=" * 60)
            print("COMPARISON WITH TASK 1")
            print("=" * 60)
            print(f"Task 1 - Value Iteration convergence: {task1_data['convergence']['value_iteration_iterations']} iterations")
            print(f"Task 2 - Monte Carlo training: {len(training_history)} episodes")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("TASK 2 COMPLETE")
    print(f"All outputs saved to: {VIS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
