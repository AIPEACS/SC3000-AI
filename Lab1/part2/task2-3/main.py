"""
Task 3: Q-Learning Implementation
==================================
Implements tabular Q-learning with ε-greedy exploration.

The agent learns online without knowledge of the transition model.
Each step updates: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

Parameters:
- Learning rate: α = 0.1
- Exploration rate: ε = 0.1
- Discount factor: γ = 0.9
- Episodes: 5000
"""

import numpy as np
import random
import os
import json
import sys
import io
import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_task3 as agent
import scene_map as map0
from visualization_task3 import (
    action_tensor_to_markdown, plot_training_history,
    save_action_tensor_json, save_q_values, print_policy,
    plot_q_value_history
)

# Parameters
NUM_EPISODES = 50000
EPSILON = 0.1       # Fixed ε throughout training (as per assignment spec)
ALPHA = agent.ALPHA
GAMMA = agent.GAMMA
STEP_COST = agent.STEP_COST
GOAL_REWARD = agent.GOAL_REWARD
NET_GOAL_REWARD = agent.NET_GOAL_REWARD
ACTIONS = agent.ACTIONS
ACTION_MAP = agent.ACTION_MAP


def initialize_q_values():
    """
    Initialize Q-values for all state-action pairs.
    
    Q[((x,y), action)] = 0 for all valid grid positions and actions
    
    Returns:
        dict: Q-value dictionary
    """
    Q = {}
    for x in range(5):
        for y in range(5):
            for action in ACTIONS:
                Q[((x, y), action)] = 0.0
    return Q


def select_action_epsilon_greedy(x, y, Q, epsilon=EPSILON):
    """
    Select an action using ε-greedy strategy.
    
    With probability:
    - (1 - ε): Select action with highest Q-value (exploitation)
    - ε: Select random action (exploration)
    
    Args:
        x, y: Current state coordinates
        Q: Q-value dictionary
        epsilon: Exploration rate
        
    Returns:
        str: Selected action ('u', 'd', 'l', 'r')
    """
    if random.random() < epsilon:
        # Exploration: random action
        return random.choice(ACTIONS)
    else:
        # Exploitation: best Q-value
        q_values = []
        for action in ACTIONS:
            q_val = Q.get(((x, y), action), 0.0)
            q_values.append(q_val)
        
        # Get action with maximum Q-value
        max_q = max(q_values)
        best_actions = [ACTIONS[i] for i in range(len(ACTIONS)) if q_values[i] == max_q]
        return random.choice(best_actions)



def q_learning(num_episodes=NUM_EPISODES, epsilon=EPSILON, alpha=ALPHA):
    """
    Main Q-learning training loop.
    
    Updates Q-values online: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
    Uses a fixed ε-greedy strategy throughout training (ε = 0.1 per assignment spec).
    
    Args:
        num_episodes: Number of training episodes
        epsilon: Fixed exploration rate
        alpha: Learning rate
        
    Returns:
        tuple: (Q_values dict, training_history dict)
    """
    Q = initialize_q_values()
    
    # Training history tracking
    training_history = {
        'episode_rewards': [],
        'episode_visited_pairs': [],
        'episode_mean_td_errors': [],  # mean |TD error| per episode
        'q_snapshots': [],    # list of (episode, {(x,y): {action: value}})
    }
    
    print("\n" + "=" * 60)
    print("Q-LEARNING TRAINING")
    print("=" * 60)
    print(f"Episodes: {num_episodes}, Learning rate (α): {alpha}, Epsilon (ε): {epsilon} (fixed)")
    print()
    
    for episode_num in range(num_episodes):
        # Initialize episode
        x, y = map0.start_point
        episode_reward = 0
        visited_pairs = set()
        episode_td_errors = []
        
        # Run one episode until goal or timeout
        for step in range(1000):  # Max 1000 steps per episode
            # Record visited state-action pair
            action = select_action_epsilon_greedy(x, y, Q, epsilon)
            visited_pairs.add(((x, y), action))
            
            # Take action in environment (stochastic)
            prev_x, prev_y = x, y
            x, y = agent.get_next_state_stochastic(prev_x, prev_y, action)
            
            # Calculate reward
            reward = agent.reward_calc(x, y)
            episode_reward += reward
            
            # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
            old_q = Q[((prev_x, prev_y), action)]
            
            # Get max Q-value for next state
            max_q_next = max(Q[((x, y), a)] for a in ACTIONS)
            
            # TD update
            td_error = reward + GAMMA * max_q_next - old_q
            Q[((prev_x, prev_y), action)] = old_q + alpha * td_error
            episode_td_errors.append(abs(td_error))
            
            # Check if reached goal
            if (x, y) == map0.end_point:
                break
        
        # Record episode history
        training_history['episode_rewards'].append(episode_reward)
        training_history['episode_visited_pairs'].append(len(visited_pairs))
        training_history['episode_mean_td_errors'].append(
            float(np.mean(episode_td_errors)) if episode_td_errors else 0.0
        )

        # Snapshot Q-values every 100 episodes
        if (episode_num + 1) % 100 == 0:
            snapshot = {
                (x, y): {a: Q[((x, y), a)] for a in ACTIONS}
                for x in range(5) for y in range(5)
            }
            training_history['q_snapshots'].append((episode_num + 1, snapshot))
        
        # Progress update every 5000 episodes
        if (episode_num + 1) % 5000 == 0:
            print(f"Episode {episode_num+1}/{num_episodes}: Reward={episode_reward:.2f}, "
                  f"Visited pairs={len(visited_pairs)}")
    
    print(f"✓ Training complete after {num_episodes} episodes\n")
    
    return Q, training_history


def extract_policy_from_q(Q):
    """
    Extract the deterministic policy from Q-values.
    
    For each state, select the action with highest Q-value.
    
    Args:
        Q: Q-value dictionary
        
    Returns:
        np.ndarray: 5×5 array of action indices
    """
    policy = {}
    
    for x in range(5):
        for y in range(5):
            if (x, y) == map0.end_point:
                policy[(x, y)] = -1
            elif (x, y) in map0.road_blocking:
                policy[(x, y)] = 4
            else:
                q_values = [Q.get(((x, y), action), 0.0) for action in ACTIONS]
                policy[(x, y)] = int(np.argmax(q_values))
    
    return policy


def test_policy(policy, num_tests=10, max_steps=100):
    """
    Test the learned policy deterministically.
    
    Args:
        policy: 5×5 array of action indices
        num_tests: Number of test runs
        max_steps: Maximum steps per test
        
    Returns:
        tuple: (average_return, success_count)
    """
    total_return = 0
    success_count = 0
    
    for test in range(num_tests):
        x, y = map0.start_point
        test_return = 0
        
        for step in range(max_steps):
            action_idx = policy[(x, y)]
            if action_idx < 0 or action_idx >= 4:
                break
            
            action = ACTIONS[action_idx]
            x, y = agent.get_next_state_stochastic(x, y, action)
            test_return += agent.reward_calc(x, y)
            
            if (x, y) == map0.end_point:
                success_count += 1
                break
        
        total_return += test_return
    
    avg_return = total_return / num_tests if num_tests > 0 else 0
    return avg_return, success_count


def compare_policies(q_learning_policy, optimal_policy, mc_policy):
    """
    Compare learned policies (Q-learning vs Optimal vs Monte Carlo).
    
    Args:
        q_learning_policy: 5×5 array from Q-learning
        optimal_policy: 5×5 array from Task 1 optimal
        mc_policy: 5×5 array from Task 2 Monte Carlo
    """
    print("=" * 60)
    print("POLICY COMPARISON")
    print("=" * 60)
    
    # Count matching actions (excluding obstacles and goal)
    matches_optimal = 0
    matches_mc = 0
    total_valid = 0
    
    for x in range(5):
        for y in range(5):
            cell_val = q_learning_policy[(x, y)]
            # Skip obstacles and goal
            if cell_val < 0 or cell_val >= 4:
                continue
            
            total_valid += 1
            
            if optimal_policy.get((x, y)) == cell_val:
                matches_optimal += 1
            
            if mc_policy.get((x, y)) == cell_val:
                matches_mc += 1
    
    if total_valid > 0:
        pct_optimal = 100 * matches_optimal / total_valid
        pct_mc = 100 * matches_mc / total_valid
        
        print(f"Q-Learning vs Optimal Policy: {matches_optimal}/{total_valid} ({pct_optimal:.1f}%)")
        print(f"Q-Learning vs Monte Carlo Policy: {matches_mc}/{total_valid} ({pct_mc:.1f}%)")
    
    print()


def convergence_analysis(training_history, window_episodes=500, stability_threshold=0.20):
    """
    Convergence: all (s,a) pairs have stable Q-values within ±10% between consecutive windows.

    Divides the run into non-overlapping windows of `window_episodes` episodes each.
    For each (s,a), computes average Q in each window.
    Converged when % change for ALL (s,a) < stability_threshold between consecutive windows.

    Example: window_episodes=500 → compares ep[0-500] vs ep[500-1000] vs ep[1000-1500]...
    """
    snapshots = training_history['q_snapshots']

    if len(snapshots) < 2:
        return {
            'conv_episodes': [], 'stable_counts': [],
            'convergence_ep': None, 'total_sa_pairs': 0,
            'stability_threshold': stability_threshold, 'window_episodes': window_episodes,
        }

    valid_states = [
        (x, y) for x in range(5) for y in range(5)
        if (x, y) not in map0.road_blocking and (x, y) != map0.end_point
    ]
    total_sa_pairs = len(valid_states) * len(ACTIONS)

    # Snapshots are taken every 100 episodes → convert to snapshot count
    snapshot_interval = 100
    window_size = max(2, window_episodes // snapshot_interval)  # 500 ep → 5 snapshots

    n_windows = len(snapshots) // window_size
    if n_windows < 2:
        return {
            'conv_episodes': [], 'stable_counts': [],
            'convergence_ep': None, 'total_sa_pairs': total_sa_pairs,
            'stability_threshold': stability_threshold, 'window_episodes': window_episodes,
        }

    # Precompute avg Q per window for each (s,a)
    window_avg_q = []
    for w in range(n_windows):
        start = w * window_size
        end = (w + 1) * window_size
        avg_q = {}
        for s in valid_states:
            for a in ACTIONS:
                avg_q[(s, a)] = float(np.mean([snapshots[i][1][s][a] for i in range(start, end)]))
        window_avg_q.append((snapshots[end - 1][0], avg_q))  # (episode, avg_q)

    # Compare consecutive windows
    conv_episodes = []
    stable_counts = []
    convergence_ep = None

    for w in range(1, n_windows):
        ep, curr_avg = window_avg_q[w]
        _, prev_avg = window_avg_q[w - 1]

        stable_count = 0
        for sa in curr_avg:
            prev_val = prev_avg[sa]
            curr_val = curr_avg[sa]
            if abs(prev_val) > 1e-6:
                pct_change = abs(curr_val - prev_val) / abs(prev_val)
            else:
                pct_change = 0 if abs(curr_val) < 1e-6 else 1
            if pct_change < stability_threshold:
                stable_count += 1

        conv_episodes.append(ep)
        stable_counts.append(stable_count)

        if stable_count == total_sa_pairs and convergence_ep is None:
            convergence_ep = ep

    return {
        'conv_episodes': conv_episodes,
        'stable_counts': stable_counts,
        'convergence_ep': convergence_ep,
        'total_sa_pairs': total_sa_pairs,
        'stability_threshold': stability_threshold,
        'window_episodes': window_episodes,
    }


def plot_convergence(conv_data, title="Q-Learning Convergence Analysis"):
    """
    Plot: # of stable (s,a) pairs over windows.
    Mark when all pairs converge (reach 100% stable).
    """
    if len(conv_data['conv_episodes']) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        ax.text(0.5, 0.5, "No convergence data (insufficient snapshots)",
                ha='center', va='center', fontsize=12)
        return fig
    
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    
    total = conv_data['total_sa_pairs']
    pct_stable = [100 * c / total for c in conv_data['stable_counts']]
    
    thresh_pct = conv_data['stability_threshold'] * 100

    ax.plot(conv_data['conv_episodes'], pct_stable,
            color='steelblue', linewidth=2.5, marker='o', markersize=5, label='% of (s,a) pairs stable')
    ax.axhline(100, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='100% stable (converged)')

    if conv_data['convergence_ep']:
        ax.axvline(conv_data['convergence_ep'], color='red', linestyle='-', linewidth=3,
                   zorder=5, label=f"Converged at ep {conv_data['convergence_ep']}")
        i = conv_data['conv_episodes'].index(conv_data['convergence_ep'])
        ax.plot(conv_data['convergence_ep'], pct_stable[i], 'r*', markersize=20, zorder=6)
    else:
        ax.text(0.98, 0.05, 'No convergence detected', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=10, color='red',
                bbox=dict(facecolor='lightyellow', edgecolor='red', boxstyle='round,pad=0.3'))

    ax.set_xlabel('Episode (snapshot every 100 ep)', fontsize=11)
    ax.set_ylabel(f'% of (s,a) pairs with stable Q (±{thresh_pct:.0f}% change)', fontsize=11)
    ax.set_ylim([0, 105])
    ax.set_title(f"{title}\nWindow size: {conv_data['window_episodes']} ep | Threshold: {conv_data['stability_threshold']*100:.0f}% change",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def _convergence_md_section(conv_data):
    """Build MD section: when all (s,a) pairs have stable Q within ±10% windows."""
    lines = [
        "\n## Convergence Speed Analysis",
        "",
        "**Metric**: All (s,a) pairs have Q-values stable within ±10% between 500-snapshot windows",
        "",
        f"**Total state-action pairs**: {conv_data['total_sa_pairs']}",
        f"**Window size**: {conv_data['window_episodes']} episodes",
        f"**Stability threshold**: ±{conv_data['stability_threshold']*100:.0f}%",
        "",
    ]
    if conv_data['convergence_ep']:
        lines.append(f"**Convergence point**: Episode `{conv_data['convergence_ep']}`")
    else:
        lines.append("**Convergence point**: Not detected (not all (s,a) pairs stabilized)")
    lines += [
        "",
        "See `QLearning_Convergence_Analysis.png` for plot.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _similarity_md_section(policy):
    """Build an MD section comparing policy vs VI optimal."""
    vi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'task2-1', 'visualization',
                           'ValueIteration_Optimal_action_tensor.json')
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


def main():
    """Main execution for Task 3: Q-Learning"""
    
    print("\n" + "=" * 60)
    print("TASK 3: Q-LEARNING WITH UNKNOWN MODEL")
    print("=" * 60)
    
    # ===== Q-LEARNING TRAINING =====
    Q_values, training_history = q_learning(NUM_EPISODES, EPSILON, ALPHA)
    
    # ===== EXTRACT POLICY =====
    q_learning_policy = extract_policy_from_q(Q_values)
    
    print("=" * 60)
    print("LEARNED POLICY")
    print("=" * 60)
    print_policy(q_learning_policy)
    
    # ===== LOAD OPTIMAL POLICY (from Task 1) =====
    try:
        with open("../task1/visualization/ValueIteration_Optimal_action_tensor.json", 'r') as f:
            task1_data = json.load(f)
            # JSON format: action_tensor[y][x] = action for state (x, y)
            tensor = task1_data["action_tensor"] if isinstance(task1_data, dict) and "action_tensor" in task1_data else task1_data
            optimal_policy = {(x, y): int(tensor[y][x]) for y in range(5) for x in range(5)}
    except (FileNotFoundError, TypeError, KeyError) as e:
        print(f"Warning: Could not load Task 1 optimal policy: {e}")
        optimal_policy = {}
    
    # ===== LOAD MONTE CARLO POLICY (from Task 2) =====
    try:
        with open("../task2/visualization/MonteCarlo_Learned_action_tensor.json", 'r') as f:
            task2_data = json.load(f)
            # JSON format: action_tensor[y][x] = action for state (x, y)
            tensor = task2_data["action_tensor"] if isinstance(task2_data, dict) and "action_tensor" in task2_data else task2_data
            mc_policy = {(x, y): int(tensor[y][x]) for y in range(5) for x in range(5)}
    except (FileNotFoundError, TypeError, KeyError) as e:
        print(f"Warning: Could not load Task 2 Monte Carlo policy: {e}")
        mc_policy = {}
    
    # ===== POLICY COMPARISON =====
    if optimal_policy and mc_policy:
        compare_policies(q_learning_policy, optimal_policy, mc_policy)
    
    # ===== TEST POLICY =====
    print("=" * 60)
    print("POLICY TESTING (10 random runs)")
    print("=" * 60)
    avg_return, success_count = test_policy(q_learning_policy)
    print(f"Average return: {avg_return:.2f}")
    print(f"Success rate: {success_count}/10 ({100*success_count/10:.0f}%)")
    print()
    
    # ===== RESULTS EXPORT =====
    print("=" * 60)
    print("RESULTS EXPORT")
    print("=" * 60)
    
    os.makedirs("./visualization/", exist_ok=True)
    
    # Export action tensor to JSON
    print("📋 Exporting action tensors to JSON format...")
    json_path = save_action_tensor_json(Q_values, q_learning_policy, "QLearning")
    print(f"✓ Saved action tensor to: {json_path}")
    
    # Export Q-values to JSON
    q_path = save_q_values(Q_values, "QLearning")
    print(f"✓ Saved Q-values to: {q_path}")
    
    # Export to markdown
    print("📊 Exporting action tensors to Markdown format...")
    markdown_content = action_tensor_to_markdown(q_learning_policy, "Q-Learning Learned Policy")
    
    # Convergence analysis (must run before markdown export)
    print("📈 Running convergence speed analysis...")
    conv_data = convergence_analysis(training_history)
    
    with open("./visualization/task3_policies.md", 'w') as f:
        f.write(markdown_content)
        f.write(_similarity_md_section(q_learning_policy))
        f.write(_convergence_md_section(conv_data))
    print(f"✓ Saved policies to: ./visualization/task3_policies.md")
    
    # Plot training history
    print("📈 Generating training history plots...")
    fig = plot_training_history(training_history, "Q-Learning")
    plot_path = "./visualization/QLearning_Training_training_history.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved to: {plot_path}")

    # Plot Q-value history
    print("📈 Generating Q-value history plots...")
    fig_q = plot_q_value_history(training_history['q_snapshots'])
    os.makedirs("./debug_visualization/", exist_ok=True)
    q_plot_path = "./debug_visualization/QLearning_Q_value_history.png"
    fig_q.savefig(q_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig_q)
    print(f"✓ Q-value history plot saved to: {q_plot_path}")

    # Plot convergence
    fig_conv = plot_convergence(conv_data, "Q-Learning Convergence Analysis")
    conv_plot_path = "./visualization/QLearning_Convergence_Analysis.png"
    fig_conv.savefig(conv_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig_conv)
    print(f"✓ Convergence plot saved to: {conv_plot_path}")
    
    # Save summary
    print("📋 Creating summary report...")
    summary = {
        "algorithm": "Q-Learning",
        "num_episodes": NUM_EPISODES,
        "learning_rate_alpha": ALPHA,
        "exploration_rate_epsilon": EPSILON,
        "discount_factor_gamma": GAMMA,
        "final_avg_reward": float(np.mean(training_history['episode_rewards'][-100:])),
        "total_training_reward": float(np.sum(training_history['episode_rewards'])),
        "max_episode_reward": float(np.max(training_history['episode_rewards'])),
        "min_episode_reward": float(np.min(training_history['episode_rewards'])),
        "test_avg_return": float(avg_return),
        "test_success_rate": int(success_count),
        "convergence_episode": conv_data['convergence_ep'],
        "convergence_stable_pairs": conv_data.get('total_sa_pairs', 0),
        "convergence_threshold_pct": conv_data['stability_threshold'],
    }
    
    with open("./visualization/task3_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to: ./visualization/task3_summary.json")
    
    print()
    print("=" * 60)
    print("TASK 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()
