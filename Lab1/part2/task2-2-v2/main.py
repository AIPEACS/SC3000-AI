# MIT License
# Copyright (c) 2026 AIPEAC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Task 2-v2: Monte Carlo Learning — Sliding-Window Returns
=========================================================
Identical to task2-2 except Q(s,a) is estimated from the most recent
1000 returns only (a fixed-size deque) rather than every return ever seen.

Rationale:
  Averaging all historical returns gives equal weight to samples collected
  under very early (and very different) policies.  Keeping only the last
  1000 returns per (s,a) pair discards stale data and lets the estimate
  track the current near-optimal policy more faithfully.

Changes vs task2-2/main.py:
  - initialize_returns()  → uses collections.deque(maxlen=WINDOW)
  - monte_carlo_control() → append() auto-pops the oldest entry once the
                            window is full; np.mean() works unchanged.
  - VIS_DIR               → points to task2-2-v2/visualization/
"""

import sys
import os

# ── shared code lives in task2-2 ──────────────────────────────────────────────
_TASK22_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'task2-2')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # part2/
sys.path.insert(0, _TASK22_DIR)

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import numpy as np
import random
from collections import deque

from agent_task2 import (  # type: ignore[import]
    STEP_COST, GOAL_REWARD, NET_GOAL_REWARD, GAMMA,
    ACTIONS, ACTION_MAP, reward_calc, get_next_state_stochastic
)
import scene_map as map0  # type: ignore[import]
import visualization_task2 as vis_mod  # type: ignore[import]

# ── redirect all outputs to this folder ───────────────────────────────────────
_V2_VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualization')
_V2_DEBUG_VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_visualization')
os.makedirs(_V2_VIS_DIR, exist_ok=True)
os.makedirs(_V2_DEBUG_VIS_DIR, exist_ok=True)
vis_mod.VIS_DIR = _V2_VIS_DIR               # patch: plot_training_history / save_* functions
vis_mod.DEBUG_VIS_DIR = _V2_DEBUG_VIS_DIR   # patch: plot_q_value_history

from visualization_task2 import (  # type: ignore[import]
    VIS_DIR, print_policy, action_tensor_to_markdown,
    save_policy_json, save_action_tensor_json, save_q_values,
    plot_training_history, plot_q_value_history
)

# ==================== CONFIGURATION ====================

WINDOW = 1000   # sliding-window size for return averaging

# ==================== INITIALIZATION ====================

def initialize_q_values():
    Q = {}
    for x in range(5):
        for y in range(5):
            for action in ACTIONS:
                Q[((x, y), action)] = 0.0
    return Q


def initialize_returns():
    """
    Each (s,a) gets a deque of at most WINDOW returns.
    Once the deque is full, appending a new return automatically
    pops the oldest one from the left.
    """
    returns = {}
    for x in range(5):
        for y in range(5):
            for action in ACTIONS:
                returns[((x, y), action)] = deque(maxlen=WINDOW)
    return returns


def initialize_visit_counts():
    visits = {}
    for x in range(5):
        for y in range(5):
            for action in ACTIONS:
                visits[((x, y), action)] = 0
    return visits

# ==================== EPISODE GENERATION ====================

def select_action_epsilon_greedy(x, y, Q, epsilon=0.1):
    for action in ACTIONS:
        if ((x, y), action) not in Q:
            Q[((x, y), action)] = 0.0

    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        q_values = [Q[((x, y), a)] for a in ACTIONS]
        return ACTIONS[int(np.argmax(q_values))]


def generate_episode(Q, epsilon=0.1, max_steps=1000):
    x, y = map0.start_point
    episode = []
    total_reward = 0

    for _ in range(max_steps):
        if (x, y) == map0.end_point:
            break
        action = select_action_epsilon_greedy(x, y, Q, epsilon)
        next_x, next_y = get_next_state_stochastic(x, y, action)
        reward = reward_calc(next_x, next_y)
        episode.append(((x, y), action, reward))
        total_reward += reward
        x, y = next_x, next_y

    return episode, total_reward

# ==================== MONTE CARLO CONTROL (sliding window) ====================

def monte_carlo_control(num_episodes=1000, epsilon=0.1):
    """
    On-policy first-visit MC control with a sliding-window return estimate.

    Q(s,a) = mean of the last WINDOW (≤1000) returns for (s,a).
    Older returns are discarded automatically by the deque.
    """
    print("=" * 60)
    print(f"MONTE CARLO CONTROL  [sliding window = {WINDOW}]")
    print("=" * 60)

    Q = initialize_q_values()
    returns = initialize_returns()
    visit_counts = initialize_visit_counts()

    training_history = []
    q_snapshots = []

    for episode_num in range(num_episodes):
        episode, episode_reward = generate_episode(Q, epsilon)

        visited_pairs = set()

        # forward pass: first time each (s,a) appears
        first_visits = {}
        for t, (state, action, _) in enumerate(episode):
            if (state, action) not in first_visits:
                first_visits[(state, action)] = t

        # backward pass: efficient G computation
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + GAMMA * G

            if first_visits.get((state, action)) == t:
                visited_pairs.add((state, action))

                # deque automatically drops oldest entry when len > WINDOW
                returns[(state, action)].append(G)
                visit_counts[(state, action)] += 1

                Q[(state, action)] = np.mean(returns[(state, action)])

        if (episode_num + 1) % 500 == 0 or episode_num == 0 or episode_num == num_episodes - 1:
            print(f"Episode {episode_num + 1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, Visited pairs={len(visited_pairs)}")

        training_history.append({
            'episode': episode_num + 1,
            'reward': episode_reward,
            'visited_pairs': len(visited_pairs),
            'epsilon': epsilon
        })

        if (episode_num + 1) % 100 == 0:
            snapshot = {
                (x, y): {a: Q[((x, y), a)] for a in ACTIONS}
                for x in range(5) for y in range(5)
            }
            q_snapshots.append((episode_num + 1, snapshot))

    # deterministic policy from final Q
    policy = {}
    for x in range(5):
        for y in range(5):
            if (x, y) == map0.end_point:
                policy[(x, y)] = -1
            elif (x, y) in map0.road_blocking:
                policy[(x, y)] = 4
            else:
                policy[(x, y)] = int(np.argmax([Q[((x, y), a)] for a in ACTIONS]))

    print(f"\n✓ Training complete after {num_episodes} episodes\n")
    return Q, policy, training_history, q_snapshots

# ==================== HELPERS ====================

def q_values_to_array(Q):
    q_state_action = np.zeros((5, 5, 4))
    v_state = np.zeros((5, 5))
    for x in range(5):
        for y in range(5):
            for a_idx, action in enumerate(ACTIONS):
                q_state_action[x, y, a_idx] = Q[((x, y), action)]
            v_state[x, y] = np.max(q_state_action[x, y, :])
    return q_state_action, v_state


def compare_policies(policy_mc, policy_optimal):
    differences = 0
    total_states = 0
    for x in range(5):
        for y in range(5):
            if (x, y) not in map0.road_blocking and (x, y) != map0.end_point:
                total_states += 1
                if policy_mc[(x, y)] != policy_optimal[(x, y)]:
                    differences += 1
    match_rate = ((total_states - differences) / total_states * 100) if total_states > 0 else 0
    print(f"Policy Comparison (MC-v2 vs Optimal):")
    print(f"  - Total states: {total_states}")
    print(f"  - Differences: {differences}")
    print(f"  - Match rate: {match_rate:.1f}%")
    print()


def _similarity_md_section(policy):
    """Build an MD section comparing policy vs VI optimal. Returns empty string on failure."""
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

# ==================== MAIN ====================

def main():
    print("\n" + "=" * 60)
    print("TASK 2-v2: MC LEARNING — SLIDING-WINDOW RETURNS")
    print("=" * 60 + "\n")

    Q, policy_mc, training_history, q_snapshots = monte_carlo_control(
        num_episodes=30000, epsilon=0.1
    )

    q_state_action, v_state = q_values_to_array(Q)

    print("\n" + "=" * 60)
    print("LEARNED POLICY")
    print("=" * 60)
    print_policy(policy_mc, "Monte Carlo v2 (window=1000) — Learned Policy")

    print("\n" + "=" * 60)
    print("RESULTS EXPORT")
    print("=" * 60)

    print("\n📋 Exporting action tensors...")
    save_action_tensor_json(policy_mc, "MonteCarlo_v2_Learned")

    print("\n📋 Exporting Q-values...")
    save_q_values(q_state_action, "MonteCarlo_v2_Q_values")

    print("\n📊 Exporting Markdown policy...")
    md_mc = action_tensor_to_markdown(policy_mc, "Monte Carlo v2 — Learned Policy")
    md_path = os.path.join(_V2_VIS_DIR, 'task2_v2_policies.md')
    with open(md_path, 'w') as f:
        f.write("# Task 2-v2: Monte Carlo Learning (Sliding Window)\n\n")
        f.write("## Configuration\n")
        f.write(f"- Window size: {WINDOW} most recent returns per (s,a)\n")
        f.write("- Epsilon-greedy exploration: ε = 0.1\n")
        f.write("- Stochastic transitions: 0.8 intended, 0.1 each perpendicular.\n\n")
        f.write("## Learned Policy\n\n")
        f.write(md_mc + "\n")
        f.write("## Legend\n")
        f.write("- `UP` / `DOWN` / `LEFT` / `RIGHT` = action\n")
        f.write("- `OBS` = Obstacle\n")
        f.write("- `GOAL` = Goal state (4,4)\n")
        f.write(_similarity_md_section(policy_mc))
    print(f"✓ Saved policies to: {md_path}")

    print("\n📈 Generating training history plots...")
    plot_training_history(training_history, "MonteCarlo_v2_Training")

    print("\n📈 Generating Q-value history debug plot...")
    plot_q_value_history(q_snapshots)

    summary = {
        "task": "Task 2-v2: Monte Carlo Learning (Sliding Window)",
        "algorithm": "First-Visit MC Control, sliding-window returns",
        "window_size": WINDOW,
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
    summary_path = os.path.join(_V2_VIS_DIR, 'task2_v2_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to: {summary_path}")

    print("\n" + "=" * 60)
    print("TASK 2-v2 COMPLETE")
    print(f"All outputs saved to: {_V2_VIS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
