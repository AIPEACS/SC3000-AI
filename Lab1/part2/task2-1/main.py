"""
Task 1: Planning Algorithms Implementation
==========================================
Main script for implementing and comparing Value Iteration and Policy Iteration.

This script:
1. Initializes the environment and value/policy functions
2. Implements Value Iteration algorithm
3. Implements Policy Iteration algorithm
4. Compares and visualizes the results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import numpy as np
from agent import (
    STEP_COST, GOAL_REWARD, NET_GOAL_REWARD, GAMMA, 
    ACTIONS, ACTION_MAP, reward_calc, get_next_state, calculate_final_reward
)
import scene_map as map0
from visualization import (
    VIS_DIR, print_value_function, print_policy, action_tensor_to_markdown,
    save_policy_json, save_action_tensor_json
)


# ==================== INITIALIZATION FUNCTIONS ====================

def initialize_value_function():
    """
    Initialize the value function V(s) = 0 for all states.
    
    Returns:
        dict: V[(x, y)] = 0.0 for all (x, y)
    """
    return {(x, y): 0.0 for x in range(5) for y in range(5)}


def initialize_policy():
    """
    Initialize a uniform random policy π(a|s).
    
    Returns:
        dict: policy[(x, y)] = np.ones(4)/4 for all (x, y)
    """
    return {(x, y): np.ones(4) / 4 for x in range(5) for y in range(5)}


def get_best_action(q_values):
    """
    Select the best action(s) given Q-values.
    
    Args:
        q_values: Array of Q-values for 4 actions at a state
        
    Returns:
        int: Index of the best action (or first best if tied)
    """
    return np.argmax(q_values)


def policy_to_deterministic(policy):
    """
    Convert a stochastic policy to a deterministic policy (greedy).
    
    Args:
        policy: dict {(x, y): np.array(4)} stochastic policy
        
    Returns:
        dict: {(x, y): int} best action index for each state, -1 for goal
    """
    result = {}
    for x in range(5):
        for y in range(5):
            if (x, y) == map0.end_point:
                result[(x, y)] = -1  # No action at goal
            else:
                result[(x, y)] = int(np.argmax(policy[(x, y)]))
    return result


# ==================== VALUE ITERATION ====================

def value_iteration(max_iterations=1000, theta=1e-6):
    """
    Value Iteration Algorithm
    =========================
    
    Algorithm:
    1. Initialize V(s) = 0 for all states
    2. For each iteration until convergence:
        - For each state s:
            - Compute the maximum Q-value over all actions
            - V(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
    3. Extract greedy policy from final value function
    
    Args:
        max_iterations: Maximum number of iterations
        theta: Convergence threshold (max change in value function)
        
    Returns:
        tuple: (value_function, policy, iteration_count, epoch_metrics)
            - value_function: 5x5 grid of state values
            - policy: 5x5 array of best actions
            - iteration_count: Number of iterations until convergence
            - epoch_metrics: List of max value changes per epoch
    """
    print("=" * 60)
    print("VALUE ITERATION")
    print("=" * 60)
    
    V = initialize_value_function()
    epoch_metrics = []  # Track max change per epoch
    
    for iteration in range(max_iterations):
        delta = 0  # Track maximum change in value function
        V_old = dict(V)
        
        # Backup: for each state, compute max Q-value over all actions
        for x in range(5):
            for y in range(5):
                # Skip goal state (terminal state)
                if (x, y) == map0.end_point:
                    V[(x, y)] = 0  # Terminal state has value 0
                    continue
                
                # Compute Q-values for all actions at this state
                q_values = []
                for action in ACTIONS:
                    next_x, next_y = get_next_state(x, y, action)
                    # Deterministic transition: P(s'|s,a) = 1
                    # Q(s,a) = R(s,a,s') + γV(s')  -- reward is for entering next state
                    q_value = reward_calc(next_x, next_y) + GAMMA * V_old[(next_x, next_y)]
                    q_values.append(q_value)
                
                # V(s) = max_a Q(s,a)
                V[(x, y)] = max(q_values)
                delta = max(delta, abs(V[(x, y)] - V_old[(x, y)]))
        
        # Log epoch metric
        epoch_metrics.append(delta)
        
        if (iteration + 1) % 100 == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}: max change = {delta:.6f}")
        
        # Check for convergence
        if delta < theta:
            print(f"✓ Converged at iteration {iteration + 1}")
            break
    
    # Extract deterministic greedy policy from value function
    policy_deterministic = {}
    for x in range(5):
        for y in range(5):
            if (x, y) == map0.end_point:
                policy_deterministic[(x, y)] = -1  # No action needed at goal
                continue
            
            q_values = []
            for action in ACTIONS:
                next_x, next_y = get_next_state(x, y, action)
                q_value = reward_calc(next_x, next_y) + GAMMA * V[(next_x, next_y)]
                q_values.append(q_value)
            
            policy_deterministic[(x, y)] = int(np.argmax(q_values))
    
    print()
    return V, policy_deterministic, iteration + 1, epoch_metrics


# ==================== POLICY ITERATION ====================

def policy_evaluation(policy, max_iterations=1000, theta=1e-6):
    """
    Policy Evaluation - Bellman Expectation Equation
    ================================================
    
    Iteratively compute V^π(s) for all states under policy π:
    V^π(s) = Σ_a π(a|s) Σ_s' P(s'|s,a)[R(s,a,s') + γV^π(s')]
    
    Args:
        policy: 5x5x4 stochastic policy tensor
        max_iterations: Maximum iterations
        theta: Convergence threshold
        
    Returns:
        tuple: (value_function, max_delta)
            - value_function: Value function V^π under the given policy
            - max_delta: Maximum value change in final iteration
    """
    V = initialize_value_function()
    
    for iteration in range(max_iterations):
        delta = 0
        V_old = V.copy()
        
        for x in range(5):
            for y in range(5):
                if (x, y) == map0.end_point:
                    V[(x, y)] = 0
                    continue
                
                # V^π(s) = Σ_a π(a|s) Q^π(s,a)
                value = 0
                for action_idx in range(4):
                    action = ACTIONS[action_idx]
                    action_prob = policy[(x, y)][action_idx]
                    
                    next_x, next_y = get_next_state(x, y, action)
                    # Q^π(s,a) = R(s') + γV^π(s')  -- reward is for entering next state
                    q_value = reward_calc(next_x, next_y) + GAMMA * V_old[(next_x, next_y)]
                    value += action_prob * q_value
                
                V[(x, y)] = value
                delta = max(delta, abs(V[(x, y)] - V_old[(x, y)]))
        
        if delta < theta:
            break
    
    return V, delta


def policy_improvement(V):
    """
    Policy Improvement - Greedy Policy Extraction
    ==============================================
    
    For each state, construct a new policy that acts greedily w.r.t. value function:
    π'(s) = argmax_a Q(s,a)
    
    Args:
        V: Current value function V(s)
        
    Returns:
        tuple: (new_policy, policy_stable)
            - new_policy: 5x5x4 deterministic greedy policy (one-hot encoded)
            - policy_stable: Boolean indicating if policy has stabilized
    """
    new_policy = {(x, y): np.zeros(4) for x in range(5) for y in range(5)}
    policy_stable = True
    
    for x in range(5):
        for y in range(5):
            if (x, y) == map0.end_point:
                new_policy[(x, y)][:] = 0  # No action at goal
                continue
            
            # Compute Q-values for all actions
            q_values = []
            for action in ACTIONS:
                next_x, next_y = get_next_state(x, y, action)
                q_value = reward_calc(next_x, next_y) + GAMMA * V[(next_x, next_y)]
                q_values.append(q_value)
            
            # Choose greedy action
            best_action = np.argmax(q_values)
            new_policy[(x, y)][best_action] = 1.0
    
    return new_policy, policy_stable


def policy_iteration(max_iterations=1000, eval_theta=1e-6):
    """
    Policy Iteration Algorithm
    ==========================
    
    Algorithm:
    1. Initialize π(s) uniformly random
    2. Repeat until convergence:
        a) Policy Evaluation: Compute V^π(s) for all states under current policy
        b) Policy Improvement: π'(s) = argmax_a Q^π(s,a)
        c) Check if π' == π; if yes, convergence reached
    
    Args:
        max_iterations: Maximum policy iterations
        eval_theta: Convergence threshold for policy evaluation
        
    Returns:
        tuple: (value_function, policy, iteration_count, epoch_metrics)
            - value_function: 5x5 grid of state values
            - policy: 5x5 array of best actions
            - iteration_count: Number of policy iterations
            - epoch_metrics: List of max value changes per epoch
    """
    print("=" * 60)
    print("POLICY ITERATION")
    print("=" * 60)
    
    policy = initialize_policy()
    epoch_metrics = []  # Track max change per epoch
    
    for iteration in range(max_iterations):
        # 1. Policy Evaluation
        V, eval_delta = policy_evaluation(policy, max_iterations=1000, theta=eval_theta)
        
        # 2. Policy Improvement
        policy_old = {k: v.copy() for k, v in policy.items()}
        policy, _ = policy_improvement(V)
        
        # 3. Check for convergence
        policy_changed = any(
            not np.array_equal(policy[(x, y)], policy_old[(x, y)])
            for x in range(5) for y in range(5)
        )
        
        # Log epoch metric (use max delta from evaluation)
        epoch_metrics.append(eval_delta)
        
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}: Policy changed = {policy_changed}")
        
        if not policy_changed:
            print(f"✓ Converged at iteration {iteration + 1}")
            break
    
    # Convert stochastic policy to deterministic
    policy_deterministic = policy_to_deterministic(policy)
    
    print()
    return V, policy_deterministic, iteration + 1, epoch_metrics


# ==================== VISUALIZATION & COMPARISON ====================

def compare_policies(policy1, policy2):
    """
    Compare two policies and report differences.
    """
    differences = 0
    for x in range(5):
        for y in range(5):
            if (x, y) != map0.end_point:
                if policy1[(x, y)] != policy2[(x, y)]:
                    differences += 1
    
    total_states = 25 - len(map0.road_blocking) - 1  # Exclude roadblocks and goal
    
    print(f"Policy Comparison:")
    print(f"  - Total states: {total_states}")
    print(f"  - Differences: {differences}")
    if differences == 0:
        print(f"  ✓ POLICIES ARE IDENTICAL")
    else:
        print(f"  - Match rate: {(total_states - differences) / total_states * 100:.1f}%")
    print()


def compare_value_functions(V1, V2):
    """
    Compare two value functions and report statistics.
    """
    diffs = [abs(V1[(x, y)] - V2[(x, y)]) for x in range(5) for y in range(5)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    
    print(f"Value Function Comparison:")
    print(f"  - Max difference: {max_diff:.6f}")
    print(f"  - Mean difference: {mean_diff:.6f}")
    if max_diff < 1e-4:
        print(f"  ✓ VALUE FUNCTIONS ARE EQUIVALENT")
    print()


# ==================== TEST POLICIES ====================

def extract_policy_actions(policy_det):
    """
    Convert deterministic policy to a dictionary mapping states to actions.
    """
    policy_dict = {}
    for x in range(5):
        for y in range(5):
            if (x, y) != map0.end_point:
                action_idx = policy_det[(x, y)]
                policy_dict[(x, y)] = ACTIONS[action_idx]
    return policy_dict


def test_policy(policy_det, name="Policy", max_steps=50):
    """
    Test a learned policy by running an episode from start to goal.
    
    Args:
        policy_det: Deterministic policy (5x5 array of action indices)
        name: Name for reporting
        max_steps: Maximum steps before timeout
        
    Returns:
        tuple: (path, total_reward, steps_to_goal, success, rewards_per_step)
    """
    x, y = map0.start_point
    path = [(x, y)]
    total_reward = 0
    rewards_per_step = [0]  # Initial state has 0 reward
    
    for step in range(max_steps):
        if (x, y) == map0.end_point:
            print(f"{name}: ✓ Reached goal in {step} steps, Total reward: {total_reward:.2f}")
            return path, total_reward, step, True, rewards_per_step
        
        action_idx = policy_det[(x, y)]
        action = ACTIONS[action_idx]
        
        x, y = get_next_state(x, y, action)
        path.append((x, y))
        step_reward = reward_calc(x, y) if (x, y) != map0.end_point else NET_GOAL_REWARD
        total_reward += step_reward
        rewards_per_step.append(step_reward)
    
    print(f"{name}: ✗ Did not reach goal within {max_steps} steps")
    return path, total_reward, max_steps, False, rewards_per_step


# ==================== MAIN EXECUTION ====================

def main():
    """
    Main execution: Run both algorithms and compare results.
    """
    print("\n" + "=" * 60)
    print("TASK 1: PLANNING WITH KNOWN ENVIRONMENT MODEL")
    print("=" * 60 + "\n")
    
    # -------- VALUE ITERATION --------
    V_vi, policy_vi, iter_vi, metrics_vi = value_iteration(max_iterations=1000, theta=1e-6)
    
    # -------- POLICY ITERATION --------
    V_pi, policy_pi, iter_pi, metrics_pi = policy_iteration(max_iterations=1000, eval_theta=1e-6)
    
    # -------- RESULTS --------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nConvergence Summary:")
    print(f"  Value Iteration:    {iter_vi} iterations")
    print(f"  Policy Iteration:   {iter_pi} iterations")
    
    # Print value functions
    print_value_function(V_vi, "Value Iteration - Value Function")
    print_value_function(V_pi, "Policy Iteration - Value Function")
    
    # Print policies
    print_policy(policy_vi, "Value Iteration - Optimal Policy")
    print_policy(policy_pi, "Policy Iteration - Optimal Policy")
    
    # -------- COMPARISONS --------
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    compare_value_functions(V_vi, V_pi)
    compare_policies(policy_vi, policy_pi)
    
    # -------- TEST POLICIES --------
    print("\n" + "=" * 60)
    print("POLICY TESTING")
    print("=" * 60 + "\n")
    
    path_vi, reward_vi, steps_vi, success_vi, rewards_vi = test_policy(policy_vi, "Value Iteration Policy")
    path_pi, reward_pi, steps_pi, success_pi, rewards_pi = test_policy(policy_pi, "Policy Iteration Policy")
    
    # -------- JSON & MARKDOWN EXPORT --------
    print("\n" + "=" * 60)
    print("RESULTS EXPORT")
    print("=" * 60)
    
    # Export action tensors to JSON
    print("\n📋 Exporting action tensors to JSON format...")
    save_action_tensor_json(policy_vi, "ValueIteration_Optimal")
    print()
    save_action_tensor_json(policy_pi, "PolicyIteration_Optimal")
    
    # Export action tensors to Markdown
    print("\n📊 Exporting action tensors to Markdown format...")
    md_vi = action_tensor_to_markdown(policy_vi, "Value Iteration - Optimal Policy")
    md_pi = action_tensor_to_markdown(policy_pi, "Policy Iteration - Optimal Policy")
    
    md_path = os.path.join(VIS_DIR, 'task1_policies.md')
    with open(md_path, 'w') as f:
        f.write("# Task 1: Planning with Known Environment\n\n")
        f.write("## Problem\n")
        f.write("5x5 Grid World with obstacles at (2,1) and (2,3). Discount factor gamma=0.9, step cost=-1, goal reward=+10.\n\n")
        f.write("## Optimal Policies\n\n")
        f.write(md_vi + "\n")
        f.write(md_pi + "\n")
        f.write("## Legend\n")
        f.write("- `UP` = Move up\n")
        f.write("- `DOWN` = Move down\n")
        f.write("- `LEFT` = Move left\n")
        f.write("- `RIGHT` = Move right\n")
        f.write("- `OBS` = Obstacle\n")
        f.write("- `GOAL` = Goal state (4,4)\n")
    print(f"✓ Saved policies to: {md_path}")
    
    # Create summary JSON
    print("\n📋 Creating summary report...")
    summary = {
        "task": "Task 1: Planning with Known Environment Model",
        "discount_factor": float(GAMMA),
        "step_cost": STEP_COST,
        "goal_reward": GOAL_REWARD,
        "convergence": {
            "value_iteration_iterations": iter_vi,
            "policy_iteration_iterations": iter_pi
        },
        "testing": {
            "value_iteration": {
                "steps_to_goal": steps_vi,
                "total_reward": float(reward_vi),
                "path": path_vi
            },
            "policy_iteration": {
                "steps_to_goal": steps_pi,
                "total_reward": float(reward_pi),
                "path": path_pi
            }
        }
    }
    
    summary_path = os.path.join(VIS_DIR, 'task1_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to: {summary_path}")
    
    print("\n" + "=" * 60)
    print("TASK 1 COMPLETE")
    print(f"All outputs saved to: {VIS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()


