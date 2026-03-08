"""
Task 1: Planning Algorithms Implementation
==========================================
Main script for implementing and comparing Value Iteration and Policy Iteration.

This script:
1. Initializes the environment and value/policy functions
2. Implements Value Iteration algorithm
3. Implements Policy Iteration algorithm
4. Visualizes and compares the results
5. Tests the learned policies by running episodes
"""

import numpy as np
from agent import (
    STEP_COST, GOAL_REWARD, NET_GOAL_REWARD, GAMMA, 
    ACTIONS, ACTION_MAP, reward_calc, get_next_state, calculate_final_reward
)
import scene_map as map0


# ==================== INITIALIZATION FUNCTIONS ====================

def initialize_value_function():
    """
    Initialize the value function V(s) for all states.
    
    Returns:
        np.ndarray: 5x5 grid of values initialized to 0
    """
    return np.zeros((5, 5))


def initialize_policy():
    """
    Initialize a uniform random policy π(a|s).
    
    Returns:
        np.ndarray: 5x5x4 tensor where policy[x,y,a] is probability of action a at state (x,y).
                   Initially, all actions have equal probability (uniform random).
    """
    policy = np.ones((5, 5, 4)) / 4  # Uniform distribution over 4 actions
    return policy


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
        policy: 5x5x4 stochastic policy tensor
        
    Returns:
        np.ndarray: 5x5 array where each cell contains the best action index
    """
    deterministic_policy = np.zeros((5, 5), dtype=int)
    for x in range(5):
        for y in range(5):
            deterministic_policy[x, y] = np.argmax(policy[x, y, :])
    return deterministic_policy


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
        tuple: (value_function, policy, iteration_count)
            - value_function: 5x5 grid of state values
            - policy: 5x5 array of best actions
            - iteration_count: Number of iterations until convergence
    """
    print("=" * 60)
    print("VALUE ITERATION")
    print("=" * 60)
    
    V = initialize_value_function()
    
    for iteration in range(max_iterations):
        delta = 0  # Track maximum change in value function
        V_old = V.copy()
        
        # Backup: for each state, compute max Q-value over all actions
        for x in range(5):
            for y in range(5):
                # Skip goal state (terminal state)
                if (x, y) == map0.end_point:
                    V[x, y] = 0  # Terminal state has value 0
                    continue
                
                # Compute Q-values for all actions at this state
                q_values = []
                for action in ACTIONS:
                    next_x, next_y = get_next_state(x, y, action)
                    # Deterministic transition: P(s'|s,a) = 1
                    # Q(s,a) = R(s,a,s') + γV(s')
                    q_value = reward_calc(x, y) + GAMMA * V_old[next_x, next_y]
                    q_values.append(q_value)
                
                # V(s) = max_a Q(s,a)
                V[x, y] = max(q_values)
                delta = max(delta, abs(V[x, y] - V_old[x, y]))
        
        if (iteration + 1) % 100 == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}: max change = {delta:.6f}")
        
        # Check for convergence
        if delta < theta:
            print(f"✓ Converged at iteration {iteration + 1}")
            break
    
    # Extract deterministic greedy policy from value function
    policy_deterministic = np.zeros((5, 5), dtype=int)
    for x in range(5):
        for y in range(5):
            if (x, y) == map0.end_point:
                policy_deterministic[x, y] = -1  # No action needed at goal
                continue
            
            q_values = []
            for action in ACTIONS:
                next_x, next_y = get_next_state(x, y, action)
                q_value = reward_calc(x, y) + GAMMA * V[next_x, next_y]
                q_values.append(q_value)
            
            policy_deterministic[x, y] = np.argmax(q_values)
    
    print()
    return V, policy_deterministic, iteration + 1


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
        np.ndarray: Value function V^π under the given policy
    """
    V = initialize_value_function()
    
    for iteration in range(max_iterations):
        delta = 0
        V_old = V.copy()
        
        for x in range(5):
            for y in range(5):
                if (x, y) == map0.end_point:
                    V[x, y] = 0
                    continue
                
                # V^π(s) = Σ_a π(a|s) Q^π(s,a)
                value = 0
                for action_idx in range(4):
                    action = ACTIONS[action_idx]
                    action_prob = policy[x, y, action_idx]
                    
                    next_x, next_y = get_next_state(x, y, action)
                    # Q^π(s,a) = R + γV^π(s')
                    q_value = reward_calc(x, y) + GAMMA * V_old[next_x, next_y]
                    value += action_prob * q_value
                
                V[x, y] = value
                delta = max(delta, abs(V[x, y] - V_old[x, y]))
        
        if delta < theta:
            break
    
    return V


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
    new_policy = np.zeros((5, 5, 4))
    policy_stable = True
    
    for x in range(5):
        for y in range(5):
            if (x, y) == map0.end_point:
                new_policy[x, y, :] = 0  # No action at goal
                continue
            
            # Compute Q-values for all actions
            q_values = []
            for action in ACTIONS:
                next_x, next_y = get_next_state(x, y, action)
                q_value = reward_calc(x, y) + GAMMA * V[next_x, next_y]
                q_values.append(q_value)
            
            # Choose greedy action
            best_action = np.argmax(q_values)
            new_policy[x, y, best_action] = 1.0
    
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
        tuple: (value_function, policy, iteration_count)
            - value_function: 5x5 grid of state values
            - policy: 5x5 array of best actions
            - iteration_count: Number of policy iterations
    """
    print("=" * 60)
    print("POLICY ITERATION")
    print("=" * 60)
    
    policy = initialize_policy()
    
    for iteration in range(max_iterations):
        # 1. Policy Evaluation
        V = policy_evaluation(policy, max_iterations=1000, theta=eval_theta)
        
        # 2. Policy Improvement
        policy_old = policy.copy()
        policy, _ = policy_improvement(V)
        
        # 3. Check for convergence
        policy_changed = not np.array_equal(policy, policy_old)
        
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}: Policy changed = {policy_changed}")
        
        if not policy_changed:
            print(f"✓ Converged at iteration {iteration + 1}")
            break
    
    # Convert stochastic policy to deterministic
    policy_deterministic = policy_to_deterministic(policy)
    
    print()
    return V, policy_deterministic, iteration + 1


# ==================== VISUALIZATION & COMPARISON ====================

def action_to_symbol(action_idx):
    """Convert action index to visual symbol."""
    symbols = {0: '↑', 1: '↓', 2: '←', 3: '→', -1: 'G'}
    return symbols.get(action_idx, '?')


def print_value_function(V, title="Value Function"):
    """
    Display the value function as a grid.
    Values are shown with 2 decimal places.
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
    Display the policy as a grid with action symbols.
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


def compare_policies(policy1, policy2):
    """
    Compare two policies and report differences.
    """
    differences = 0
    for x in range(5):
        for y in range(5):
            if (x, y) != map0.end_point:
                if policy1[x, y] != policy2[x, y]:
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
    differences = np.abs(V1 - V2)
    max_diff = np.max(differences)
    mean_diff = np.mean(differences)
    
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
                action_idx = policy_det[x, y]
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
        tuple: (path, total_reward, steps_to_goal, success)
    """
    x, y = map0.start_point
    path = [(x, y)]
    total_reward = 0
    
    for step in range(max_steps):
        if (x, y) == map0.end_point:
            print(f"{name}: ✓ Reached goal in {step} steps, Total reward: {total_reward:.2f}")
            return path, total_reward, step, True
        
        action_idx = policy_det[x, y]
        action = ACTIONS[action_idx]
        
        x, y = get_next_state(x, y, action)
        path.append((x, y))
        total_reward += reward_calc(x, y) if (x, y) != map0.end_point else NET_GOAL_REWARD
    
    print(f"{name}: ✗ Did not reach goal within {max_steps} steps")
    return path, total_reward, max_steps, False


# ==================== MAIN EXECUTION ====================

def main():
    """
    Main execution: Run both algorithms and compare results.
    """
    print("\n" + "=" * 60)
    print("TASK 1: PLANNING WITH KNOWN ENVIRONMENT MODEL")
    print("=" * 60 + "\n")
    
    # -------- VALUE ITERATION --------
    V_vi, policy_vi, iter_vi = value_iteration(max_iterations=1000, theta=1e-6)
    
    # -------- POLICY ITERATION --------
    V_pi, policy_pi, iter_pi = policy_iteration(max_iterations=1000, eval_theta=1e-6)
    
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
    
    path_vi, reward_vi, steps_vi, success_vi = test_policy(policy_vi, "Value Iteration Policy")
    path_pi, reward_pi, steps_pi, success_pi = test_policy(policy_pi, "Policy Iteration Policy")
    
    print("\n" + "=" * 60)
    print("TASK 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()


