"""
Task 1: Planning with Known Transition Model
============================================
This module implements two planning algorithms for the grid world:
1. Value Iteration
2. Policy Iteration

The agent KNOWS the stochastic transition probabilities:
- 0.8 probability: Execute intended action
- 0.1 probability each: Execute one of two perpendicular actions

Parameters:
- Grid size: 5 × 5
- Start state: (0, 0)
- Goal state: (4, 4)
- Obstacles: (2, 1), (2, 3)
- Step cost: -1
- Goal reward: +10 (total = 10 - 1 = 9 net reward for reaching goal)
- Discount factor: γ = 0.9
- Transition probabilities: P_intended=0.8, P_perp=0.1 each
"""

import scene_map as map0
import numpy as np


# ==================== ENVIRONMENT CONSTANTS ====================
STEP_COST = -1
GOAL_REWARD = 10
NET_GOAL_REWARD = GOAL_REWARD + STEP_COST  # +9 for reaching goal
GAMMA = 0.9  # Discount factor
ACTIONS = ['u', 'd', 'l', 'r']  # Up, Down, Left, Right
ACTION_MAP = {'u': 0, 'd': 1, 'l': 2, 'r': 3}


def reward_calc(x=0, y=0):
    """
    Calculate the immediate reward for reaching state (x, y).
    
    Args:
        x, y: Grid coordinates
        
    Returns:
        float: Reward for this transition
            - NET_GOAL_REWARD (+9) if reaching goal state
            - STEP_COST (-1) for any other step
    """
    if (x, y) == map0.end_point:
        return NET_GOAL_REWARD
    else:
        return STEP_COST


# Transition probabilities (known to the agent in Task 1)
P_INTENDED = 0.8
P_PERPENDICULAR = 0.1  # Each of the two perpendicular directions


def get_next_state(x, y, action):
    """
    Deterministic transition: execute action exactly as intended.
    
    Args:
        x, y: Current position
        action: Action to take ('u', 'd', 'l', 'r')
        
    Returns:
        tuple: Next state (x', y')
    """
    return map0.move_function_at_position(x, y, action)


def get_perpendicular_actions(action):
    """
    Get the two perpendicular actions for a given action.
    
    Args:
        action: Reference action ('u', 'd', 'l', 'r')
        
    Returns:
        list: Two perpendicular actions
    """
    if action in ['u', 'd']:
        return ['l', 'r']
    else:
        return ['u', 'd']


def get_transition_outcomes(x, y, action):
    """
    Return all possible (probability, next_state) outcomes for taking
    an action in the stochastic environment.
    
    The agent knows:
      - 0.8: intended action executes
      - 0.1: slip to perpendicular direction 1
      - 0.1: slip to perpendicular direction 2
    
    Args:
        x, y: Current position
        action: Intended action ('u', 'd', 'l', 'r')
        
    Returns:
        list of (prob, (nx, ny)): Three transitions
    """
    perp = get_perpendicular_actions(action)
    return [
        (P_INTENDED, map0.move_function_at_position(x, y, action)),
        (P_PERPENDICULAR, map0.move_function_at_position(x, y, perp[0])),
        (P_PERPENDICULAR, map0.move_function_at_position(x, y, perp[1])),
    ]


def calculate_final_reward(path):
    """
    Calculate the total discounted reward for a given path.
    
    Args:
        path: List of (x, y) positions visited
        
    Returns:
        float: Total discounted reward
    """
    total_reward = 0
    for step, position in enumerate(path):
        discounted_reward = (GAMMA ** step) * reward_calc(position[0], position[1])
        total_reward += discounted_reward
    return total_reward