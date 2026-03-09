"""
Task 3: Q-Learning with Unknown Transition Model
================================================
This module implements the stochastic environment for Q-learning
where the agent learns online without knowing transition probabilities.

Parameters:
- Grid size: 5 × 5
- Start state: (0, 0)
- Goal state: (4, 4)
- Obstacles: (2, 1), (2, 3)
- Stochastic transitions:
  * 0.8 probability: Execute intended action
  * 0.1 probability each: Execute perpendicular action (left or right)
- Step cost: -1
- Goal reward: +10
- Discount factor: γ = 0.9
- Learning rate: α = 0.1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scene_map as map0
import numpy as np
import random


# ==================== ENVIRONMENT CONSTANTS ====================
STEP_COST = -1
GOAL_REWARD = 10
NET_GOAL_REWARD = GOAL_REWARD + STEP_COST  # +9 for reaching goal
GAMMA = 0.9  # Discount factor
ALPHA = 0.1  # Learning rate for Q-learning
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


def get_next_state_stochastic(x, y, action, intended_prob=0.8):
    """
    Stochastic transition function for Task 3.
    
    With probability:
    - 0.8: Execute intended action
    - 0.1 each: Execute one of two perpendicular actions
    
    Args:
        x, y: Current position
        action: Intended action ('u', 'd', 'l', 'r')
        intended_prob: Probability of executing intended action (default 0.8)
        
    Returns:
        tuple: Next state (x', y')
    """
    rand = random.random()
    
    if rand < intended_prob:
        # Execute intended action (0.8 probability)
        actual_action = action
    else:
        # Execute perpendicular action (0.2 probability split)
        if action in ['u', 'd']:
            actual_action = random.choice(['l', 'r'])
        else:  # action in ['l', 'r']
            actual_action = random.choice(['u', 'd'])
    
    # Get next state based on actual action
    return map0.move_function_at_position(x, y, actual_action)
