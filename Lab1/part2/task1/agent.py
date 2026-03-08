import scene_map as map0
import numpy as np


instant_reward = -1
end_reward = 10


# a 3 dimensional tensor: x,y,probablity of taking action A at position (x,y)
# A in {u,d,l,r} which stands for up, down, left and right respectively.
action_dict_initially = np.ones((5, 5, 4)) / 4


def reward_calc(x=0, y=0):
    if (x,y) == map0.end_point:
        return end_reward - instant_reward
    else:
        return instant_reward



def calculate_final_reward(path):
    total_reward = 0
    for position in path:
        total_reward += reward_calc(position[0], position[1])
    return total_reward