import numpy as np
import itertools
import gymnasium as gym
from gymnasium import spaces

class StickGameEnv(gym.Env):
    def __init__(self):
        super(StickGameEnv, self).__init__()
        self.all_combinations = generate_all_combinations()
        self.action_space = spaces.Discrete(len(self.all_combinations))
        self.observation_space = spaces.Box(low=0, high=12, shape=(13,), dtype=int)

        self.state = None
        self.dice_roll = None
        self.reset()

    def reset(self, seed=None):
        self.state = np.ones(12, dtype=int)  # Reset stick state
        self.dice_roll = self._roll_dice()   # Initial dice roll
        return np.append(self.state, self.dice_roll), {}  # Return only the initial observation

    def step(self, action):
        action = int(action)
        combination = self.all_combinations[action]

        if not self._is_valid_action(combination):
            return np.append(self.state, self.dice_roll), -1, False, False, {}

        for stick in combination:
            self.state[stick - 1] = 0

        terminated = bool(np.all(self.state == 0))
        reward = 1 if terminated else 0

        self.dice_roll = self._roll_dice()
        truncated = False  # You can determine if the episode is truncated based on your environment's logic
        return np.append(self.state, self.dice_roll), reward, terminated, truncated, {}



    def _roll_dice(self):
        return np.random.randint(1, 7) + np.random.randint(1, 7) 

    def _is_valid_action(self, combination):
        return all(self.state[stick - 1] == 1 for stick in combination)

    def valid_actions(self):
        dice_roll = np.random.randint(2, 13)
        sticks = [i + 1 for i, present in enumerate(self.state) if present == 1]

        valid = []
        for L in range(1, len(sticks) + 1):
            for subset in itertools.combinations(sticks, L):
                if sum(subset) == dice_roll:
                    subset_action = np.zeros(12, dtype=int)
                    for stick in subset:
                        subset_action[stick - 1] = 1
                    if np.all(subset_action <= self.state):
                        valid.append(subset_action)
        return valid

def generate_all_combinations():
    all_combinations = {}
    action_id = 0

    for dice_roll in range(2, 13):
        for num_sticks in range(1, min(dice_roll + 1, 13)):
            for combination in itertools.combinations(range(1, 13), num_sticks):
                if sum(combination) == dice_roll:
                    all_combinations[action_id] = combination
                    action_id += 1

    return all_combinations