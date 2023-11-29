import gym
import numpy as np
import itertools
from gym import spaces
import itertools

class StickGameEnv(gym.Env):
    def __init__(self):
        super(StickGameEnv, self).__init__()
        self.all_combinations = generate_all_combinations()
        self.action_space = spaces.Discrete(len(self.all_combinations))
        self.observation_space = spaces.MultiBinary(12)
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.ones(12, dtype=int)
        return self.state

    def step(self, action):
        action = int(action)
        combination = self.all_combinations[action]
        if not self._is_valid_action(combination):
            return self.state, 0, True, {}  # Invalid action, end the game

        for stick in combination:
            self.state[stick] = 0

        done = np.all(self.state == 0)
        reward = 1 if done else 0

        return self.state, reward, done, {}

    def _is_valid_action(self, combination):
        return all(self.state[stick] == 1 for stick in combination)

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
        for num_sticks in range(1, dice_roll + 1):
            for combination in itertools.combinations(range(12), num_sticks):
                if sum(combination) + len(combination) == dice_roll:
                    all_combinations[action_id] = combination
                    action_id += 1

    return all_combinations

all_combinations = generate_all_combinations()
