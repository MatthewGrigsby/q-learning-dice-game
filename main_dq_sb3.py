from src.dq_learning_training import *
from src.dq_learning_assessment import *
import numpy as np
import itertools
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# [Include your StickGameEnv and generate_all_combinations definitions here]

# Create the environment
env = make_vec_env(lambda: StickGameEnv(), n_envs=1)

# Specify the directory where TensorBoard logs will be saved
tensorboard_log_directory = "./tensorboard_logs/"

# Initialize the DQN model with TensorBoard logging
model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log_directory)

# Number of steps for training
num_steps = int(5e5)

# Train the model
print("Training the model...")
model.learn(total_timesteps=num_steps)

# After training, you can view logs with TensorBoard by running:
# tensorboard --logdir=./tensorboard_logs/

# Number of episodes to evaluate the model
num_eval_episodes = 10000

# Estimate win rate
print("Evaluating the model...")
max_steps_per_episode = 12  # Since there are 12 sticks
total_wins = 0
for i in range(num_eval_episodes):
    obs = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps_per_episode:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        steps += 1

        if done and reward > 0:
            total_wins += 1

    if i % 100 == 0:
            print(f"Step: {steps}, Action: {action}, Reward: {reward}, Done: {done}")

win_rate = total_wins / num_eval_episodes
print(f"Win Rate: {win_rate}")
