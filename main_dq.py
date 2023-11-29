import gym
from src.dq_learning_training import StickGameEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Create and wrap the environment
env = StickGameEnv()
env = make_vec_env(lambda: env, n_envs=1)

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=1e6)

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")