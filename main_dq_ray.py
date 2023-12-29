import ray
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.rllib.algorithms.dqn import D
from ray.tune.registry import register_env
import gymnasium as gym
from src.dq_learning_training import *
from src.dq_learning_assessment import *

# Function to create the environment
def env_creator(_):
    return StickGameEnv()

# Register the environment
register_env("stick_game_env", env_creator)

# Initialize Ray
ray.init()

# Configure the DQN Trainer
config = DEFAULT_CONFIG.copy()
config["env"] = "stick_game_env"
# Set other configurations as needed

# Initialize the Trainer
trainer = DQNTrainer(config=config)

# Training parameters
NUM_TRAINING_ITERATIONS = 1000
EVALUATION_INTERVAL = 100
checkpoint_dir = "path_to_save_checkpoint"

# Training loop with periodic evaluation
for i in range(NUM_TRAINING_ITERATIONS):
    result = trainer.train()
    print(f"Iteration: {i}, reward: {result['episode_reward_mean']}")

    if i % EVALUATION_INTERVAL == 0:
        win_rate = evaluate_model(trainer, num_episodes=100)
        print(f"Win Rate at iteration {i}: {win_rate}")

# Save the trained model
trainer.save(checkpoint_dir)
