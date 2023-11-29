from src.q_learning_training import *
from src.q_learning_assessment import *

# Define the initial set of sticks
initial_sticks = list(range(1, 13))

# Set Q-learning parameters
num_episodes = int(1e7)  # Total number of training episodes
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Initial exploration rate
min_epsilon = 0.01  # Minimum exploration rate
decay_rate = 0.995  # Decay rate for exploration probability (this is adjusted dynamically)

# Train the agent
q_table, rewards = train_agent(
    initial_sticks, num_episodes, alpha, gamma, epsilon, min_epsilon, decay_rate
)

# Example of Q learning strategy
print("=====================================")
print(f"Assess the performance of the agent after training for {num_episodes:,} episodes")
win_rate, wins, losses = simulate_games(q_table, 100000)
print(f"Win rate: {win_rate*100:.2f}% - Wins: {wins} - Losses: {losses}")
print("=====================================")

print("=====================================")
win_rate, wins, losses = simulate_random_games(100000)
print(f"Assess the performance of a completely random strategy for comparison")
print(
    f"Random strategy - Win rate: {win_rate*100:.2f}% - Wins: {wins} - Losses: {losses}"
)
print("=====================================")
