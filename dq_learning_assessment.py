import numpy as np

def evaluate_dqn_model(model, env, num_episodes=100):
    """
    Simulate a specified number of episodes to evaluate the DQN model.

    Args:
        model: The trained DQN model.
        env: The environment for evaluation.
        num_episodes: Number of episodes to simulate.

    Returns:
        win_rate: The win rate (proportion of episodes won).
    """
    win_count = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action, _ = model.predict(state)  # Choose action using the trained model
            state, reward, done, _ = env.step(action)

        # Check if the episode was won (all sticks removed)
        if np.all(state == 0):
            win_count += 1

    win_rate = win_count / num_episodes
    return win_rate
