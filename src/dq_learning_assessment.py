from src.dq_learning_training import *

def evaluate_model(model, num_episodes=100):
    env = StickGameEnv()
    win_count = 0

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            if done and rewards == 1:
                win_count += 1
        if episode % 100 == 0:
            print(f"Episode {episode} - Win count: {win_count}")

    win_rate = win_count / num_episodes
    return win_rate

