# main.py

import argparse
from agent.dqn_agent import DQNAgent
from game.snake_game import SnakeGame
from utils.plot import plot_scores
import torch

def train():
    game = SnakeGame()
    agent = DQNAgent()
    total_rewards = []
    mean_rewards = []
    cumulative_reward = 0

    total_scores = []  # List to store the game scores (number of apples eaten)
    mean_scores = []
    cumulative_score = 0

    for episode in range(1, NUM_EPISODES + 1):
        state = game.reset()
        done = False
        episode_reward = 0  # Total reward accumulated in the episode

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = game.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            episode_reward += reward  # Accumulate reward

        agent.update_epsilon()

        # Get the game score (number of apples eaten)
        episode_score = game.score

        total_rewards.append(episode_reward)
        cumulative_reward += episode_reward
        mean_reward = cumulative_reward / episode
        mean_rewards.append(mean_reward)

        total_scores.append(episode_score)
        cumulative_score += episode_score
        mean_score = cumulative_score / episode
        mean_scores.append(mean_score)

        # Determine the color based on the episode_score
        if episode_score <= 0:
            color_code = '\033[91m'  # Red
        else:
            color_code = '\033[92m'  # Green

        reset_code = '\033[0m'

        print(f'Episode {episode}, Score: {color_code}{episode_score}{reset_code}, Total Reward: {episode_reward}')

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        if episode % MODEL_SAVE_INTERVAL == 0:
            agent.save_model(f'checkpoints/model_{episode}.pth')

    # After training, plot the scores (number of apples eaten)
    plot_scores(total_scores, mean_scores)
    

def play():
    game = SnakeGame()
    agent = DQNAgent()
    agent.load_model('checkpoints/model_final.pth')
    agent.epsilon = 0

    state = game.reset()
    done = False

    while not done:
        action = agent.get_action(state)
        state, _, done = game.step(action)
        game.render()

if __name__ == "__main__":
    from config import NUM_EPISODES, TARGET_UPDATE, MODEL_SAVE_INTERVAL

    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help='Watch the trained agent play')
    args = parser.parse_args()

    if args.play:
        play()
    else:
        train()
