# config.py

import os

# Game Settings
SCREEN_WIDTH = 640          # Width of the game window
SCREEN_HEIGHT = 480         # Height of the game window
BLOCK_SIZE = 20             # Size of the grid block (snake and food)
FPS = 15                    # Frames per second (game speed)

# Colors (RGB Format)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)           # Food color
GREEN = (0, 200, 0)         # Snake color

# Agent Settings
LEARNING_RATE = 0.001       # Learning rate for the optimizer
DISCOUNT_FACTOR = 0.90      # Discount factor for future rewards (gamma)
MEMORY_SIZE = 100_000       # Maximum size of the replay memory
BATCH_SIZE = 64             # Mini-batch size for training
EPSILON_START = 1.0         # Starting value of epsilon for exploration
EPSILON_END = 0.01          # Minimum value of epsilon
EPSILON_DECAY = 0.995       # Decay rate of epsilon per episode

# Training Settings
NUM_EPISODES = 1000         # Total number of training episodes
MAX_STEPS_PER_EPISODE = 1000  # Maximum steps per episode
TARGET_UPDATE = 10          # Frequency (in episodes) to update the target network
MODEL_SAVE_INTERVAL = 100   # Frequency (in episodes) to save the model checkpoint

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Model Paths
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'model_final.pth')

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
