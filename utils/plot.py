# utils/plot.py

import matplotlib.pyplot as plt
import os
from config import LOG_DIR

def plot_scores(scores, mean_scores):
    # Ensure the logs directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores, label='Score per Episode')
    plt.plot(mean_scores, label='Mean Score')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file in the logs directory
    plot_path = os.path.join(LOG_DIR, 'training_progress.png')
    plt.savefig(plot_path)
    plt.close()
