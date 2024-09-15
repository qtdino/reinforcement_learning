<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
</head>
<body>

<h1><strong>Reinforcement Learning Snake Game</strong></h1>

<p>A Python implementation of the classic Snake game, where the snake is controlled by a Deep Q-Network (DQN) reinforcement learning agent. Built with PyTorch and Pygame.</p>

<h2><strong>Table of Contents</strong></h2>

<ul>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#dqn-algorithm">Understanding the DQN Algorithm</a></li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#training-the-model">Training the Model</a></li>
    <li><a href="#testing-the-model">Testing the Model</a></li>
    <li><a href="#understanding-the-plots">Understanding the Plots</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#license">License</a></li>
</ul>

<hr>

<h2 id="introduction"><strong>Introduction</strong></h2>

<p>This project demonstrates how reinforcement learning techniques can be applied to teach an AI agent to play the Snake game. The agent uses a Deep Q-Network to learn optimal strategies through trial and error, improving over time as it interacts with the game environment.</p>

<hr>

<h2 id="dqn-algorithm"><strong>Understanding the DQN Algorithm</strong></h2>

The Deep Q-Network (DQN) algorithm combines Q-learning with deep neural networks to approximate the optimal action-value function $Q^*(s, a)$, which represents the maximum expected future reward achievable from a state-action pair $(s, a)$.

### **DQN Update Rule**

The core of the DQN algorithm involves updating the network parameters $\theta$ to minimize the loss function. The parameters from the `config.py` file play crucial roles in this process.

#### **Loss Function**

The loss function $L(\theta)$ for the network parameters $\theta$ is defined as:

$$
L(\theta) = \frac{1}{\text{BATCHSIZE}} \sum_{i=1}^{\text{BATCHSIZE}} \left( y_i - Q(s_i, a_i; \theta) \right)^2
$$

where:

$$
y_i = r_i + \gamma \max_{a'} Q(s_{i+1}, a'; \theta^{-})
$$

- $\gamma$ is the **discount factor**, defined in `config.py` as `DISCOUNT_FACTOR`.
- `BATCH_SIZE` is the size of the mini-batch sampled from the replay memory.

#### **Gradient Descent Update**

The network parameters are updated using gradient descent with the **learning rate** $\alpha$ (from `config.py` as `LEARNING_RATE`):

$$
\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)
$$

#### **Epsilon-Greedy Policy**

The agent selects actions using an epsilon-greedy policy, where the probability of choosing a random action (exploration) is $\epsilon$:

- $\epsilon$ starts at `EPSILON_START` and decays over time to `EPSILON_END` using the decay rate `EPSILON_DECAY`.

#### **Epsilon Decay Formula**

The decay of $\epsilon$ after each episode can be modeled as:

$$
\epsilon \leftarrow \max(\epsilon_{\text{end}}, \epsilon \times \epsilon_{\text{decay}})
$$

where:

- $\epsilon_{\text{end}}$ is `EPSILON_END`.
- $\epsilon_{\text{decay}}$ is `EPSILON_DECAY`.

### **Explanation of Config Parameters in the Algorithm**

- **Learning Rate ($\alpha$):**

  - Determines the step size during gradient descent optimization.
  - Affects how quickly the network learns; too high can cause instability, too low can slow down learning.

- **Discount Factor ($\gamma$):**

  - Balances the importance of immediate and future rewards.
  - A value close to 1 emphasizes future rewards, while a value close to 0 emphasizes immediate rewards.

- **Batch Size (`BATCH_SIZE`):**

  - Number of experience samples used in each training iteration.
  - Larger batch sizes provide more stable gradients but require more memory.

- **Epsilon Parameters (`EPSILON_START`, `EPSILON_END`, `EPSILON_DECAY`):**

  - Control the exploration-exploitation trade-off.
  - `EPSILON_START`: Initial exploration rate.
  - `EPSILON_END`: Minimum exploration rate.
  - `EPSILON_DECAY`: Rate at which exploration decreases.

<hr>

<h2 id="prerequisites"><strong>Prerequisites</strong></h2>

<ul>
    <li><strong>Python 3.6 or higher</strong></li>
    <li><strong>Git</strong> (for cloning the repository)</li>
    <li><strong>pip</strong> (Python package installer)</li>
</ul>

<hr>

<h2 id="installation"><strong>Installation</strong></h2>

<h3><strong>1. Clone the Repository</strong></h3>

<p>Clone the repository to your local machine using the following command:</p>

<pre><code>git clone https://github.com/qtdino/snake_learning.git
</code></pre>

<p>Navigate to the project directory:</p>

<pre><code>cd snake_learning
</code></pre>

<h3><strong>2. Create a Virtual Environment (Recommended)</strong></h3>

<p>Creating a virtual environment is recommended to manage project dependencies separately from your system packages.</p>

<p><strong>On Windows:</strong></p>

<pre><code>python -m venv venv
venv\Scripts\activate
</code></pre>

<p><strong>On macOS/Linux:</strong></p>

<pre><code>python3 -m venv venv
source venv/bin/activate
</code></pre>

<h3><strong>3. Install Dependencies</strong></h3>

<p>Install the required Python packages using the <code>requirements.txt</code> file:</p>

<pre><code>pip install -r requirements.txt
</code></pre>

<hr>

<h2 id="configuration"><strong>Configuration</strong></h2>

<p>Configuration parameters for the game and the agent are stored in the <code>config.py</code> file. You can modify this file to adjust settings such as learning rate, discount factor, exploration rate, and more.</p>

<h3><strong>Key Parameters in <code>config.py</code>:</strong></h3>

<ul>
    <li><strong>Game Settings:</strong>
        <ul>
            <li><code>SCREEN_WIDTH</code>, <code>SCREEN_HEIGHT</code>: Dimensions of the game window.</li>
            <li><code>BLOCK_SIZE</code>: Size of each block in the grid.</li>
            <li><code>FPS</code>: Frames per second (game speed).</li>
        </ul>
    </li>
    <li><strong>Agent Settings:</strong>
        <ul>
            <li><code>LEARNING_RATE</code>: Learning rate for the optimizer.</li>
            <li><code>DISCOUNT_FACTOR</code>: Discount factor for future rewards.</li>
            <li><code>MEMORY_SIZE</code>: Maximum size of the replay memory.</li>
            <li><code>BATCH_SIZE</code>: Mini-batch size for training.</li>
            <li><code>EPSILON_START</code>, <code>EPSILON_END</code>, <code>EPSILON_DECAY</code>: Parameters for the epsilon-greedy policy.</li>
        </ul>
    </li>
    <li><strong>Training Settings:</strong>
        <ul>
            <li><code>NUM_EPISODES</code>: Total number of training episodes.</li>
            <li><code>TARGET_UPDATE</code>: Frequency to update the target network.</li>
            <li><code>MODEL_SAVE_INTERVAL</code>: Frequency to save the model checkpoint.</li>
        </ul>
    </li>
</ul>

<h3><strong>Example: Changing the Number of Training Episodes</strong></h3>

<p>To change the number of training episodes to 2000, modify the following line in <code>config.py</code>:</p>

<pre><code>NUM_EPISODES = 2000  # Increase from the default value
</code></pre>

<hr>

<h2 id="training-the-model"><strong>Training the Model</strong></h2>

<p>To train the reinforcement learning agent, run the <code>main.py</code> script without any arguments:</p>

<pre><code>python main.py
</code></pre>

<h3><strong>What Happens During Training:</strong></h3>

<ul>
    <li>The agent plays the Snake game, learning from its actions using the DQN algorithm.</li>
    <li>Training progress is displayed in the terminal, showing the episode number, score, total reward, mean score, and epsilon value.</li>
    <li>A plot of the training progress is generated and saved at the end of training.</li>
</ul>

<p><strong>Note:</strong> Training may take some time depending on the number of episodes and your computer's performance.</p>

<hr>

<h2 id="testing-the-model"><strong>Testing the Model</strong></h2>

<p>After training, you can watch the trained agent play the game:</p>

<pre><code>python main.py --play
</code></pre>

<h3><strong>What to Expect:</strong></h3>

<ul>
    <li>A game window will open, and the agent will play the Snake game autonomously.</li>
    <li>The current score is displayed on the game window.</li>
    <li>You can observe the agent's performance based on the training it received.</li>
</ul>

<hr>

<h2 id="understanding-the-plots"><strong>Understanding the Plots</strong></h2>

<p>After training is complete, a plot of the training progress is saved in the <code>logs/</code> directory as <code>training_progress.png</code>.</p>

<h3><strong>Plot Details:</strong></h3>

<ul>
    <li><strong>Score per Episode:</strong> Shows the total reward or score the agent achieved in each episode.</li>
    <li><strong>Mean Score:</strong> Displays the running average score over episodes, helping you see the overall trend.</li>
</ul>

<h3><strong>Viewing the Plot:</strong></h3>

<ul>
    <li>Navigate to the <code>logs/</code> directory:</li>
</ul>

<pre><code>cd logs
</code></pre>

<ul>
    <li>Open <code>training_progress.png</code> using an image viewer.</li>
</ul>

<hr>

<h2 id="project-structure"><strong>Project Structure</strong></h2>

<pre><code>reinforcement_learning/
├── agent/
│   ├── __init__.py
│   ├── dqn_agent.py
│   ├── model.py
│   └── replay_memory.py
├── assets/
│   └── .gitkeep
├── checkpoints/
│   └── .gitkeep
├── config.py
├── game/
│   ├── __init__.py
│   └── snake_game.py
├── logs/
│   └── .gitkeep
├── main.py
├── README.md
├── requirements.txt
└── utils/
    ├── __init__.py
    └── plot.py
</code></pre>

<h3><strong>Key Directories and Files:</strong></h3>

<ul>
    <li><strong><code>agent/</code>:</strong> Contains the DQN agent implementation.</li>
    <li><strong><code>game/</code>:</strong> Contains the Snake game environment code.</li>
    <li><strong><code>utils/</code>:</strong> Utility functions for plotting and other helper methods.</li>
    <li><strong><code>config.py</code>:</strong> Configuration parameters for the game and agent.</li>
    <li><strong><code>main.py</code>:</strong> Main script to train or test the agent.</li>
    <li><strong><code>requirements.txt</code>:</strong> Lists all Python dependencies.</li>
    <li><strong><code>checkpoints/</code>:</strong> Directory where model checkpoints are saved.</li>
    <li><strong><code>logs/</code>:</strong> Directory where training logs and plots are saved.</li>
    <li><strong><code>assets/</code>:</strong> Contains assets like images (currently unused but reserved for future use).</li>
</ul>

<hr>

<h2 id="acknowledgments"><strong>Acknowledgments</strong></h2>

<p>This project was developed to demonstrate the application of reinforcement learning using Deep Q-Networks in a classic game environment.</p>

<p><strong>Enjoy experimenting with the Reinforcement Learning Snake Game! If you have any questions or suggestions, feel free to open an issue or submit a pull request.</strong></p>

</body>
</html>
