# game/snake_game.py

import pygame
import random
import numpy as np
from config import *

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Snake RL')
        self.clock = pygame.time.Clock()
        
        # Initialize the font for rendering text
        self.font = pygame.font.SysFont('arial', 25)
        
        self.reset()

    def reset(self):
        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        x = SCREEN_WIDTH // 2
        y = SCREEN_HEIGHT // 2
        self.head = [x, y]
        self.snake = [self.head[:], [x - BLOCK_SIZE, y], [x - 2 * BLOCK_SIZE, y]]
        self.score = 0
        self.food = None
        self.frame_iteration = 0  # To prevent infinite loops
        self._place_food()
        return self._get_state()

    def _place_food(self):
        x = random.randrange(0, SCREEN_WIDTH, BLOCK_SIZE)
        y = random.randrange(0, SCREEN_HEIGHT, BLOCK_SIZE)
        self.food = [x, y]
        if self.food in self.snake:
            self._place_food()

    def step(self, action):
        self._handle_events()
        self._move(action)
        reward = -1             # Penalty for each movement
        done = False

        if self._is_collision():
            done = True
            reward = -1000      # Penalty for dying
            return self._get_state(), reward, done

        self.snake.insert(0, self.head[:])

        if self.head == self.food:
            self.score += 1
            reward = 1000       # Reward for eating an apple
            self._place_food()
        else:
            self.snake.pop()

        return self._get_state(), reward, done

    def _move(self, action):
        # action: 0 - straight, 1 - right, 2 - left
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = directions.index(self.direction)

        if action == 1:  # Turn right
            idx = (idx + 1) % 4
        elif action == 2:  # Turn left
            idx = (idx - 1) % 4

        self.direction = directions[idx]

        x, y = self.head
        if self.direction == 'UP':
            y -= BLOCK_SIZE
        elif self.direction == 'DOWN':
            y += BLOCK_SIZE
        elif self.direction == 'LEFT':
            x -= BLOCK_SIZE
        elif self.direction == 'RIGHT':
            x += BLOCK_SIZE

        self.head = [x, y]

    def _get_state(self):
        head_x, head_y = self.head

        # Points to the left, right, up, and down of the head
        point_l = [head_x - BLOCK_SIZE, head_y]
        point_r = [head_x + BLOCK_SIZE, head_y]
        point_u = [head_x, head_y - BLOCK_SIZE]
        point_d = [head_x, head_y + BLOCK_SIZE]

        dir_l = self.direction == 'LEFT'
        dir_r = self.direction == 'RIGHT'
        dir_u = self.direction == 'UP'
        dir_d = self.direction == 'DOWN'

        # Danger straight
        danger_straight = (
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d))
        )

        # Danger right
        danger_right = (
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d))
        )

        # Danger left
        danger_left = (
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d))
        )

        # Food location relative to head
        food_left = self.food[0] < head_x
        food_right = self.food[0] > head_x
        food_up = self.food[1] < head_y
        food_down = self.food[1] > head_y

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down)
        ]

        return np.array(state, dtype=int)

    def _is_collision(self, point=None):
        if point is None:
            point = self.head
        x, y = point
        if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
            return True
        if point in self.snake[1:]:
            return True
        return False

    def render(self):
        self.display.fill(BLACK)

        for block in self.snake:
            pygame.draw.rect(self.display, GREEN, (*block, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, (*self.food, BLOCK_SIZE, BLOCK_SIZE))

        # Render the score text
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, [10, 10])

        pygame.display.flip()
        self.clock.tick(FPS)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
