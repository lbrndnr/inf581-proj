import numpy as np
from collections import deque
from geometry import *

SNAKE = -1
WALL = -2

class environment:
    
    """
    Snake game on a square maze of size maze_size, snake with initial size of initial_snake_size.
    Instead of a single mouse, there may be multiple mice. Their points are given by mice_points list.
    """
    def __init__(self, maze_size=20, initial_snake_size=5, mice_points=[1,2,3]):
        self.maze = np.zeros((maze_size, maze_size), dtype=int)

        self.snake = deque()
        for i in range (initial_snake_size):
            p = (int(maze_size/2+i), int(maze_size/2))
            self.maze[p] = SNAKE
            self.snake.append(p)

        for i in mice_points:
            self.add_mouse(i)
        
    
    def step(self, direction):
        head = tuple(np.array(self.snake[0], dtype=int) + vectorize(direction))
        ended = False
        reward = 0

        if self.check_field(head) < 0:
            ended = True
            reward = -1
        else:
            reward = self.maze[head]
            self.maze[head] = SNAKE
            self.snake.appendleft(head)

            if reward == 0:
                self.maze[self.snake.pop()] = 0
            
            self.add_mouse(reward)
        
        return (self.state, reward, ended)
    

    @property
    def state(self):
        head = np.array(self.snake[0], dtype=int)
        tail = np.array(self.snake[-1], dtype=int)
        return (self.maze, head, tail)


    def check_field(self, p):
        p = p if type(p) is tuple else tuple(p)
        if p[0] >= self.maze.shape[0] or p[1] >= self.maze.shape[1] or p[0] < 0 or p[1] < 0:
            return WALL
        
        return self.maze[p]

    # infinite loop if snake gets too long
    def add_mouse(self, points):
        select = lambda: tuple(np.random.random_integers(0, self.maze.shape[0]-1, 2))
        p = select()

        while (self.maze[p] != 0):
            p = select()

        self.maze[p] = points


