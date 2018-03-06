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
    def __init__(self, maze_size=20, initial_snake_size=5, mice_points=[1,2,3], walls=[]):
        self.maze = np.zeros((maze_size, maze_size), dtype=int)
        self.direction = np.array([-1, 0])
        self.snake = deque()
        for i in range(initial_snake_size):
            p = (int(maze_size/2+i), int(maze_size/2))
            self.maze[p] = SNAKE
            self.snake.append(p)

        for p in walls:
            self.maze[p] = WALL

        for i in mice_points:
            self.add_mouse(i)


    @classmethod
    def random(cls, maze_size, mice_points=[1,2,3], walls=[]):
        env = cls(maze_size=maze_size, initial_snake_size=0, mice_points=[], walls=walls)

        maze_area = maze_size**2
        length = np.random.randint(5, maze_area)
        p = env.random_free_field()
        blocked = False

        a_idx = np.random.randint(0, len(actions))
        a = actions[a_idx]

        env.direction = vectorize(env.direction, a)
        env.snake.append(p)
        env.maze[p] = SNAKE

        while not blocked and length > 0:
            body = [(p[0]+1, p[1]), (p[0]-1, p[1]), (p[0], p[1]-1), (p[0], p[1]+1)]
            body = [p for p in body if env.check_field(p) == 0]
            if len(body) == 0:
                blocked = True
            else:
                idx = np.random.randint(0, len(body))
                p = body[idx]

                env.snake.append(p)
                env.maze[p] = SNAKE
                length -= 1

        for i in mice_points:
            env.add_mouse(i)

        return env

    
    def step(self, action):
        new_direction = vectorize(self.direction, action)
        head = tuple(np.array(self.snake[0], dtype=int) + new_direction)
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
        
        self.direction = new_direction
        return (self.state, reward, ended)
    

    @property
    def state(self):
        head = np.array(self.snake[0], dtype=int)
        tail = np.array(self.snake[-1], dtype=int)
        return (self.maze, head, tail, self.direction)


    def check_field(self, p):
        p = p if type(p) is tuple else tuple(p)
        if p[0] >= self.maze.shape[0] or p[1] >= self.maze.shape[1] or p[0] < 0 or p[1] < 0:
            return WALL
        
        return self.maze[p]


    def random_free_field(self):
        free_fields = np.where(self.maze == 0)
        if len(free_fields[0]) == 0:
            return None

        idx = np.random.randint(0, len(free_fields[0]))
        return (free_fields[0][idx], free_fields[1][idx])


    def add_mouse(self, points):
        p = self.random_free_field()
        if p is None:
            return False

        self.maze[p] = points
        return True


    def maze_string(self):
        n,m = self.maze.shape
        char_map = {WALL: "#", SNAKE: "@", 0: " "}

        desc = (m+2) * "#" + "\n"

        for i in range(n):
            line = "#"
            for j in range(m):
                v = self.maze[i, j]
                c = char_map[v] if v in char_map else str(v)
                line += c

            line += "#"
            desc += line + "\n"
        
        desc += (m+2) * "#" + "\n"
        return desc

        

