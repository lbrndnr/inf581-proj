from queue import *
import random

def test():
    print("AAAA")

class environment:
    
    # mov[direction] is the displacement for the snake's head.
    mov = [(0,1), (0,-1), (1,0), (-1,0)]
    
    """
    Snake game on a square maze of size maze_size, snake with initial size of initial_snake_size.
    Instead of a single mouse, there may be multiple mice. Their points are given by mice_points list.
    """
    def __init__(self, maze_size=20, initial_snake_size=5, mice_points=[1,2,3], fixed_obstacles=[(3,4),(3,5),(3,6)]):
        self.score = 0
        self.snake_q = Queue()
        # obstacles are walls + fixed obstacles + the snake itself
        self.obstacles = set()
        for obs in fixed_obstacles:
            self.obstacles.add(obs)
        # maze walls
        for i in range (maze_size):
            self.obstacles.add( (-1       ,i        ) );
            self.obstacles.add( (maze_size,i        ) );
            self.obstacles.add( (i        ,-1       ) );
            self.obstacles.add( (i        ,maze_size) );
        # snake
        for i in range (initial_snake_size):
            self.obstacles.add( (0,i) )
            self.snake_q.put( (0,i) )
        self.head = (0,initial_snake_size-1)
        self.curr_direction = 0
        # mice = dict{ key=coordinate, value=points }
        self.mice = {}
        for i in mice_points:
            # takes care to not place a mouse on the snake or over another mouse
            coord = (0,0)
            while (coord in self.mice or coord in self.obstacles):
                coord = (random.randint(0,maze_size), random.randint(0,maze_size))
            self.mice[coord] = i
    
    # returns true if the game has ended. False othersie.
    def move(self, direction):
        self.head = (self.head[0]+mov[direction][0], self.head[1]+mov[direction][1])
        # checks if snake hits an obstacle
        if self.head in self.obstacles:
            return true
        # adds new position
        self.obstacles.add(self.head)
        self.snake_q.add(self.head)
        self.curr_direction = direction
        # checks if ate a mouse
        if self.head in self.mice:
            points = self.mice[head]
            self.score += points
            # replace eaten mouse in a random position (without another mouse or the snake).
            self.mice.remove[self.head]
            coord = self.head
            while (coord in self.mice or coord in self.obstacles):
                coord = (random.randint(0,maze_size), random.randint(0,maze_size))
            self.mice[coord] = points 
        else:
            # remove tail
            self.obstacles.remove(self.snake_q.get())
        return false
        
    def see_maze(self):
        return (self.obstacles, self.mice, self.head, self.curr_direction, self.score)
