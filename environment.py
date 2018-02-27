from collections import deque
import random

class environment:
    
    # mov[direction] is the displacement for the snake's head.
    mov = [(0,1), (0,-1), (1,0), (-1,0)]
    
    """
    Snake game on a square maze of size maze_size, snake with initial size of initial_snake_size.
    Instead of a single mouse, there may be multiple mice. Their points are given by mice_points list.
    """
    def __init__(self, maze_size=20, initial_snake_size=5, mice_points=[1,2,3], fixed_obstacles=[(3,4),(3,5),(3,6)]):
        self.score = 0
        self.snake_q = deque()
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
            self.snake_q.append( (0,i) )
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
        self.head = (self.head[0]+environment.mov[direction][0], self.head[1]+environment.mov[direction][1])
        # checks if snake hits an obstacle
        if self.head in self.obstacles:
            return True
        # adds new position
        self.obstacles.add(self.head)
        self.snake_q.append(self.head)
        self.curr_direction = direction
        # checks if ate a mouse
        if self.head in self.mice:
            points = self.mice[self.head]
            self.score += points
            # replace eaten mouse in a random position (without another mouse or the snake).
            del self.mice[self.head]
            coord = self.head
            while (coord in self.mice or coord in self.obstacles):
                coord = (random.randint(0,self.maze_size), random.randint(0,self.maze_size))
            self.mice[coord] = points 
        else:
            # remove tail
            self.obstacles.remove(self.snake_q.popleft())
        return False
        
    def see_maze(self):
        return (self.obstacles, self.mice, self.head, self.curr_direction, self.score)

    def print_maze(self):
        print('score =', self.score)
        for i in range(-1, self.maze_size + 1):
            print('')
            for j in range(-1, self.maze_size + 1):
                if (i,j) in self.obstacles:
                    print ('@', end='')
                elif (i,j) in self.mice:
                    print ('m', end='')
                else:
                    print (' ', end='')
        
        
