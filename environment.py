from queue import *
import random

class environment:
    
    # mov[direction] is the displacement for the snake's head.
    mov = [(0,1), (0,-1), (1,0), (-1,0)]
    
    """
    Snake game on a square maze of size maze_size, snake with initial size of initial_snake_size.
    Instead of a single mouse, there may be multiple mice. Their points are given by mice_points list.
    """
    def __init__(self, maze_size=10, initial_snake_size=5, mice_points=[1,2,3]):
        score = 0
        snake_q = Queue()
        # obstacles are walls + the snake itself
        obstacles = set()
        # maze walls
        for i in range (maze_size):
            obstacles.add( (-1       ,i        ) );
            obstacles.add( (maze_size,i        ) );
            obstacles.add( (i        ,-1       ) );
            obstacles.add( (i        ,maze_size) );
        # snake
        for i in range (initial_snake_size):
            obstacles.add( (0,i) )
            snake_q.put( (0,i) )
        head = (0,initial_snake_size-1)
        curr_direction = 0
        # mice = dict{ key=coordinate, value=points }
        mice = {}
        for i in mice_points:
            # takes care to not place a mouse on the snake or over another mouse
            coord = (0,0)
            while (coord in mice or coord in self.obstacles):
                coord = (random.randint(0,maze_size), random.randint(0,maze_size))
            mice[coord] = i
    
    # returns true if the game has ended. False othersie.
    def move(direction):
        # remove tail
        obstacles.remove(snake_q.get())
        self.head = (head[0]+mov[direction][0], head[1]+mov[direction][1])
        # checks if snake hits an obstacle
        if head in obstacles:
            return true
        # adds new position
        obstacles.add(head)
        snake_q.add(head)
        curr_direction = direction
        # checks if ate a mouse
        if head in mice:
            points = mice[head]
            score += points
            # replace eaten mouse in a random position (without another mouse or the snake).
            mice.remove[head]
            coord = head
            while (coord in mice or coord in obstacles):
                coord = (random.randint(0,maze_size), random.randint(0,maze_size))
            mice[coord] = points 
        return false
        
    def see_maze():
        return obstacles, mice
