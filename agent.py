from environment import environment

# mov[direction] is the displacement for the snake's head.
mov = [(0,1), (0,-1), (1,0), (-1,0)]
    
# the size of the maze region accessible from the snake's head
# (i.e.: that the snake can go to without hitting an obstacle).
def size_of_accessible_region(state):
    def DFS(tmp_obstacles, coord):
        if coord in tmp_obstacles:
            return 0
        size = 1
        # mark as visited
        tmp_obstacles.add(coord)
        # for each direction, visit adjacent cell
        for dir in range(4):
            size += DFS( tmp_obstacles, (coord[0]+mov[dir][0],coord[1]+mov[dir][1]) )
        return size
    
    obstacles, mice, head, curr_direction = state        
    tmp = obstacles.copy()
    tmp.remove(head)
    return DFS(tmp, head)

def reward(state):
    obstacles, mice, head, curr_direction, score = state
    return 10 * score + size_of_accessible_region(state)

env = environment()



# just to test
state = env.see_maze()
obstacles, mice, head, curr_direction, score = state
#print(obstacles)
print(mice)
print(head)
print(curr_direction)

print(size_of_accessible_region(state))
