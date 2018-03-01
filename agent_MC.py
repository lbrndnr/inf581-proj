import numpy as np
from environment import *
from geometry import *
    
# the size of the maze region accessible from the snake's head
# (i.e.: that the snake can go to without hitting an obstacle).
def size_of_accessible_region(state):
    return 10
    # def DFS(tmp_obstacles, coord):
    #     if coord in tmp_obstacles:
    #         return 0
    #     size = 1
    #     # mark as visited
    #     tmp_obstacles.add(coord)
    #     # for each direction, visit adjacent cell
    #     for dir in range(4):
    #         size += DFS( tmp_obstacles, (coord[0]+mov[dir][0],coord[1]+mov[dir][1]) )
    #     return size
    
    # obstacles, mice, head, curr_direction, terminated = state        
    # tmp = obstacles.copy()
    # tmp.remove(head)
    # return DFS(tmp, head)


def decide(env, state):
    maze, head, tail = state
    maze_size = maze.shape[0]
    
    ds = [head + vectorize(d) for d in directions]
    ds = [1 if env.check_field(p) >= 0 else 0 for p in ds]
    
    mice = np.where(maze > 0)

    return (vec_to_int(mice[0], maze_size), vec_to_int(head, maze_size), ds)
    

epochs = 500000 # Number of episodes/plays
epsilon = 1. # E-greedy
gamma = 1.0 # Discount factor

maze_size = 20
maze_area = maze_size**2
n_m = maze_area #coordinates of one mouse
n_h = maze_area #position of the head
n_d = len(directions) #possible directions of the head

dim = (n_m, n_h, n_d, n_d)
qv_mc_table = np.zeros(dim) #This creates our Q-value look-up table
sa_count    = np.zeros(dim) #Record how many times we've seen a given state-action pair.
returnSum   = 0

for i in range(epochs):
    env = environment(maze_size=maze_size)
    state = env.state
    ended = False
    reward = 0
   
    vs = []
    va = []
    
    while not ended:
        d = decide(env, state)

        # E-greedy policy
        if (np.random.random() < epsilon or len(vs) == 0):
            act = np.random.randint(0, n_d)
        else:
            act = np.argmax(qv_mc_table[d[0], d[1], d[2],:]) #select the best action

        sa_count[d[0], d[1], d[2], act] +=1
        vs.append(d)
        va.append(act)
        
        state, reward, ended = env.step(act)
    epsilon = epsilon*0.9999

    # Update Q values of the visited states
    for cs, a in zip(vs,va):
        qv_mc_table[d[0], d[1], d[2],a] = qv_mc_table[d[0], d[1], d[2], a]+ (1./sa_count[d[0], d[1], d[2], a])*(reward - qv_mc_table[d[0], d[1], d[2], a])

    returnSum = returnSum + reward
    if (i % 100 == 0 and i > 0):    
       print("Episode: ", i, "Average Return: ", returnSum/ float(i))
       returnSum = 0
