from environment import *

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

maze_size = 20
epochs = 500000 # Number of episodes/plays
epsilon = 1. # E-greedy
gamma = 1.0 # Discount factor 

dim = (21,10,2,3)
qv_mc_table = np.zeros(dim) #This creates our Q-value look-up table
sa_count    = np.zeros(dim) #Record how many times we've seen a given state-action pair.
returnSum   = 0

env = environment(maze_size=maze_size)

for i in range(epochs):
    state  = env.reset() # Initialize new game; observe current state
    endsim = False
    reward = 0
   
    vs = []
    va = []
    while not endsim:
        # E-greedy policy
        if (np.random.random() < epsilon or len(vs) == 0):
            act = np.random.randint(0,3)
        else:
            act = np.argmax(qv_mc_table[state[0]-1, state[1]-1, int(state[2]),:]) #select the best action

        sa_count[state[0]-1, state[1]-1, int(state[2]),act] +=1
        vs.append(state)
        va.append(act)
        
        state, reward, endsim, info = env.step(act)
    epsilon = epsilon*0.9999
    # Update Q values of the visited states
    for s, a in zip(vs,va):
        qv_mc_table[s[0]-1, s[1]-1, int(s[2]),a] = qv_mc_table[s[0]-1, s[1]-1, int(s[2]),a]+ (1./sa_count[s[0]-1, s[1]-1, int(s[2]),a])*(reward - qv_mc_table[s[0]-1, s[1]-1, int(s[2]),a])

    returnSum = returnSum + reward
    if (i % 100 == 0 and i > 0):    
       print "Episode: ", i, "Average Return: ", returnSum/ float(i)
       returnSum = 0


# just to test
state = env.see_maze()
obstacles, mice, head, curr_direction, score = state
#print(obstacles)
print(mice)
print(head)
print(curr_direction)

print(size_of_accessible_region(state))
