import numpy as np
from environment import *
from geometry import *


def decide(env, state):
    maze, head, tail, direction = state
    maze_size = maze.shape[0]
    
    ds = [head + vectorize(direction, a) for a in actions]
    ds = [1 if env.check_field(p) >= 0 else 0 for p in ds]
    blocked_actions = int("".join(str(e) for e in ds), 2)
    
    mice = np.where(maze > 0)

    return (vec_to_int(mice[0], maze_size), vec_to_int(head, maze_size), blocked_actions)


def train():
    epochs = 500000 # Number of episodes/plays
    epsilon = 1. # E-greedy
    gamma = 1.0 # Discount factor

    maze_size = 10
    maze_area = maze_size**2
    n_m = maze_area #coordinates of one mouse
    n_h = maze_area #position of the head
    n_a = len(actions) #possible actions
    n_d = 2**n_a #directions the snake can take without hitting an obstacle

    dim = (n_m, n_h, n_d, n_a)
    qv_mc_table = np.zeros(dim) #This creates our Q-value look-up table
    sa_count = np.zeros(dim) #Record how many times we've seen a given state-action pair.
    returnSum = 0
    stepSum = 0

    for i in range(epochs):
        env = environment(maze_size=maze_size)
        state = env.state
        ended = False
        episodeReward = 0
    
        vs = []
        va = []
        
        while not ended:
            d = decide(env, state)

            # E-greedy policy
            if (np.random.random() < epsilon or len(vs) == 0):
                act = np.random.randint(0, n_a)
            else:
                act = np.argmax(qv_mc_table[d[0], d[1], d[2],:]) #select the best action

            sa_count[d[0], d[1], d[2], act] +=1
            vs.append(d)
            va.append(act)
            
            state, reward, ended = env.step(act)
            episodeReward += reward
        epsilon = epsilon*0.9999

        # Update Q values of the visited states
        for cs, a in zip(vs,va):
            qv_mc_table[d[0], d[1], d[2],a] = qv_mc_table[d[0], d[1], d[2], a]+ (1./sa_count[d[0], d[1], d[2], a])*(episodeReward - qv_mc_table[d[0], d[1], d[2], a])

        returnSum += episodeReward
        stepSum += len(vs)
        if (i % 100 == 0 and i > 0):
            print("Episode: ", i, "Average Return: ", returnSum/100.0, "Average Steps: ", stepSum/100.0)
            returnSum = 0
            stepSum = 0


train()
