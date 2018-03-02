import numpy as np
from environment import *
from geometry import *


def compute_direction(a, b):
    diff = b - a
    if abs(diff[0]) > abs(diff[1]):
        if diff[0] > 0:
            return 0 #up
        else:
            return 1 #down
    else:
        if diff[1] > 0:
            return 2 #right
        else:
            return 3 #left


def decide(env, state):
    maze, head, tail, direction = state
    maze_size = maze.shape[0]
    
    ds = [head + vectorize(direction, a) for a in actions]
    ds = [1 if env.check_field(p) >= 0 else 0 for p in ds]
    blocked_actions = int("".join(str(e) for e in ds), 2)
    
    mice = np.where(maze > 0)
    mice = np.transpose(mice)
    mice_directions = [compute_direction(head, m) for m in mice]

    return (mice_directions[0], blocked_actions)

    return tuple(mice_directions + [blocked_actions])


def train_MC():
    epochs = 500000 # Number of episodes/plays
    epsilon = 1. # E-greedy

    maze_size = 10
    n_c = 4 # direction of the mice
    n_a = len(actions) #possible actions
    n_d = 2**n_a #directions the snake can take without hitting an obstacle

    dim = tuple(1*[n_c] + [n_d, n_a])
    qv = np.zeros(dim) #This creates our Q-value look-up table
    sa_count = np.zeros(dim) #Record how many times we've seen a given state-action pair.
    returnSum = 0
    stepSum = 0

    for i in range(epochs):
        env = environment(maze_size=maze_size)
        state = env.state
        ended = False
        episodeReward = 0
    
        ds = []
        
        while not ended:
            d = decide(env, state)

            # E-greedy policy
            if (np.random.random() < epsilon or np.count_nonzero(qv[d[0], d[1],:]) == 0):
                act = np.random.randint(0, n_a)
            else:
                act = np.argmax(qv[d[0], d[1],:]) #select the best action

            d = tuple(list(d) + [act]) #append the chosen action to the decision

            sa_count[d] +=1
            ds.append(d)
            
            state, reward, ended = env.step(act)
            episodeReward += reward
        
        epsilon = epsilon*0.9999

        # Update Q values of the visited states
        for d in ds:
            qv[d] = qv[d]+ (1./sa_count[d])*(episodeReward - qv[d])

        returnSum += episodeReward
        stepSum += len(ds)
        if (i % 100 == 0 and i > 0):
            print("Episode: ", i, "Average Return: ", returnSum/100.0, "Average Steps: ", stepSum/100.0)
            returnSum = 0
            stepSum = 0


def train_QL():
    epochs = 500000 # Number of episodes/plays
    epsilon = 1. # E-greedy
    gamma = 1.0
    alpha = 0.1

    maze_size = 10
    n_c = 4 # direction of the mice
    n_a = len(actions) #possible actions
    n_d = 2**n_a #directions the snake can take without hitting an obstacle

    dim = tuple(1*[n_c] + [n_d, n_a])
    qv = np.zeros(dim) #This creates our Q-value look-up table
    returnSum = 0
    stepSum = 0

    for i in range(epochs):
        env = environment(maze_size=maze_size)
        state = env.state
        ended = False
        
        while not ended:
            d = decide(env, state)

            # E-greedy policy
            if (np.random.random() < epsilon or np.count_nonzero(qv[d[0], d[1],:]) == 0):
                act = np.random.randint(0, n_a)
            else:
                act = np.argmax(qv[d[0], d[1],:]) #select the best action

            d = tuple(list(d) + [act]) #append the chosen action to the decision
            
            state_new, reward, ended = env.step(act)

            q_next = 0 if ended else np.max(qv[d[0], d[1],:])
            qv[d] += alpha*(reward + gamma*q_next  -  qv[d])
            state = state_new
            returnSum += reward
            stepSum += 1  

        epsilon = epsilon*0.9999
        if (i % 100 == 0 and i > 0):
            print("Episode: ", i, "Average Return: ", returnSum/100.0, "Average Steps: ", stepSum/100.0)
            returnSum = 0
            stepSum = 0


train_QL()
