from environment import *
import numpy as np

# mov[direction] is the displacement for the snake's head.
mov = [(0,1), (0,-1), (1,0), (-1,0)]

def calculate_mean_reward(R, T, policy):
    if(len(policy.shape)==1):
        n_s, n_a, n_s = T.shape
        policy = vectorize_policy(policy,n_s,n_a)
    n_s, n_a, n_s = T.shape
    mR = np.zeros(n_s)
    for i in range(n_s):
        for a in range(n_a):
            k = 0
            for sn in range(n_s):
                k = T[i,a,sn]*R[i,a,sn]
            mR[i] += policy[i,a]*k
    return mR

#calculate the transition probability under the given policy
def calculate_mean_transition(T, policy):
    if(len(policy.shape)==1):
        policy = vectorize_policy(policy,nS,nA)

    n_s, n_a, n_s = T.shape
    mT = np.zeros((n_s,n_s)) 
    for i in range(n_s):
        for a in range(n_a):
            for sn in range(n_s):
                mT[i,sn] += policy[i,a]*T[i,a,sn]
    return mT 

def policy_evaluation(R, T, policy, k=10000, gamma=1.0):
    #calculate mean reward and the mean transition matrix
    mean_R = calculate_mean_reward(R, T, policy)
    mean_T = calculate_mean_transition(T, policy)
    #initializes value function to 0
    value_function = np.zeros(mean_R.shape)

    #iterate k times the Bellman Equation
    for i in range(k):
        value_function = mean_R + gamma * np.dot(mean_T, value_function)
    print(value_function)
    return value_function

def policy_improvement(R,T,V,gamma=1.0):
    n_s,n_a,n_s = T.shape
    policy = np.zeros((n_s,n_a))
   
    for s in range(n_s):
        Q = np.zeros(n_a)
        for a in range(n_a):
            for ns in range(n_s):
                Q[a] += T[s,a,ns]*R[s,a,ns] + gamma*T[s,a,ns]*V[ns]
        best_action = np.argmax(Q)
        policy[s,best_action] = 1
    return policy
            
#find optimal policy through POLICY ITERATION algorithm
def policy_iteration(R, T, policy, max_iter = 10000, k = 100, gamma=1.0):

    n_s,n_a,n_s = T.shape
    opt = np.zeros(n_s)
    
    for iter in range(max_iter):
        #store current policy 
        opt = policy.copy()
        
        #evaluate value function (at least approximately)
        V = policy_evaluation(R, T, policy, k, gamma)
        
        #policy improvement
        policy = policy_improvement(R,T,V,gamma=1.0)
        
        #if policy did not change, stop 
        if np.array_equal(policy,opt):
            break
    return policy


# Evaluate a policy on n runs
def final_evaluate_policy(policy,env,eval_episodes,max_horizon, gamma):
    success_rate = 0.0
    mean_return = 0.0

    for i in range(eval_episodes):
        discounted_return = 0.0
        s = env.reset()

        for step in range(max_horizon):
            s,r, done, info = env.step(np.argmax(policy[s]))
           
            discounted_return += np.power(gamma,step) * r

            if done:
                success_rate += float(r)/eval_episodes
                mean_return += float(discounted_return)/eval_episodes
                break

    return success_rate, mean_return
    
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
env = environment(maze_size=maze_size)
n_a = env.action_space
n_s = env.observation_space
depth = 5

#reward and transition matrices
T = np.zeros([n_s, n_a, n_s])
R = np.zeros([n_s, n_a, n_s])
for s in range(n_s):
    for a in range(n_a):
        transitions = env.P[s][a]
        for p_trans,next_s,rew,done in transitions:
            T[s,a,next_s] += p_trans
            R[s,a,next_s] = rew
        T[s,a,:]/=np.sum(T[s,a,:])

policy = (1.0/n_a)*np.ones((n_s,n_a)) #initilize policy randomly

#calculate optimal policy
policy = policy_iteration(R, T, policy, max_iter = 10000, k = 100000, gamma=1.)
print(final_evaluate_policy(policy, env, eval_episodes = 10000, max_horizon = 10000, gamma=1.))
