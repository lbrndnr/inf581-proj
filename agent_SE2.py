from environment_RL import environment
import copy
import numpy as np
from math import sqrt
import os

# mov[direction] is the displacement for the snake's head.
mov = [(0,1), (0,-1), (1,0), (-1,0)]
    
# the size of the maze region accessible from the snake's head
# (i.e.: that the snake can go to without hitting an obstacle).
def size_of_accessible_region(state):
    maze, head, tail, direction = state
    m,n = maze.shape
    tmp_obstacles = maze.copy()
    tmp_obstacles[head[0]][head[1]] = 0
    
    def DFS(x, y):
        if x < 0 or x >= m or y < 0 or y >= n:
            return 0
        if tmp_obstacles[x][y] < 0:
            return 0
        size = 1
        # mark as visited
        tmp_obstacles[x][y] = -1
        # for each direction, visit adjacent cell
        for dir in range(4):
            size += DFS(x+mov[dir][0], y+mov[dir][1])
        return size
    
    return DFS(head[0], head[1])


def p_mice(state):
    maze, head, tail, direction = state
    m,n = maze.shape
    proximity_to_mice = 0
    for i in range(m):
        for j in range(n):
            if maze[i][j] > 0:
                dist_to_mouse = abs(head[0]-i) + abs(head[1]-j)
                #dist_to_mouse = sqrt((head[0]-i)**2 + (head[1]-j)**2)
                if dist_to_mouse != 0:
                    proximity_to_mice += maze[i][j] / dist_to_mouse
    return proximity_to_mice

def power_list(list_, power):
    if power == 1:
        return list_
    else:
        return [power_list(list_, power - 1) for i in list_]

def get_rewards_in_power_list(env_to_copy, power_list_, reward_until_now):
    if type(power_list_) is list:    
        to_return = []
        for i,inner_pl in enumerate(power_list_):
            tmp_environment = copy.deepcopy(env_to_copy)
            _,reward_step,ended = tmp_environment.step(i)
            if reward_step < 0:
                reward_step = -10100100
            if reward_step > 0:
                reward_step *= 1
            to_return.append(get_rewards_in_power_list(tmp_environment, inner_pl, 2 * reward_until_now + reward_step))
        return to_return
    else:
       return reward_until_now + p_mice(env_to_copy.state) + size_of_accessible_region(env_to_copy.state)

def get_max_from_power_list(power_list_):
    if type(power_list_) is list:
        return max([get_max_from_power_list(inner_pl) for inner_pl in power_list_])
    else:
        return power_list_     

def run(using_terminal=False):
    env = environment()
    hit_an_obstacle = False
    depth = 2 ######### parameter to change ##########
    max_n_iter = 100100100
    n_iter = 0
    while hit_an_obstacle == False and n_iter < max_n_iter:
        n_iter += 1
        pl = power_list([0,0,0], depth)
        rewards = get_rewards_in_power_list(env, pl, 0)
        optimal_direction = np.argmax([get_max_from_power_list(rewards[i]) for i in range(3)])        
        state,reward,hit_an_obstacle = env.step(optimal_direction)
        if using_terminal:
            os.system('cls' if os.name == 'nt' else 'clear')
            print('')
            print(env.maze_string())
            print('')
        else:
            print(env.maze_string())

run(using_terminal=True)
