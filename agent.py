from environment import environment
import copy

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
    
    obstacles, mice, head, curr_direction, score, game_terminated = state
     
    tmp = obstacles.copy()
    tmp.remove(head)
    return DFS(tmp, head)

def reward(state):
    obstacles, mice, head, curr_direction, score, game_terminated = state
    if game_terminated:
        return 10 * score + size_of_accessible_region(state) - 100100100
    else:
        return 10 * score + size_of_accessible_region(state)


def power_list(list_, power):
    if power == 1:
        return list_
    else:
        return [power_list(list_, power - 1) for i in list_]

def get_rewards_in_power_list(env_to_copy, power_list):
    if type(power_list) is list:    
        to_return = []
        for i,inner_pl in enumerate(power_list):
            tmp_environmnet = copy.deepcopy(env_to_copy)
            tmp_environmnet.move(i)
            to_return.append(get_rewards_in_power_list(tmp_environmnet, inner_pl))
        return to_return
    else:
       return reward(env_to_copy.get_state()) 

env = environment()
hit_an_obstacle = False
depth = 3 ######### parameter to change ##########
while hit_an_obstacle == False:
    pl = power_list([0,0,0,0], depth)
    rewards = get_rewards_in_power_list(env, pl)
    # to do
    
    

# just to test
state = env.get_state()
obstacles, mice, head, curr_direction, score, game_terminated = state
#print(obstacles)
print(mice)
print(head)
print(curr_direction)

print(size_of_accessible_region(state))
