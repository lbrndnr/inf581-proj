import numpy as np

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

actions = [NORTH, EAST, SOUTH, WEST]

#given a current direction d as a vector and a new action a, compute the new direction as a vector
def vectorize(a):
    assert(a >= NORTH and a <= WEST)

    ds = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    return ds[a]


def vec_to_int(vec, maze_size):
    if not (vec[0] >= 0 and vec[1] >= 0):
        print(vec)
    assert(vec[0] >= 0 and vec[1] >= 0)
    return vec[0] * maze_size + vec[1]


def int_to_vec(i, maze_size):
    return np.array([i/maze_size, i%maze_size])


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


#this function returns the direction to take in order to reach b from a
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