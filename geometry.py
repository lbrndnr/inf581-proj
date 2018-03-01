import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

directions = [UP, RIGHT, DOWN, LEFT]

def vectorize(direction):
    if direction == UP:
        return np.array([-1, 0])
    elif direction == RIGHT:
        return np.array([0, 1])
    elif direction == DOWN:
        return np.array([1, 0])
    elif direction == LEFT:
        return np.array([0, -1])
    else:
        NotImplemented


def vec_to_int(vec, maze_size):
    if not (vec[0] >= 0 and vec[1] >= 0):
        print(vec)
    assert(vec[0] >= 0 and vec[1] >= 0)
    return vec[0] * maze_size + vec[1]


def int_to_vec(i, maze_size):
    return np.array([i/maze_size, i%maze_size])