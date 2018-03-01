import numpy as np

STRAIGHT = 0
RIGHT = 1
LEFT = 2

actions = [STRAIGHT, RIGHT, LEFT]

#given a current direction d as a vector and a new action a, compute the new direction as a vector
def vectorize(d, a):
    assert(a >= STRAIGHT and a <= LEFT)

    if a == STRAIGHT:
        return d
    else:
        ds = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        offset = 1 if a == RIGHT else -1
        curr_idx = ds.index(tuple(d))
        next_idx = (curr_idx + offset) % len(ds)

        return np.array(ds[next_idx])


def vec_to_int(vec, maze_size):
    if not (vec[0] >= 0 and vec[1] >= 0):
        print(vec)
    assert(vec[0] >= 0 and vec[1] >= 0)
    return vec[0] * maze_size + vec[1]


def int_to_vec(i, maze_size):
    return np.array([i/maze_size, i%maze_size])