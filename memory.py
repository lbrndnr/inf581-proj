from collections import namedtuple
import numpy as np


class trans:

    def __init__(self, s, a, r, s_next, done):
        self.s = s
        self.a = a
        self.r = r
        self.s_next = s_next
        self.done = done


class memory:

    def __init__(self, max_transitions):
        self.max_transitions = max_transitions
        self.reset()


    def reset(self):
        self.transitions = []


    @property
    def is_full(self):
        return len(self.transitions) >= self.max_transitions


    def extend(self, ts):     
        self.transitions.extend(ts)


    def randomized_batch(self):
        return np.random.permutation(self.transitions)[:self.max_transitions]
        
