from collections import deque
import random
import numpy as np

class memory:

    def __init__(self, n_actions, max_memory=1024):
        self.max_memory = max_memory
        self.n_actions = n_actions
        self.reset()


    def reset(self):
        self.mem = deque()


    def add(self, s, a, r, s_next, game_over):
        self.mem.append((s, a, r, s_next, game_over))
        if len(self.mem) > self.max_memory:
            self.mem.popleft()


    def random_batch(self, batch_size, gamma):
        batch_size = min(batch_size, len(self.mem))
        batch = random.sample(self.mem, batch_size)

        select = lambda idx: np.array([m[idx] for m in batch])

        states = select(0)
        actions = select(1)
        rewards = select(2).repeat(self.n_actions).reshape((batch_size, self.n_actions))
        next_states = select(3)
        game_overs = select(4).repeat(self.n_actions).reshape((batch_size, self.n_actions))

        for m in batch:
            self.mem.remove(m)    
        
        return states, actions, rewards, next_states, game_overs, batch_size
