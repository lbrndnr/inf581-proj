import numpy as np
import atexit
import functools
from DQN import dqn
from environment_3d import environment
from geometry_3d import actions
from memory import *
from collections import deque
import itertools


class agent_dqn:

    def __init__(self, env, num_frames=4, max_memory=1000, alpha=0.01, gamma=0.9, dropout_prob=0.1, path=None):
        self.env = env
        self.experience = memory(max_memory)
        self.num_frames = num_frames

        input_shape = tuple([self.num_frames] + list(self.env.maze_shape))
        self.net = dqn(len(actions), input_shape, alpha, gamma, dropout_prob, path=path)


    def get_action(self, state):
        q_values = self.net.predict(state)
        return np.argmax(q_values)


    def train(self, episodes=60000, update=True):
        max_episode_length = 1000

        max_exploration_rate, min_exploration_rate = (1.0, 0.3)
        exploration_decay = ((max_exploration_rate - min_exploration_rate) / (episodes * 0.8))
        exploration_rate = max_exploration_rate
        stats = np.zeros((int(episodes/100), 2))
        returnSum = 0
        stepSum = 0

        current_state = lambda d: np.asarray([list(itertools.islice(d, 0, self.num_frames))])
        next_state = lambda d: np.asarray([list(itertools.islice(d, 1, self.num_frames+1))])

        for e in range(episodes):
            self.env.reset()
            done = False
            states = deque()
            for i in range(self.num_frames):
                states.append(self.env.state[0])

            episode_reward = 0
            experience_buffer = []  # This will store the SARS tuples at each episode

            while not done:
                if (np.random.random() < exploration_rate) and update:
                    a = np.random.randint(0, len(actions))
                else:
                    a = self.get_action(current_state(states))

                s, r, done = self.env.step(a)
                states.append(s[0])

                # Add SARS tuple to experience_buffer
                source_frames = current_state(states)
                dest_frames = next_state(states)
                experience_buffer.append(trans(source_frames, a, r, dest_frames, done))
                episode_reward += r

                states.popleft()

                if len(experience_buffer) > max_episode_length:
                    break

            if update:
                self.experience.extend(experience_buffer)
                batch = self.experience.randomized_batch(50)
                loss = self.net.train(batch)  # Train the DQN

                summary = 'Episode {:5d}/{:5d} | Exploration {:.2f} | Loss {:.2f} | ' + \
                      'Episode Length {:4d} | Total Reward {:4d}'
                print(summary.format(
                    e + 1, episodes, exploration_rate, loss, len(experience_buffer), episode_reward
                ))
            
            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay

            returnSum += episode_reward
            stepSum += len(experience_buffer)
            if (e % 100 == 0 and e > 0):
                print("Summary: ", e+1, "Average Return: ", returnSum/100.0, "Average Steps: ", stepSum/100.0)
                stats[int(e/100)-1, 0] = returnSum/100.0
                stats[int(e/100)-1, 1] = stepSum/100.0
                returnSum = 0
                stepSum = 0

        return stats


if __name__ == "__main__":
    path = "res/snake_dqn.h5"
    walls = [(3,4), (3, 5), (3, 6)]
    mice_points=[1]
    env = environment(maze_size=10, walls=walls, mice_points=mice_points)
    agent = agent_dqn(env)

    save_f = functools.partial(agent.net.save, path)
    atexit.register(save_f)

    stats = agent.train()
    agent.net.save(path)

    np.save("stats_DQN.npy", stats)
