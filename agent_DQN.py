import numpy as np
import atexit
import functools
from DQN import dqn
from environment_3d import environment
from geometry_3d import actions
from memory import *


class agent_dqn:

    def __init__(self, env, batch_size=100, n_frames=2, alpha=0.005, gamma=0.9, dropout_prob=0.1, path=None):
        self.env = env
        self.experience = memory(batch_size)

        input_shape = tuple([n_frames] + list(self.env.maze_shape))
        self.net = dqn(len(actions), input_shape, alpha, gamma, dropout_prob, path=path)


    def get_action(self, state):
        q_values = self.net.predict(state)
        return np.argmax(q_values)


    def train(self, episodes=30000, update=True):
        max_episode_length = 1000

        max_exploration_rate, min_exploration_rate = (1.0, 0.1)
        exploration_decay = ((max_exploration_rate - min_exploration_rate) / (episodes * 0.8))
        exploration_rate = max_exploration_rate

        for e in range(episodes):
            self.env.reset()
            done = False
            state = [self.env.state[0], self.env.state[0]]
            next_state = [self.env.state[0], self.env.state[0]]
            episode_reward = 0
            experience_buffer = []  # This will store the SARS tuples at each episode

            while not done:
                if (np.random.random() < exploration_rate) and update:
                    a = np.random.randint(0, len(actions))
                else:
                    a = self.get_action(np.asarray([state]))

                s, r, done = self.env.step(a)
                next_state[1] = s[0]

                # Add SARS tuple to experience_buffer
                experience_buffer.append(trans(np.asarray([state]), a, r, np.asarray([next_state]), done))
                episode_reward += r

                # Change current state
                state = list(next_state)

                if len(experience_buffer) > max_episode_length:
                    break

            if update:
                self.experience.extend(experience_buffer)
                batch = self.experience.randomized_batch()
                self.net.train(batch)  # Train the DQN

                summary = 'Episode {:5d}/{:5d} | Exploration {:.2f} | ' + \
                      'Episode Length {:4d} | Total Reward {:4d}'
                print(summary.format(
                    e + 1, episodes, exploration_rate, len(experience_buffer), episode_reward
                ))
            
            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay



if __name__ == "__main__":
    path = "res/snake_dqn.h5"
    walls = [(3,4), (3, 5), (3, 6)]
    mice_points=[1]
    env = environment(maze_size=10, walls=walls, mice_points=mice_points)
    agent = agent_dqn(env)

    save_f = functools.partial(agent.net.save, path)
    atexit.register(save_f)

    agent.train()
    agent.net.save()
