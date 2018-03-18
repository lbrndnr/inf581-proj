import numpy as np
import atexit
import functools
from DQN_tf import dqn
from environment import environment
from geometry import actions
from memory import *


class agent_dqn:

    def __init__(self, env, batch_size=1024, n_frames=2, alpha=0.01, gamma=0.99, dropout_prob=0.1, path=None):
        self.batch_size = batch_size  
        self.env = env
        self.experience = memory(batch_size)

        input_shape = tuple([n_frames] + list(self.env.maze_shape))
        self.net = dqn(len(actions), input_shape, alpha, gamma, dropout_prob, path=path)


    def get_action(self, state):
        q_values = self.net.predict(state)
        return np.argmax(q_values)


    def train(self, max_training_count=1000, max_episodes=None, update=True):
        exp_backup_counter = 0
        epsilon = 1.0
        training_count = 0
        episodes = 0
        max_episode_length = 500

        while training_count < max_training_count and (True if max_episodes is None else episodes < max_episodes):
            self.env.reset()
            done = False
            state = [self.env.state[0], self.env.state[0]]
            next_state = [self.env.state[0], self.env.state[0]]
            episode_reward = 0
            experience_buffer = []  # This will store the SARS tuples at each episode

            episodes += 1
            while not done:
                if (np.random.random() < epsilon) and update:
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

            if not update:
                print(episode_reward, len(experience_buffer))
            
            if episode_reward >= 0 and len(experience_buffer) >= 5 and update:
                exp_backup_counter += len(experience_buffer)
                print('Adding episode to experiences - Score: %s; Episode length: %s' % (episode_reward+1, len(experience_buffer)))
                print('Got %s samples of %s' % (exp_backup_counter, self.experience.max_transitions))
                self.experience.extend(experience_buffer)

            if self.experience.is_full and update:
                exp_backup_counter = 0

                training_count += 1
                print('Training session #', training_count, ' - epsilon:', epsilon)
                batch = self.experience.randomized_batch()
                self.net.train(batch)  # Train the DQN
                self.experience.reset()

                print("Verify =========================")
                self.train(max_episodes=3, update=False)
                print("================================")

                if epsilon > 0.3:
                    epsilon *= 0.999



if __name__ == "__main__":
    path = "res/snake_dqn"
    walls = [(3,4), (3, 5), (3, 6)]
    env = environment(maze_size=20, walls=walls)
    agent = agent_dqn(env)

    save_f = functools.partial(agent.net.save, path)
    atexit.register(save_f)

    agent.train()
