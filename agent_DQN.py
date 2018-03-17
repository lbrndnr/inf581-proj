import numpy as np
import atexit
import functools
from DQN import dqn
from environment import environment
from geometry import actions


class agent_dqn:

    def __init__(self, env, batch_size=1024, n_frames=2, alpha=0.01, gamma=0.99, dropout_prob=0.1):
        self.batch_size = batch_size  
        self.env = env

        # Experience variables
        self.experiences = []
        self.training_count = 0

        input_shape = tuple([n_frames] + list(self.env.maze_shape))
        self.net = dqn(len(actions), input_shape, alpha=alpha, gamma=gamma, dropout_prob=dropout_prob)


    def get_action(self, state):
        q_values = self.net.predict(state)
        return np.argmax(q_values)


    def add_experience(self, source, action, reward, dest, final):
        self.experiences.append({'source': source,
                                 'action': action,
                                 'reward': reward,
                                 'dest': dest,
                                 'final': final})


    def sample_batch(self):
        out = [self.experiences.pop(np.random.randint(0, len(self.experiences)))
               for _ in range(self.batch_size)]
        return np.asarray(out)


    def must_update(self):
        return len(self.experiences) >= self.batch_size
    
    
    def screen(self, s):
        maze = s[0]
        return maze


    def train(self, update=True):
        exp_backup_counter = 0
        experience_buffer = []  # This will store the SARS tuples at each episode
        epsilon = 1.0

        for e in range(100000):
            self.env.reset()
            done = False
            state = [self.screen(self.env.state), self.screen(self.env.state)]
            next_state = state
            episode_reward = 0
            episode_length = 0

            while not done:
                if (np.random.random() < epsilon) and update:
                    a = np.random.randint(0, len(actions))
                else:
                    a = self.get_action(np.asarray([state]))

                s, reward, done = self.env.step(a)
                next_state[1] = self.screen(s)

                # Add SARS tuple to experience_buffer
                experience_buffer.append((np.asarray([state]), a, reward,
                                        np.asarray([next_state]),
                                        done))
                episode_reward += reward

                # Change current state
                state = list(next_state)
                episode_length += 1

                # Add the episode to the experience buffer
            print(episode_reward, episode_length)
            if episode_reward >= 0 and episode_length >= 5:
                exp_backup_counter += len(experience_buffer)
                print('Adding episode to experiences - Score: %s; Episode length: %s' % (episode_reward+1, episode_length))
                print('Got %s samples of %s' % (exp_backup_counter, self.batch_size))
                for exp in experience_buffer:
                    self.add_experience(*exp)

            if self.must_update() and update:
                exp_backup_counter = 0

                self.training_count += 1
                print('Training session #', self.training_count, ' - epsilon:', epsilon)
                batch = self.sample_batch()
                self.net.train(batch)  # Train the DQN

                if epsilon > 0.3:
                    epsilon *= 0.999

            experience_buffer = []


if __name__ == "__main__":
    walls = [(3,4), (3, 5), (3, 6)]
    env = environment(maze_size=20, walls=walls)
    agent = agent_dqn(env)

    save_f = functools.partial(agent.net.save, "res/snake_dqn.ckpt")
    atexit.register(save_f)

    agent.train()
