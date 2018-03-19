from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import numpy as np


class dqn:

    def __init__(self, n_actions, input_shape, alpha, gamma, dropout_prob, path=None):
        self.model = Sequential()
        self.gamma = gamma
        self.dropout_prob = dropout_prob

        # Define neural network
        self.model.add(Conv2D(
            16,
            kernel_size=(3, 3),
            strides=(1, 1),
            data_format='channels_first',
            input_shape=input_shape
        ))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            data_format='channels_first'
        ))
        self.model.add(Activation('relu'))

        # Dense layers.
        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dense(n_actions))

        if path is not None:
            self.load(load_path)

        self.model.compile(RMSprop(), 'MSE')


    def train(self, batch):
        x_train = []
        t_train = []

        # Generate training set and targets
        for dp in batch:
            x_train.append(dp.s.astype(np.float64))

            # Get the current Q-values for the next state and select the best
            next_state_pred = self.predict(dp.s_next)
            next_q_value = np.max(next_state_pred)

            # The error must be 0 on all actions except the one taken
            t = list(self.predict(dp.s))
            t[dp.a] = dp.r if dp.done else dp.r + self.gamma * next_q_value

            t_train.append(t)

        x_train = np.asarray(x_train).squeeze()
        t_train = np.asarray(t_train).squeeze()

        return float(self.model.train_on_batch(x_train, t_train))


    def predict(self, state):
        state = state.astype(np.float64)
        return self.model.predict(state, batch_size=1).ravel()


    def save(self, path):
        self.model.save_weights(path)


    def load(self, path):
        self.model.load_weights(path)
