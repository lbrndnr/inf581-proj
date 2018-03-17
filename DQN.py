import tensorflow as tf
import numpy as np


class dqn:

    def __init__(self, n_actions, input_shape, alpha, gamma, dropout_prob):
        self.gamma = gamma

        self.__init_network(n_actions, input_shape, alpha, dropout_prob)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def __init_network(self, n_actions, input_shape, alpha, dropout_prob):
        input_shape = list(input_shape)

        # Placeholder for inputs (states)
        with tf.name_scope('inputs'):
            self.tf_mazes = tf.placeholder(tf.float32, [None] + input_shape, name="maze")
            self.tf_q_targets = tf.placeholder(tf.float32, [None, n_actions], name="q_target")

        transpose = tf.transpose(self.tf_mazes, [0, 2, 3, 1])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=transpose,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            name="conv1")

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            name="conv2")

        # Dense Layer
        flatten = tf.reshape(conv2, [-1, 20*20*2*16])
        dropout = tf.nn.dropout(flatten, dropout_prob)
        fc1 = tf.layers.dense(dropout, units=512, activation=tf.nn.relu, name="fc1")
        
        self.action_q = tf.layers.dense(fc1, units=n_actions, activation=tf.nn.relu, name="action_q")

        loss = tf.reduce_mean(tf.square(self.action_q - self.tf_q_targets))
        self.optimizer = tf.train.AdamOptimizer(alpha).minimize(loss)


    def train(self, batch):
        x_train = []
        t_train = []

        # Generate training set and targets
        for datapoint in batch:
            x_train.append(datapoint['source'].astype(np.float64))

            # Get the current Q-values for the next state and select the best
            next_state_pred = self.predict(datapoint['dest'].astype(np.float64)).ravel()
            next_q_value = np.max(next_state_pred)

            # The error must be 0 on all actions except the one taken
            t = list(self.predict(datapoint['source'])[0])
            if datapoint['final']:
                t[datapoint['action']] = datapoint['reward']
            else:
                t[datapoint['action']] = datapoint['reward'] + \
                                         self.gamma * next_q_value

            t_train.append(t)

        # Prepare inputs and targets
        x_train = np.asarray(x_train).squeeze()
        t_train = np.asarray(t_train).squeeze()

        self.sess.run(self.optimizer, feed_dict={self.tf_mazes:x_train, self.tf_q_targets:t_train})


    def predict(self, state):
        state = state.astype(np.float64)
        q = self.sess.run(self.action_q, feed_dict={self.tf_mazes: state})
        return q


    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
    

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
