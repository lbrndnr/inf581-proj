import os
import tensorflow as tf
import numpy as np


class dqn:

    def __init__(self, n_actions, input_shape, alpha, gamma, dropout_prob, path=None):
        self.gamma = gamma

        if path is None:
            self.__init_network(n_actions, input_shape, alpha, dropout_prob)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if path is not None:
            self.load(path)


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
        fc1 = tf.layers.dense(flatten, units=512, activation=tf.nn.relu, name="fc1")
        
        self.action_q = tf.layers.dense(fc1, units=n_actions, activation=tf.nn.relu, name="action_q")
        tf.add_to_collection("predict", self.action_q)

        loss = tf.reduce_mean(tf.square(self.action_q - self.tf_q_targets))
        self.optimizer = tf.train.AdamOptimizer(alpha).minimize(loss)
        tf.add_to_collection("predict", self.optimizer)
        


    def train(self, batch):
        mazes = []
        q_targets = []

        # Generate training set and targets
        for t in batch:
            mazes.append(t.s)

            # Get the current Q-values for the next state and select the best
            next_state_pred = self.predict(t.s_next)
            next_q_value = np.max(next_state_pred)

            # The error must be 0 on all actions except the one taken
            q = self.predict(t.s)
            q[t.a] = t.r if t.done else t.r + self.gamma * next_q_value

            q_targets.append(q)

        mazes = np.asarray(mazes).squeeze()
        q_targets = np.asarray(q_targets).squeeze()

        self.sess.run(self.optimizer, feed_dict={self.tf_mazes: mazes, self.tf_q_targets: q_targets})


    def predict(self, state):
        state = state.astype(np.float64)
        q = self.sess.run(self.action_q, feed_dict={self.tf_mazes: state})
        return q.ravel()


    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
    

    def load(self, path):
        p, _ = os.path.split(path)

        saver = tf.train.import_meta_graph(path + ".meta")
        saver.restore(self.sess, tf.train.latest_checkpoint(p))

        graph = tf.get_default_graph()
        self.tf_mazes = graph.get_tensor_by_name("inputs/maze:0")
        self.tf_q_targets = graph.get_tensor_by_name("inputs/q_target:0")
        self.action_q = graph.get_collection("predict")[0]
        self.optimizer = graph.get_collection("predict")[1]
