import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self.__build_network()

    def __build_network(self, h_size=10, learning_rate=0.1):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))

            W2 = tf.get_variable("W2", shape=[h_size, self.output_size],
                                 initializer = tf.contrib.layers.xavier_initializer()
            )

            self._Qpred = tf.matmul(layer1, W2)

        self._Y = tf.placeholder(dtype=tf.float32, shape=[None, self.output_size])

        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

        self._train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
