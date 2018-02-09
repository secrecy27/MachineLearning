import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

env = gym.make("CartPole-v0")

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.1

X = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
W = tf.get_variable("W", shape=[input_size, output_size], dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X, W)

Y = tf.placeholder(dtype=tf.float32, shape=[1, output_size])

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

dis = 0.99
rList = []
num_episodes = 2000

start_time = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_episodes):
        done = False
        e = 1. / ((i / 10) + 1)
        state = env.reset()
        step_count = 0
        while not done:
            step_count += 1
            x = np.reshape(state, [1, input_size])
            Qs = sess.run(Qpred, feed_dict={X: x})

            # E-greedy
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qs)

            new_state, reward, done, info = env.step(action)

            if done:
                # 실패했을 경우 불이익
                Qs[0, action] = -100
            else:
                x1 = np.reshape(new_state, [1, input_size])
                Qnew = sess.run(Qpred, feed_dict={X: x1})

                Qs[0, action] = reward + dis * np.max(Qnew)

            sess.run(train, feed_dict={X: x, Y: Qs})
            state = new_state

        rList.append(step_count)
        print("Episodes : {} steps : {}".format(i, step_count))

        if len(rList) > 10 and np.mean(rList[-10:]) > 500:
            break

    print("--- %s seconds ---" % (time.time() - start_time))

    # Total reward
    observation = env.reset()
    reward_sum = 0

    while True:
        env.render()

        x = np.reshape(observation, [1, input_size])
        Qs = sess.run(Qpred, feed_dict={X: x})
        action = np.argmax(Qs)

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        if done:
            print("Total score : {}".format(reward_sum))
            break
