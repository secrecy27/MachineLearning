import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

env = gym.make("FrozenLake-v0")

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

X = tf.placeholder(dtype=tf.float32, shape=[None, input_size])  # state input
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))  # weight

Qpred = tf.matmul(X, W)
Y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])  # Y label

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

num_episodes = 2000
rList = []
dis = 0.99


def one_hot(x):
    return np.identity(16)[x:x + 1]


# 시간 측정
start_time = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        state = env.reset()
        rAll = 0
        done = False
        e = 1. / ((i / 50) + 10)

        while not done:
            Qs = sess.run(Qpred, feed_dict={X: one_hot(state)})
            # E - greedy
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qs)

            new_state, reward, done, info = env.step(action)

            if done:
                # terminal state
                # Qs = [[x,x]] 2차원 배열이기에 앞에 0이 들어감
                # [[a1,a2,a3,a4]] 1x4 array
                Qs[0, action] = reward
            else:
                # new_state
                Qnew = sess.run(Qpred, feed_dict={X: one_hot(new_state)})

                # 새로운 Q 업데이트
                Qs[0, action] = reward + dis * np.max(Qnew)

            sess.run(train, feed_dict={X: one_hot(state), Y: Qs})

            rAll += reward
            state = new_state
        rList.append(rAll)

print("---%s second---" % (time.time() - start_time))
print("Success rate : " + str(sum(rList) / num_episodes))
print(Qs)
plt.bar(range(len(rList)), rList)
plt.show()
