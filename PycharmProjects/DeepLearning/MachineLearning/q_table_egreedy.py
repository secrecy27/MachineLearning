import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id="FrozenLake-v3",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={'map_name': "4x4", "is_slippery": False}
)

env = gym.make("FrozenLake-v3")
Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000
rList = []
dis = 0.9

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    e = 1. / ((i // 100) + 1)

    while not done:
        # E-greedy
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        new_state, reward, done, info = env.step(action)

        Q[state, action] = reward + dis * np.argmax(Q[new_state, :])

        rAll += reward
        state = new_state
    rList.append(rAll)

print("Success rate : ", str(len(rList) / num_episodes))
print("Left Down Right Up")
print(Q)

plt.bar(range(num_episodes), rList, color="r")
plt.show()
