import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id="FrozenLake-v3",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False}
)

env = gym.make("FrozenLake-v3")
Q = np.zeros([env.observation_space.n, env.action_space.n])

rList = []
num_episodes = 2000

for i in range(num_episodes):
    state = env.reset()
    done = False
    rAll = 0
    # discounted reward
    dis = 0.9

    while not done:
        # 노이즈 추가
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

        new_state, reward, done, info = env.step(action)

        # discounted reward
        Q[state, action] = reward + dis * np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate : ", str(sum(rList) / num_episodes))
print("Left Down Right Up")
print(Q)

plt.bar(range(len(rList)), rList, color="b")
plt.show()
