import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr


# https://gist.github.com/stober/1943451
# Random argmax in Python
def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


register(
    id="FrozenLake-v3",
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make("FrozenLake-v3")

# 초기 Q 설정
Q = np.zeros([env.observation_space.n, env.action_space.n])

# reward를 담을 배열
rList = []
num_episodes = 2000

for i in range(num_episodes):
    # env reset
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        # 처음 action은 rargmax 함수를 통해 임의로 지정
        action = rargmax(Q[state, :])
        new_state, reward, done, info = env.step(action)

        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

# 성공률 : reward가 1인 횟수/총시도횟수
print("Success rate : ", str(sum(rList) / num_episodes))
print("Left Down Right Up")
print(Q)

# plt.bar(left, height)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
