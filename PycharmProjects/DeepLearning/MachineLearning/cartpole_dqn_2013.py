import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque
import MachineLearning.dqn as dqn

env = gym.make("CartPole-v0")

# CartPole-v0 max step이 200이기 때문에 더 크게 확장
env._max_episode_steps = 10001
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 50000


def simple_replay_train(DQN, train_batch):
    # empty함수는 초기화 되지않은 배열을 생성
    x_stack = np.empty(0).reshape(0, DQN.input_size)
    y_stack = np.empty(0).reshape(0, DQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = DQN.predict(state)

        if done:
            Q[0, action] = reward

        else:
            Q[0, action] = reward + dis * np.max(DQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return DQN.update(x_stack, y_stack)


# 학습된 후 검증
def bot_play(mainDQN, env=env):
    state = env.reset()
    reward_sum = 0

    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, info = env.step(action)
        reward_sum += reward
        if done:
            print("Total score : {}".format(reward_sum))
            break


def main():
    max_episodes = 5000

    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size)
        tf.global_variables_initializer().run()

        for i in range(max_episodes):
            e = 1. / ((i / 10) + 1)

            state = env.reset()
            done = False
            step_count = 0

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, info = env.step(action)

                if done:
                    reward=-100

                # 버퍼에 저장
                replay_buffer.append((state, action,reward, next_state, done))

                # 일정 갯수 이상이면 꺼내기
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:
                    break
            print("Episodes : {} steps: {}".format(i, step_count))


            # 10번마다
            if i % 10 == 1:
                for _ in range(50):
                    # 미니배치를 통해 임의로 10개씩 가져와서 train
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = simple_replay_train(mainDQN, minibatch)
                print("Loss: ", loss)
        bot_play(mainDQN)

if __name__ == "__main__":
    main()
