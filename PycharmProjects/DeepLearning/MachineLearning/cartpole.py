import gym

env = gym.make("CartPole-v0")
env.reset()
random_episodes = 0
reward_sum = 0

while random_episodes < 10:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    print(observation, reward, done)

    # ex
    # [ 0.12732121  0.82116741 -0.15028906 -1.1948945 ] 1.0 False
    # [ 0.14374456  1.01788119 -0.17418695 -1.53065793] 1.0 False
    # [ 0.16410219  0.82523435 -0.20480011 -1.29701283] 1.0 False
    # [ 0.18060687  1.02228189 -0.23074037 -1.64619839] 1.0 True

    reward_sum+=reward

    # 실패했을 경우
    # 출력 후 값 리셋
    if done:
        random_episodes+=1
        print("Reward for this episodes was : ", reward_sum)
        reward_sum=0
        env.reset()


# 실패했을 경우 done = True 반환
# reward는 1를 반환