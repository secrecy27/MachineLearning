import numpy as np


def softmax(data):
    exp_data = np.exp(data)
    sum_data = np.sum(exp_data)
    return exp_data / sum_data


data = np.array([0.3, 2.9, 0.5])
print(softmax(data))

a = np.array([1010, 1000, 990])
# error :  수의 한정으로 인한 overflow 발생
# 아무런 조치 없이 실행할 경우 발생
# print(softmax(a))

# 보통을 최댓값을 빼준다
c = np.max(a)

print(softmax(a - c))

def softmax2(data):
    max_data = np.max(data)
    max_data=data-max_data
    exp_data = np.exp(max_data)
    sum_data = np.sum(exp_data)
    return exp_data/sum_data

print(softmax2(a))