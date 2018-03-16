import tensorflow as tf
import numpy as np

# char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
#             'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

char_arr = [chr(i) for i in range(97, 123)]

num_dic = {n: i for i, n in enumerate(char_arr)}
# num_dic = {'a':0, 'b':1, ...}

dic_len = len(num_dic)

# 학습에 사용할 단어 3글자 입력시 마지막 글자 자동완성
seq_data = ["word", "wood", "deep", "dive", "cold", "cool", "load", "love", "kiss", "kind"]


def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        # 입력값 : 단어의 처음 세글자의 알파벳 인덱스
        input = [num_dic[n] for n in seq[:-1]]
        # 출력값 : 마지막 글자의 인덱스
        target = num_dic[seq[-1]]
        # 원핫인코딩
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)

    return input_batch, target_batch


learning_rate = 0.01
n_hidden = 128
total_batch = 30
n_step = 3
n_input = n_class = dic_len

X = tf.placeholder(tf.float32, [None, n_step, n_input])

# Y: 원핫 인코딩이 아닌 인덱스 숫자를 그대로 사용
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# 여러 셀 조합
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
outputs, state = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]

model = tf.matmul(outputs, W) + b
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_batch):
    c, _ = sess.run([cost, optimizer], feed_dict={X: input_batch, Y: target_batch})

    print("Epoch : ", "%04d" % (epoch + 1), " cost : {:.4f}".format(c))

# 실측값이 원핫인코딩이 아닌 인덱스이므로 Y는 정수이므로 int형으로 변경
prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy=tf.reduce_mean(tf.cast(prediction_check, tf.float32))

input_batch,target_batch =make_batch(seq_data)

predict, accuracy_value=sess.run([prediction,accuracy], feed_dict={X:input_batch, Y:target_batch})

predict_words=[]
for idx, val in enumerate(seq_data):
    last_char=char_arr[predict[idx]]
    predict_words.append(val[:3]+last_char)

print("----예측결과-----")
print("입력값 : ",[w[:3]+" " for w in seq_data])
print("예측값 : ",predict_words)
print("정확도 : ",accuracy_value)


# Epoch :  0001  cost : 3.6213
# Epoch :  0002  cost : 2.5779
# Epoch :  0003  cost : 1.4928
# Epoch :  0004  cost : 1.3239
# Epoch :  0005  cost : 0.9008
# Epoch :  0006  cost : 0.5080
# Epoch :  0007  cost : 0.3604
# Epoch :  0008  cost : 0.5053
# Epoch :  0009  cost : 0.4937
# Epoch :  0010  cost : 0.1198
# Epoch :  0011  cost : 0.2552
# Epoch :  0012  cost : 0.1757
# Epoch :  0013  cost : 0.0548
# Epoch :  0014  cost : 0.1741
# Epoch :  0015  cost : 0.1856
# Epoch :  0016  cost : 0.0880
# Epoch :  0017  cost : 0.0532
# Epoch :  0018  cost : 0.0808
# Epoch :  0019  cost : 0.0551
# Epoch :  0020  cost : 0.0486
# Epoch :  0021  cost : 0.0540
# Epoch :  0022  cost : 0.0222
# Epoch :  0023  cost : 0.0158
# Epoch :  0024  cost : 0.1080
# Epoch :  0025  cost : 0.0052
# Epoch :  0026  cost : 0.0686
# Epoch :  0027  cost : 0.0027
# Epoch :  0028  cost : 0.0236
# Epoch :  0029  cost : 0.0007
# Epoch :  0030  cost : 0.0014
# ----예측결과-----
# 입력값 :  ['wor ', 'woo ', 'dee ', 'div ', 'col ', 'coo ', 'loa ', 'lov ', 'kis ', 'kin ']
# 예측값 :  ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']
# 정확도 :  1.0