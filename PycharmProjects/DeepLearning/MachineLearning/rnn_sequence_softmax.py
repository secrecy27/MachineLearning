import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

sentence = (" if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

# 문장을 set()에 넣어 해당 알파벳을 하나만 가지는 리스트 생성
char_set = list(set(sentence))

# char_set을 통해 dic형성 key:value = char:index
char_dic = {c: i for i, c in enumerate(char_set)}

hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10  # 임의의 숫자
learning_rate = 0.1

data_x = []
data_y = []

# X, Y 데이터 할당
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1:i + sequence_length + 1]

    # char_dic을 통해 x,y로 인덱스 할당
    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    # ex
    # print(i, x_str, "->", y_str)
    # 0 if you wan -> f you want
    # x= [10, 24, 9, 4, 14, 6, 9, 13, 21, 23]

    # 1 f you want ->  you want
    # x = [24, 9, 4, 14, 6, 9, 13, 21, 23, 5]

    data_x.append(x)
    data_y.append(y)

batch_size = len(data_x)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# one-hot encoding
X_one_hot = tf.one_hot(X, num_classes)


# X_one_hot : Tensor("one_hot:0", shape=(?, 10, 25), dtype=float32)

def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    return cell


multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

# Fully connected
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for i in range(500):
    l, results, _ = sess.run([loss, outputs, train], feed_dict={X: data_x, Y: data_y})

    for j , result in enumerate(results):
        index=np.argmax(result, axis=1)
        print(i,j,''.join([char_set[c] for c in index])," loss : ", l)


results= sess.run(outputs, feed_dict={X:data_x})

for j, result in enumerate(results):
    index=np.argmax(result, axis=1)

    # j==0일때 처음일때는 한줄 전부 출력후  (j==0일때 if you wan)
    if j is 0:
        print("".join([char_set[c] for c in index]), end="")
    # 뒤에 추가된 한글자씩 출력
    else:
        print(char_set[index[-1]], end="")