import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

learning_rate = 0.001
total_epoch = 30
batch_size = 128

n_input = 28  # 가로 픽셀수
n_step = 28  # 세로 픽셀수
n_hidden = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]

model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(total_epoch):
    avg_cost = 0
    for i in range(total_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(batch_size, n_step, n_input)
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / total_batch

    print("Epoch : ", "%04d" % (epoch + 1), " cost : {:.4f}".format(avg_cost))

prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

test_batch_size = len(mnist.test.images)

print("Accuracy : ", sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(test_batch_size, n_step, n_input),
                                                   Y: mnist.test.labels}))


# Epoch :  0001  cost : 0.1239
# Epoch :  0002  cost : 0.0736
# Epoch :  0003  cost : 0.0483
# Epoch :  0004  cost : 0.0343
# Epoch :  0005  cost : 0.0269
# Epoch :  0006  cost : 0.0233
# Epoch :  0007  cost : 0.0217
# Epoch :  0008  cost : 0.0172
# Epoch :  0009  cost : 0.0151
# Epoch :  0010  cost : 0.0146
# Epoch :  0011  cost : 0.0156
# Epoch :  0012  cost : 0.0124
# Epoch :  0013  cost : 0.0141
# Epoch :  0014  cost : 0.0132
# Epoch :  0015  cost : 0.0129
# Epoch :  0016  cost : 0.0099
# Epoch :  0017  cost : 0.0100
# Epoch :  0018  cost : 0.0096
# Epoch :  0019  cost : 0.0098
# Epoch :  0020  cost : 0.0097
# Epoch :  0021  cost : 0.0102
# Epoch :  0022  cost : 0.0082
# Epoch :  0023  cost : 0.0085
# Epoch :  0024  cost : 0.0085
# Epoch :  0025  cost : 0.0080
# Epoch :  0026  cost : 0.0078
# Epoch :  0027  cost : 0.0077
# Epoch :  0028  cost : 0.0064
# Epoch :  0029  cost : 0.0078
# Epoch :  0030  cost : 0.0061
# Accuracy :  0.9735