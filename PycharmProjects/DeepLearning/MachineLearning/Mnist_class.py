import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15  # 반복횟수
batch_size = 100  # 한번에 읽어들이는 크기


class Mnist_model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.__build_net()

    def __build_net(self):
        with tf.variable_scope(self.name):
            self.keep_prob = tf.placeholder(tf.float32)

            self.X = tf.placeholder(tf.float32, [None, 784])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # 28*28*1 (흑백이미지)
            X_img = tf.reshape(self.X, shape=[-1, 28, 28, 1])

            # 3*3*1
            # 32개의 필터
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="SAME")
            L1 = tf.nn.relu(L1)
            # stride가 2*2 이므로 크기가 반으로 감소
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
            print(L1)
            # conv2d -> (?,28,28,32)
            # max_pool ->(?,14,14,32)

            # 64개의 필터
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
            print(L2)
            # conv2d -> (?,14,14,64)
            # max_pool ->(?,7,7,64)

            # 128개의 필터
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding="SAME")
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
            print(L3)
            # conv2d -> (?,7,7,128)
            # max_pool -> (?,4,4,128)

            L3_flat = tf.reshape(L3, shape=[-1, 4 * 4 * 128])

            W4 = tf.get_variable("W4", shape=[4 * 4 * 128, 625],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            W5 = tf.get_variable("W5", shape=[625, 10],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]))
            self.logits = tf.matmul(L4, W5) + b5
            print(self.logits)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 모델 학습
    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

    # 예측
    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    # 정확도 측정
    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})


# 세션, 클래스 객체 생성
sess = tf.Session()
m1 = Mnist_model(sess, "model1")
sess.run(tf.global_variables_initializer())

# 모델 학습
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print("Epoch : ", "%04d" % (epoch + 1), " cost : ", "{:0.9f}".format(avg_cost))

print("Accuracy : ", m1.get_accuracy(mnist.test.images, mnist.test.labels))
