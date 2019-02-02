import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_hidden = 256
n_input = 28 * 28
n_noise = 128

# GAN : Generative Adversarial Network , 비지도학습
# Discriminator : 이미지가 진짜인지 판단한다.
# Generator :노이즈로부터 임의의 이미지를 만들어 구분자를 속인다.

X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

# 생성자 신경망에 사용할 변수 설정
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 구분자 신경망에 사용할 변수 설정
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))


# 실제 이미지를 판별하는 구분자 신경망과 생성한 이미지를 판별하는 구분자 신경망은
# 같은 변수를 사용해야 한다. 같은 신경망으로 구분을 시켜야 진짜와 가짜를 구별하는
# 특징들을 동시에 잡아낼수 있기 때문

# 생성자
def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
    return output


# 구분자
def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    return output


# 무작위로 노이즈를 만들어주는 함수
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))


G = generator(Z)
D_gene = discriminator(G)  # 생성한 가짜
D_real = discriminator(X)  # 진짜

# cost도 두개가 필요

# 생성자가 만든 이미지를 가짜라고 판단하도록 하는 손실값(경찰 학습용)
# D_real은 1에 가깝게(진짜라고 판별), D_gene은 0에 가깝게(가짜라고 판별)
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))

# 진짜라고 판단하도록 하는 손실값(위조지폐범 학습용)
# D_gene을 1에 가깝게 만들어야한다.
loss_G = tf.reduce_mean(tf.log(D_gene))

# loss_D와 loss_G 두 값 모두 최대화 하여야 하지만 경쟁관계이기에 비례관계는 아니다.

D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# 값을 최대화하여야 하므로 minimize함수에 음수를 붙인다.
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

    print("Epoch : ", "%04d" % epoch, " D loss : {:.4}".format(loss_val_D), " G loss : {:.4}".format(loss_val_G))

    # 이미지 생성 확인

    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig("gan_samples/{}.png".format(str(epoch).zfill(3)), bbox_inches="tight")
        plt.close(fig)
