import threading
import tensorflow as tf
import time

with tf.Session() as sess:
    gen_random_normal = tf.random_normal(shape=())
    queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
    enque = queue.enqueue(gen_random_normal)


    def add(coord, i):
        while not coord.should_stop():
            sess.run(enque)
            print("i : ",i)
            if i == 10:
                coord.request_stop()


    coord = tf.train.Coordinator()
    # 10개의 스레드를 생성하는 데 각 스레드가 add 함수를 병렬로 수행하며
    # add함수는 10개의 항목을 큐에 동기화되지 않은 상태로 넣는다.
    threads = [threading.Thread(target=add, args=(coord, i)) for i in range(10)]
    coord.join(threads)

    for t in threads:
        t.start()

    print(sess.run(queue.size()))
    time.sleep(0.01)
    print(sess.run(queue.size()))
    time.sleep(0.01)
    print(sess.run(queue.size()))

    #
    # queue = tf.RandomShuffleQueue(capacity=100, dtypes=[tf.float32], min_after_dequeue=1)
    # enqueue_op = queue.enqueue(gen_random_normal)
    #
    # qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
    # coord = tf.train.Coordinator()
    # enqueue_threads = qr.create_threads(sess,coord=coord, start=True)
    # coord.request_stop()
    # coord.join(enqueue_op)