import tensorflow as tf

# 최대 10개의 항목을 넣을 수 있는 큐 생성
# 큐는 연산 그래프의 일부이므로 세션 안에서 수행됨
sess = tf.InteractiveSession()
queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string])
enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["I"])
enque_op.run()
enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["O"])
enque_op.run()
print(sess.run(queue1.size()))

x=queue1.dequeue()
print(x.eval())
x=queue1.dequeue()
print(x.eval())
x=queue1.dequeue()
print(x.eval())
x=queue1.dequeue()
print(x.eval())
x=queue1.dequeue()

queue2 = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
enque_op = queue2.enqueue(["F"])
enque_op.run()
enque_op = queue2.enqueue(["I"])
enque_op.run()
enque_op = queue2.enqueue(["F"])
enque_op.run()
enque_op = queue2.enqueue(["O"])
enque_op.run()

# 한번에 꺼내는 코드
inputs = queue2.dequeue_many(4)
print(inputs.eval())