from keras.datasets import reuters
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import matplotlib.pyplot as plt

# num_words는 1~1000 나온 데이터의 한해서 로딩
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1000, test_split=0.2)
category = np.max(Y_train) + 1
print(category, "카테고리")
print(len(X_train), "학습용 뉴스 기사")
print(len(X_test), " 테스트용 뉴스 기사")
print(X_train[0])

# 단어 수를 100개로 맞추라는 뜻 / 100보다 크면 100개째 단어만 선택하고 나머지는 버림
# 100보다 작을 때는 모자라는 부분을 0으로 채움
x_train = sequence.pad_sequences(X_train, maxlen=100)
x_test = sequence.pad_sequences(X_test, maxlen=100)

y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

model = Sequential()
model.add(Embedding(1000, 10))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=100, epochs=20, validation_data=(x_test, y_test))

print("\n Test Accuracy : %.4f " % (model.evaluate(x_test, y_test)[1]))

# 데이터셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
