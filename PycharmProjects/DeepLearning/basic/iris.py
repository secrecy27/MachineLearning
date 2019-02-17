import pandas as pd
import numpy as np
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense

from keras.callbacks import ModelCheckpoint
import os
from keras.callbacks import EarlyStopping

df = pd.read_csv("iris.csv")
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)
# print(df.head())
# sns.pairplot(df, hue="Species")
# plt.show()

dataset = df.values
X = dataset[1:, 0:5].astype(float)
Y_obj = dataset[1:, 5]

# 품종이  ['iris-setosa','Iris-versicolor','Iris-virginica']에서
# array([1,2,3])으로 바뀜
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
# print(Y_obj)
# print(Y.shape)

# 활성화 함수를 적용해야 하므로 0과 1로 변경
# array([1.,0.,0.],[0.,1.,0.],[0.,0.,1.])으로 변경(원 핫 인코딩)
Y_encoded = np_utils.to_categorical(Y)

model = Sequential()
model.add(Dense(16, input_dim=5, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

MODEL_DIR = './model'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelPath = './model/{epoch:02d}-{loss:.4f}.hdf5'
checkPointer = ModelCheckpoint(filepath=modelPath, monitor='loss', verbose=1, save_best_only=True)

early_stopping_callbacks = EarlyStopping(monitor='loss', patience=100)

history = model.fit(X, Y_encoded, validation_split=0.2, epochs=50, batch_size=1, callbacks=[checkPointer, early_stopping_callbacks])

print("\n Accuracy : %.4f" % (model.evaluate(X, Y_encoded)[1]))

y_vloss = history.history['loss']
y_acc = history.history['acc']

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, "o", c="red", markersize=3)
plt.plot(x_len, y_acc, "o", c="blue", markersize=3)
plt.show()
