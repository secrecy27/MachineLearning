import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np

save_dir = './mnist'

data_sets = mnist.read_data_sets(save_dir, dtype=tf.uint8, reshape=False, validation_size=1000)

data_splits = ["train", "test", "validation"]
for d in range(len(data_splits)):
    print("saving : " + data_splits[d])
    data_set = data_sets[d]

    # 각 데이터를 나누어 저장
    filename = os.path.join(save_dir, data_splits[d] + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(data_set.images.shape[0]):
        # 이미지에 대해서 넘파이 배열의 값을 바이트 스트링으로 변환
        image = data_set.images[index].tostring()
        # 이미지를 프로토콜 버퍼 형식으로 변환
        # tf.train.Example은 예제 데이터를 저장하는 자료구조
        # Example객체는 Features 객체를 포함한다. Features 객체는 속성 이름에서 Feature로의 맵을 포함한다.
        # Feature는 하나의 Int64List, ByteList, FloatList를 포함할 수 있다.
        example = tf.train.Example(features=tf.train.Features(feature={
            "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[1]])),
            "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[2]])),
            "depth": tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[3]])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(data_set.labels[index])])),
            "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

filename = os.path.join(save_dir, "train.tfrecords")
record_iterator = tf.python_io.tf_record_iterator(filename)
serialized_img_example = next(record_iterator)

# TFRecord에 이미지를 저장할 때 사용했던 구조체를 복구하려면 바이트 스트링을 파싱한다.
example = tf.train.Example()
example.ParseFromString(serialized_img_example)
image = example.features.feature['image_raw'].bytes_list.value
label = example.features.feature['label'].int64_list.value[0]
width = example.features.feature['width'].int64_list.value[0]
height = example.features.feature['height'].int64_list.value[0]

# 이미지도 바이트 스트링이였으므로 넘파이 배열(28,28,1)로 변환
img_flat = np.fromstring(image[0], dtype=np.uint8)
img_reshaped = img_flat.reshape((height, width, -1))

print(image)
print(label)
print(width)
print(height)
