from __future__ import absolute_import, print_function, division, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

(trian_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = trian_images.reshape((60000, 28, 28,1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images/255.0, test_images/255.0
# train_labels = tf.one_hot(train_labels, depth=10)
# test_labels = tf.one_hot(test_labels, depth=10)

# 创建卷积基
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.summary()

# 在顶部添加密集层
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# 编译和训练模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,train_labels, epochs=10)

loss, accu = model.evaluate(test_images, test_labels)