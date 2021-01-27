from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, losses, optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


(x_train, y_train),(x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress','Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# 显示图像
# plt.figure()
# plt.imshow(x_train[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

x_train, x_test = x_train/255., x_test/255.

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[y_train[i]])
# plt.show()


# 设置层
model = Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 编译模型
# 编译时进行一些设置； 损失函数， 优化器， 指标
model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
# 1 将训练数据送给模型(x_train, y_train)
# 2 模型学习将图像与标签关联起来
# 3 要求模型对测试集x_test进行预测
# 4 验证预测是否与y_test数组中的标签相匹配
model.fit(x_train, y_train, epochs=10)

# 评估准确率
# 比较模型在测试集上的表现
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy: ', test_acc)

# 进行预测
# 模型具有线性输出，即logits
probalility_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probalility_model.predict(x_test)


# 模型对于全部10个类的预测
plt.figure(figsize=(15, 15))
for i in range(9):
    mask = (y_test==i)
    pred = predictions.argmax(axis=1)[mask]

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i],  img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label], color=color))


def plot_value_array(i, preditions_array, true_label):
    predictions_arry, true_label = preditions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), preditions_array, color='#777777')
    plt.ylim([0, 1])
    predited_label = np.argmax(predictions_arry)
    thisplot[predited_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(5, 3))
plt.subplot(1,2, 1)
plot_image(i, predictions[i], y_test, x_test)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], y_test)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], y_test, x_test)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  y_test)
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], y_test, x_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], y_test)
plt.tight_layout()
plt.show()