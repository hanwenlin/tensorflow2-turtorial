from __future__ import division, absolute_import, print_function, unicode_literals
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from tensorflow.keras import layers


# 在eager模式下可以直接进行运算
x = [[3.]]
m = tf.matmul(x, x)
print(m.numpy())

a = tf.constant([[1,9],[3,6]])
print(a)

b = tf.add(a, 2)
print(b)
print(a*b)

s = np.multiply(a,b)
print(s)

# 2.动态控制流
def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(1, max_num.numpy()+1):
        num = tf.constant(num)
        if int(num %3) ==0 and int(num %5)==0:
            print('FizzBuzz')
        elif int(num %3) ==0:
            print('Fizz')
        elif int(num %5) ==0:
            print('Buzz')
        else:
            print(num)
        counter += 1

fizzbuzz(16)

# 3.构建模型
# 如果必须强制执行该层，则在构造函数中设置self.dynamic = True：
class MySimpleLayer(tf.keras.layers.Layer):
    def __init__(self, output_units):
        super(MySimpleLayer, self).__init__()
        self.output_units = output_units
        self.dynamic = True

    def build(self, input_shape):
        self.kernel = self.add_variable('kernel', [input_shape[-1], self.output_units])

    def call(self, inputs, *args, **kwargs):
        return tf.matmul(input, self.kernel)


class MnistMOdel(tf.keras.Model):
    def __init__(self):
        super(MnistMOdel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=10)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs, training=None, mask=None):
        result = self.dense1(inputs)
        result = self.dense2(result)
        result = self.dense2(result)       # reuse vaiables from dense2 layer
        return result

model = MnistMOdel()


# 使用eager模式训练
w = tf.Variable([1.0])
with tf.GradientTape() as tape:
    loss = w * w
grad = tape.gradient(loss, w)
print(grad)

    
# 训练一个模型
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

mnist_model = tf.keras.models.Sequential([
    layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, 1)),
    layers.Conv2D(16, [3, 3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

for images, labels in dataset.take(1):
    print('Logits: ', mnist_model(images[0:1].numpy()))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []
for (batch, (images, labels)) in enumerate(dataset.take(400)):
    if batch % 10 ==0:
        print('.', end='')
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)
        loss_value = loss_object(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
# plt.show()


# 5.变量求导优化
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.W = tf.Variable(5, dtype=tf.float32, name='weight')
        self.B = tf.Variable(10, dtype=tf.float32, name='bias')

    def call(self, inputs, training=None, mask=None):
        return inputs * self.W + self.B

# 构建数据集
NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# LOSS FUNCTION
def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.B])

model = MyModel()
optimizer = tf.keras.optimizers.SGD(0.001)
print('Initial loss: {:.3f}'.format( loss(model, training_inputs, training_outputs)))
# training
for i in range(400):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if i % 20 ==0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))


# 6.eager模式下的对象
# 变量即对象
if tf.test.is_gpu_available():
    with tf.device("gpu:0"):
        v = tf.Variable(tf.random.normal([1000, 1000]))
        v = None  # v no longer takes up GPU memory
# 对象保存
x = tf.Variable(6.0)
checkpoint = tf.train.Checkpoint(x=x)

x.assign(1.0)
checkpoint.save('./ckpt/')

x.assign(8.0)
checkpoint.restore(tf.train.latest_checkpoint('./ckpt/'))
print(x)

# 未完待续。。。