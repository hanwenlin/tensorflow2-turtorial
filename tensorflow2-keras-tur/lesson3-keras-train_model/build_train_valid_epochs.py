import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784)
x_train, x_val = x_train[:50000], x_train[50000:]
y_train, y_val = y_train[:50000], y_train[50000:]

# 构建一个全连接网络.
inputs = keras.Input(shape=(784, ), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
model = keras.Model(inputs, outputs)

# optimizer
optimizer = tf.keras.optimizers.SGD(1e-3)
# loss
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# prepare data
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10240).batch(batch_size)

# build epochs self
# for epoch in range(3):
#     print('epoch: {}'.format(epoch))
#     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#         # 开一个gradient tape, 计算梯度
#         with tf.GradientTape() as tape:
#             logits = model(x_batch_train)
#             loss_value = loss_fn(y_batch_train, logits)
#         grads = tape.gradient(loss_value, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#         if step % 200 == 0:
#             print('Traing loss (for one batch) at step %s: %s' %(step, float(loss_value)))
#             print('Seen so far: %s samples' % ((step+1)*64))

# training and validataion
# 设定统计参数
train_acc_metric = keras.metrics.SparseCategoricalCrossentropy()
val_acc_metric = keras.metrics.SparseCategoricalCrossentropy()

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.shuffle(8000).batch(batch_size)

# 迭代训练
for epoch in range(5):
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 更新统计传输
        train_acc_metric(y_batch_train, logits)

        # output
        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s' % (step, float(loss)))
            print('Seen so far: %s samples' % ((step + 1) * 64))

    # 输出统计参数的值
    train_acc = train_acc_metric.result()
    print('Training acc over epoch: %s' % (float(train_acc),))
    # 重置统计参数
    train_acc_metric.reset_states()

    # 用模型进行验证
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val)
        # 根据验证的统计参数
        val_acc_metric(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print('Validation acc:L %s' %(float(val_acc),))

# 添加自己构造的loss
##　添加自己构造的loss, 每次只能看到最新一次训练增加的loss
class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * tf.reduce_sum(inputs))
        return inputs

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
# Insert activity regularization as a layer
x = ActivityRegularizationLayer()(x)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model2 = keras.Model(inputs, outputs)
logits = model2(x_train[:64])
print(model.losses)
logits = model2(x_train[:64])
logits = model2(x_train[64:128])
logits = model2(x_train[128:192])
print(model2.losses)

optimizer = keras.optimizers.SGD(1e-3)

for epoch in range(3):
    print('Start of epoch %d ' %(epoch))

    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model2(x)
            loss_values = loss_fn(y, logits)

            #添加额外的loss
            loss_values += sum(model2.losses)
        grads = tape.gradient(loss_values, model2.trainable_variables)
        optimizer.apply_gradients(zip(grads, model2.trainable_variables))

        # 每200个batch输出一次学习.
        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s' % (step, float(loss_values)))
            print('Seen so far: %s samples' % ((step + 1) * 64))
