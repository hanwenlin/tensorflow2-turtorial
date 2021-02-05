import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784)
def get_complied_model():
    inputs = keras.Input(shape=(784,), name='mnist_input')
    h1 = layers.Dense(64, activation='relu')(inputs)
    h2 = layers.Dense(64, activation='relu')(h1)
    h3 = layers.Dense(32, activation='relu')(h2)
    outputs = layers.Dense(10, activation='softmax')(h3)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model

# model = get_complied_model()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_dataset = train_dataset.shuffle(60000).batch(64)

# model.fit(train_dataset, epochs=10)

# 样本权重和类权重
# “样本权重”数组是一个数字数组，用于指定批处理中每个样本在计算总损失时应具有多少权重。 它通常用于不平衡的分类问题（这个想法是为了给予很少见的类更多的权重）。
# 当使用的权重是1和0时，该数组可以用作损失函数的掩码（完全丢弃某些样本对总损失的贡献）。
# “类权重”dict是同一概念的更具体的实例：它将类索引映射到应该用于属于该类的样本的样本权重。
# 例如，如果类“0”比数据中的类“1”少两倍，则可以使用class_weight = {0：1.，1：0.5}。

# 增加第5类的权重
# 类权重
# model = get_complied_model()
class_weight = {i:1.0 for i in range(10)}
class_weight[5] = 2.0
# model.fit(train_dataset, class_weight=class_weight, epochs=4)
# 样本权重
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train==5] = 2.0
# model.fit(x_train, y_train, sample_weight=sample_weight, epochs=4)

# model.fit(train_dataset, sample_weight=sample_weight, epochs=4)
# ValueError: `sample_weight` argument is not supported when using dataset as input.
# 对于tf.data  sample_weight应该换一种方式
train_dataset2 = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weight))
train_dataset2 = train_dataset2.shuffle(60000).batch(64)
# model.fit(train_dataset, epochs=4)

## 多输入多输出模型
image_input = keras.Input(shape=(32, 32, 3), name='img_input')
timeseries_input = keras.Input(shape=(None, 10), name='ts_input')

x1 = layers.Conv2D(3, 3)(image_input)
x1 = layers.GlobalMaxPool2D()(x1)

x2 = layers.Conv1D(3, 3)(timeseries_input)
x2 = layers.GlobalMaxPool1D(x2)

x = layers.concatenate([x1, x2])
score_output = layers.Dense(1, name='score_output')(x)
class_output = layers.Dense(5, activation='softmax', name='class_output')(x)

model = keras.Model(inputs=[image_input, timeseries_input],
                    outputs=[score_output, class_output])

keras.utils.plot_model(model, 'multi_input_output_model.png',
                       show_shapes=True)
# 可以为模型指定不通的loss和metrics
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
              loss={'score_output': keras.losses.MeanSquaredError(),
                    'class_output': keras.losses.CategoricalCrossentropy()},
              metrics={'score_output':[keras.metrics.MeanAbsolutePercentageError,
                                       keras.metrics.MeanSquaredError()],
                       'class_output':[keras.metrics.CategoricalAccuracy()]},
              loss_weight={'score_output':2, 'class_output':1})

# 可以把不需要传播的loss置为0
# model.compile(optimizer=keras.optimizers.RMSprop(),
#               loss=[None, keras.losses.CategoricalCrossentropy()])
#
# # or dict loss version
# model.compile(
#     optimizer=keras.optimizers.RMSprop(1e-3),
#     loss={'class_output': keras.losses.CategoricalCrossentropy()})