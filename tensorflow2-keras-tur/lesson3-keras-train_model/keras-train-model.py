import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 构建模型
inputs = keras.Input(shape=(784,), name='mnist_input')
h1 = layers.Dense(64, activation='relu')(inputs)
h1 = layers.Dense(64, activation='relu')(h1)
outputs = layers.Dense(10, activation='softmax')(h1)
# model = keras.Model(inputs=inputs, outputs=outputs)


# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

# -----------------------------------func 1------------------------------------------
# 训练模型
# model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
#               loss=keras.losses.SparseCategoricalCrossentropy(),
#               metrics=[keras.metrics.SparseCategoricalAccuracy()])

# history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)
# print('history: ', history.history)
#
# result = model.evaluate(x_test, y_test, batch_size=128)
# print('evaluate: ')
# print(result)
# pred = model.predict(x_test[:2])
# print('predict: ', pred)

# --------------------------------- fun 2 -------------------------------------
# 自定义损失和指标
# 自定义指标只需继承Metric类，并重写下函数
# __init__(self),初始化
# update_state(self, y_true, y_pred, sample_weight=None), 它使目标y_true和模型预测y_pred来更新状态变量
# result(self) 使用状态变量来计算最终结果
# reset_states(self), 重新初始化度量的状态

# 这是一个简单的示例，显示如何实现CatgoricalTruePositives指标，该指标计算正确分类为属于给定类的样本数量
class CatgoricalTruePosives(keras.metrics.Metric):
    def __init__(self, name='binary_true_postives', **kwargs):
        super(CatgoricalTruePosives, self).__init__(name=name, **kwargs)
        self.true_postives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred)
        y_true = tf.equal(tf.cast(y_pred, dtype=tf.int32), tf.cast(y_true, dtype=tf.int32))
        y_true = tf.cast(y_true, dtype=tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            y_true = tf.multiply(sample_weight, y_true)

        return self.true_postives.assign_add((tf.reduce_sum(y_true)))

    def result(self):
        return tf.identity(self.true_postives)

    def reset_states(self):
        self.true_postives.assign(0.)


# model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
#               loss=keras.losses.SparseCategoricalCrossentropy(),
#               metrics=[CatgoricalTruePosives()])
#
# model.fit(x_train, y_train, batch_size=64, epochs=3)
#
# model.evaluate(x_test, y_test)

# ---------------------------------- fun 3 --------------------------
# 以定义网络层的方式添加网络loss
class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs)*0.1)
        return inputs

inputs = keras.Input(shape=(784,), name='mnist_input')
h1 = layers.Dense(64, activation='relu')(inputs)
h1 = ActivityRegularizationLayer()(h1)
h1 = layers.Dense(64, activation='relu')(h1)
outputs = layers.Dense(10, activation='softmax')(h1)
# model = keras.Model(inputs, outputs)
# keras.utils.plot_model(model, 'net001.png', show_shapes=True)

# model.compile(optimizer=keras.optimizers.RMSprop(),
#              loss=keras.losses.SparseCategoricalCrossentropy(),
#              metrics=[keras.metrics.SparseCategoricalAccuracy()])
# model.fit(x_train, y_train, batch_size=32, epochs=1)

# 也可以定义网络层的方式添加要统计的metric
class MetricLoggingLayer(layers.Layer):
    def call(self, inputs):
        self.add_metric(keras.backend.std(inputs),
                        name='std_of_activation',
                        aggregation='mean')

        return inputs

inputs = keras.Input(shape=(784,), name='mnist_input')
h1 = layers.Dense(64, activation='relu')(inputs)
h1 = MetricLoggingLayer()(h1)
h1 = layers.Dense(64, activation='relu')(h1)
outputs = layers.Dense(10, activation='softmax')(h1)
# model = keras.Model(inputs, outputs)
# keras.utils.plot_model(model, 'net001.png', show_shapes=True)

# model.compile(optimizer=keras.optimizers.RMSprop(),
#              loss=keras.losses.SparseCategoricalCrossentropy(),
#              metrics=[keras.metrics.SparseCategoricalAccuracy()])
# model.fit(x_train, y_train, batch_size=32, epochs=1)

# 也可以直接在model上面添加
# 也可以定义网络层的方式添加要统计的metric
inputs = keras.Input(shape=(784,), name='mnist_input')
h1 = layers.Dense(64, activation='relu')(inputs)
h2 = layers.Dense(64, activation='relu')(h1)
outputs = layers.Dense(10, activation='softmax')(h2)
model = keras.Model(inputs, outputs)
model.add_metric(keras.backend.std(inputs),name='std_of_activation',
                 aggregation='mean')

model.add_loss(tf.reduce_sum(h1)*0.1)
model.compile(optimizer=keras.optimizers.RMSprop(),
             loss=keras.losses.SparseCategoricalCrossentropy(),
             metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(x_train, y_train, batch_size=32, epochs=1)

# 处理使用validation_data传入测试数据，还可以使用validation_split划分验证数据
# ps:validation_split只能在用numpy数据训练的情况下使用
model.fit(x_train, y_train, batch_size=32, epochs=1, validation_split=0.2)

