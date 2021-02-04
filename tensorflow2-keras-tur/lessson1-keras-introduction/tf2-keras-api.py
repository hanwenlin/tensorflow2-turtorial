import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))

val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))

# model.fit(train_x, train_y, epochs=10, batch_size=100,
#           validation_data=(val_x, val_y))

# tf.data输入数据
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.batch(32)
dataset = dataset.repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()
test_x = np.random.random((1000, 72))
test_y = np.random.random((1000, 10))

test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_data = test_data.batch(32).repeat()

# 构建高级模型
# tf.keras.Sequential 模型是层的简单堆叠，无法表示任意模型。使用 Keras 函数式 API 可以构建复杂的模型拓扑
# 多输入模型，
# 多输出模型，
# 具有共享层的模型（同一层被调用多次），
# 具有非序列数据流的模型（例如，残差连接）

# 层实例可调用并返回张量。 输入张量和输出张量用于定义 tf.keras.Model 实例。
# 此模型的训练方式和 Sequential 模型一样。
input_x = tf.keras.Input(shape=(72,))
hidden1 = layers.Dense(32, activation='relu')(input_x)
hidden2 = layers.Dense(16, activation='relu')(hidden1)
pred = layers.Dense(10, activation='softmax')(hidden2)

model = tf.keras.Model(inputs=input_x, outputs=pred)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=32, epochs=5)

# 模型子类化
# 通过对 tf.keras.Model 进行子类化并定义您自己的前向传播来构建完全可自定义的模型。
# 在 init 方法中创建层并将它们设置为类实例的属性。在 call 方法中定义前向传播
class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.layer1 = layers.Dense(32, activation='relu')
        self.layer2 = layers.Dense(num_classes, activation='softmax')
    def call(self, inputs):
        h1 = self.layer1(inputs)
        out = self.layer2(h1)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

model = MyModel(num_classes=10)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=16, epochs=5)

# 自定义层
# 通过对 tf.keras.layers.Layer 进行子类化并实现以下方法来创建自定义层：
# build：创建层的权重。使用 add_weight 方法添加权重。
# call：定义前向传播。
# compute_output_shape：指定在给定输入形状的情况下如何计算层的输出形状。 或者，可以通过实现 get_config 方法和 from_config 类方法序列化层。

class MyLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        self.kernel = self.add_weight(name='kernel1', shape=shape,
                                   initializer='uniform', trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

model = tf.keras.Sequential(
[
    MyLayer(10),
    layers.Activation('softmax')
])


model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=16, epochs=5)


# 回调
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(train_x, train_y, batch_size=16, epochs=5,
         callbacks=callbacks, validation_data=(val_x, val_y))

# 保持和恢复
# 权重保存
model = tf.keras.Sequential([
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.save_weights('./weights/model')
model.load_weights('./weights/model')
model.save_weights('./model.h5')
model.load_weights('./model.h5')
# 这种只保存了权重，load_weights；需要的model是和该模型一样的网络结构才可以；就是重新构建一次该网络


# 保存网络结构
# 序列化成json
import json
import pprint
json_str = model.to_json()
pprint.pprint(json.loads(json_str))
fresh_model = tf.keras.models.model_from_json(json_str)

# 保持为yaml格式  #需要提前安装pyyaml
# yaml_str = model.to_yaml()
# print(yaml_str)
# fresh_model = tf.keras.models.model_from_yaml(yaml_str)

# 保存整个模型
model = tf.keras.Sequential([
  layers.Dense(10, activation='softmax', input_shape=(72,)),
  layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=32, epochs=5)
model.save('all_model.h5')
model = tf.keras.models.load_model('all_model.h5')

# 将keras用于Estimator

# Estimator API 用于针对分布式环境训练模型。它适用于一些行业使用场景，例如用大型数据集进行分布式训练并导出模型以用于生产
model2 = tf.keras.Sequential([layers.Dense(10,activation='softmax'),
                          layers.Dense(10,activation='softmax')])

model2.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(model2)