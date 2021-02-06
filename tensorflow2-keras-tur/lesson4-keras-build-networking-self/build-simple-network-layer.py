from __future__ import division, absolute_import, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from tensorflow.keras import layers

# 网络层就是：设置网络权重和输出到输入的计算过程
class MyLayer(layers.Layer):
    def __init__(self, input_dim=32, units=32):
        super(MyLayer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                       dtype=tf.float32), trainable=True)
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(shape=(units,),
                                                     dtype=tf.float32), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias


x = tf.ones((3, 5))
my_layer = MyLayer(5, 4)
out = my_layer(x)
# print(out)

# 按上面构建网络层，图层会自动跟踪权重w和b，当然我们也可以直接用add_weight的方法构建权重
class MyLayerAddWeight(layers.Layer):
    def __init__(self, input_dim=32, unit=32):
        super(MyLayerAddWeight, self).__init__()
        self.weight = self.add_weight(shape=(input_dim, unit),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)
        self.bias = self.add_weight(shape=(unit,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)
    def call(self, inputs, *args, **kwargs):
        return tf.matmul(inputs, self.weight) + self.bias

x2 = tf.ones((3, 5))
my_layer2 = MyLayerAddWeight(5, 4)
out2 = my_layer2(x2)
# print(out2)

# 也可以设置不可训练的权重
class AddLayer(layers.Layer):
    def __init__(self, input_dim=32):
        super(AddLayer, self).__init__()
        self.sum = self.add_weight(shape=(input_dim, ),
                                   initializer=keras.initializers.Zeros(),
                                   trainable=False)

    def call(self, inputs, *args, **kwargs):
        self.sum.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.sum


x3 = tf.ones((3, 3))
my_layer3 = AddLayer(3)
out3 = my_layer3(x3)
print('first: ', out3.numpy())
out3 = my_layer3(x3)
print('second:', out3.numpy())
print('weight: ', my_layer3.weights)
print('non-trainable weight: ', my_layer3.non_trainable_weights)
print('trainable weight: ', my_layer3.trainable_weights)

# 当定义网络时不知道网络的维度是可以重写build()函数，用获得的shape构建网络
class MyLayerNonShape(layers.Layer):
    def __init__(self, unit=32):
        super(MyLayerNonShape, self).__init__()
        self.unit = unit

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.unit),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.unit,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)

    def call(self, inputs, *args, **kwargs):
        return tf.matmul(inputs, self.weight) + self.bias


my_layer4 = MyLayerNonShape(3)
x4 = tf.ones((3, 5))
out4 = my_layer4(x4)
print(out4)
my_layer4 = MyLayerNonShape(3)
x4 = tf.ones((2, 2))
out4 = my_layer4(x4)
print(out4)


# 2.使用子层递归构建网络层
class MyBlock(layers.Layer):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.layer1 = MyLayerNonShape(32)
        self.layer2 = MyLayerNonShape(16)
        self.layer3 = MyLayerNonShape(2)
    def call(self, inputs, *args, **kwargs):
        h1 = self.layer1(inputs)
        h1 = tf.nn.relu(h1)
        h2 = self.layer2(h1)
        h2 = tf.nn.relu(h2)
        return self.layer3(h2)

my_block = MyBlock()
print('trainable weights: ', len(my_block.trainable_weights))
y = my_block(tf.ones(shape=(3, 64)))
# 构建网络在build()里面，所以执行了才有网络
print('trainable weights: ', len(my_block.trainable_weights))


# 可以通过构建网络层的方法来收集loss
class LossLayer(layers.Layer):
    def __init__(self, rate=1e-2):
        super(LossLayer, self).__init__()
        self.rate = rate
    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs

class OutLayer(layers.Layer):
    def __init__(self):
        super(OutLayer, self).__init__()
        self.losses_fn = LossLayer(1e-2)
    def call(self, inputs, *args, **kwargs):
        return self.losses_fn(inputs)

my_layer5 = OutLayer()
print(len(my_layer5.losses))  # 还未call
y = my_layer5(tf.zeros(1, 1))
print(len(my_layer5.losses))  # 执行call之后
y = my_layer5(tf.zeros(1, 1))
print(len(my_layer5.losses))   # call之前会重新置0


# 如果中间调用了keras网络层，里面的正则化loss也会被加入进来
class OuterLayer(layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.dense = layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, inputs, *args, **kwargs):
        return self.dense(inputs)

my_layer6 = OuterLayer()
y = my_layer6(tf.zeros((1,1)))
print(my_layer6.losses)
print(my_layer6.weights)


# 3.其他网络层配置
# 使自己的网络层可以序列化

class Linear(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.weights = self.add_weight(shape=(input_shape[-1], self.units),
                                       initializer=tf.keras.initializers.RandomNormal(),
                                       trainable=True)
        self.bias = self.add_weight(shape=(self.units, ),
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=True)

    def call(self, inputs, *args, **kwargs):
        return tf.matmul(inputs, self.weights) + self.bias

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config


layer = Linear(125)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)

# 配置只有训练时可以执行的网络层
class MyDropout(layers.Layer):
    def __init__(self, rate, **kwargs):
        super(MyDropout, self).__init__(**kwargs)
        self.rate = rate
    def call(self, inputs,training=None):
        return tf.cond(training, lambda: tf.nn.dropout(inputs, rate=self.rate),
                       lambda:inputs)