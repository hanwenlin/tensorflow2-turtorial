from __future__ import division, absolute_import, print_function, unicode_literals
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from tensorflow.keras import layers

# tf.function的一个很酷的新功能是AutoGraph，它允许使用自然的Python语法编写图形代码
# 1.tf.function装饰器

# 当使用tf.function注释函数时，可以像调用任何其他函数一样调用它。
# 它将被编译成图，这意味着可以获得更快执行，更好地在GPU或TPU上运行或导出到SavedModel。
@tf.function
def simple_nn_layer(x, y):
    return tf.nn.relu(tf.matmul(x, y))

x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

z = simple_nn_layer(x, y)

# 如果代码使用多个函数，则无需对它们进行全部注释 - 从带注释函数调用的任何函数也将以图形模式运行。
def linear_layer(x):
    return 2 * x +1

@tf.function
def deep_net(x):
    return tf.nn.relu(linear_layer(x))

print(deep_net(tf.constant((1, 2, 3))))


# 2.使用Python控制流程

# 在tf.function中使用依赖于数据的控制流时，可以使用Python控制流语句，
# AutoGraph会将它们转换为适当的TensorFlow操作。
# 例如，如果语句依赖于Tensor，则语句将转换为tf.cond（）。
@tf.function
def square_if_positive(x):
    if x>0:
        x = x*x
    else:
        x = 0
    return x

print('square_if_positive(2) = {}'.format(square_if_positive(tf.constant(2))))
print('square_if_positive(-2) = {}'.format(square_if_positive(tf.constant(-2))))


# AutoGraph支持常见的Python语句，例如while，if，break，continue和return，支持嵌套。
# 这意味着可以在while和if语句的条件下使用Tensor表达式，或者在for循环中迭代Tensor。
@tf.function
def sum_even(items):
    s = 0
    for c in items:
        if c % 2 > 0:
            continue
        s += c
    return s

print(sum_even(tf.constant([10,12,15,20])))

# AutoGraph还为高级用户提供了低级API。 例如，我们可以使用它来查看生成的代码。
print(tf.autograph.to_code(sum_even.python_function, experimental_optional_features=None))


# 3.Keras和AutoGraph
#
# 也可以将tf.function与对象方法一起使用。 例如，可以通过注释模型的调用函数来装饰自定义Keras模型。
class CustomModel(tf.keras.models.Model):

    @tf.function
    def call(self, input_data):
        if tf.reduce_mean(input_data) > 0:
            return input_data
        else:
            return input_data//2

model = CustomModel()
model(tf.constant([-2, -4]))

# 副作用 就像在eager模式下一样，你可以使用带有副作用的操作，
# 比如通常在tf.function中的tf.assign或tf.print，它会插入必要的控件依赖项以确保它们按顺序执行。
v = tf.Variable(5)
@tf.function
def find_next_odd():
  v.assign(v + 1)
  if tf.equal(v % 2, 0):
    v.assign(v + 1)


find_next_odd()


# 4.用AutoGraph训练一个简单模型
def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32)/255.
    y = tf.cast(y, tf.int64)
    return x, y

def mnist_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    traindb = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    testdb = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    traindb = traindb.map(prepare_mnist_features_and_labels)
    traindb = traindb.take(20000).shuffle(20000).batch(100)
    return traindb

train_dataset = mnist_dataset()
model = tf.keras.models.Sequential([
    layers.Reshape(target_shape=(28*28,), input_shape=(28, 28)),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.build()
optimizer = tf.keras.optimizers.Adam()
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = compute_loss(y, logits)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    compute_accuracy(y, logits)
    return loss_value

@tf.function
def train(model, optimizer):
    train_ds = mnist_dataset()
    step = 0
    loss = 0.0
    accuracy = 0.0
    for x, y in train_ds:
        step += 1
        loss = train_one_step(model, optimizer, x, y)
        if tf.equal(step % 10, 0):
            tf.print('Step', step, ': loss', loss, ': accuracy', compute_accuracy.result())
    return step, loss, accuracy

step, loss, accuracy = train(model, optimizer)
print('Final step', step, ': loss', loss, '; accuracy', compute_accuracy.result())


# 5.关于批处理的说明

# 在实际应用中，批处理对性能至关重要。
# 转换为AutoGraph的最佳代码是在批处理级别决定控制流的代码。
# 如果在单个示例级别做出决策，请尝试使用批处理API来维护性能。

def square_if_positive(x):
    return [i**2 if i>0 else i for i in x]

print(square_if_positive(range(-5, 5)))

# 在tensorflow中上面的代码应该改成下面所示
@tf.function
def square_if_positive_naive(x):
    result = tf.TensorArray(tf.int32, size=x.shape[0])
    for i in tf.range(x.shape[0]):
        if x[i] > 0:
            result = result.write(i, x[i]**2)
        else:
            result = result.write(i, x[i])
    return result.stack()

print(square_if_positive_naive(tf.range(-5, 5)))


# 也可以这么写,
def square_if_positive_vectorized(x):
  return tf.where(x > 0, x ** 2, x)