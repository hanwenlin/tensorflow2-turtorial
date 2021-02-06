from __future__ import division, absolute_import, print_function, unicode_literals
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from tensorflow.keras import layers

# 1.7子类模型参数保存

# 子类模型的结构无法保存和序列化，只能保持参数
class ThreeLayerMLP(keras.Model):
    def __init__(self, name=None):
        super(ThreeLayerMLP, self).__init__(name=name)
        self.dense1 = layers.Dense(64, activation='relu', name='dense1')
        self.dense2 = layers.Dense(64, activation='relu', name='dense2')
        self.pred_layer = layers.Dense(10, activation='softmax', name='pred_layer')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.pred_layer(x)

def get_model():
    return ThreeLayerMLP(name='2_layer_mlp')

model = get_model()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

model.fit(x_train, y_train, epochs=1, batch_size=64)

# 保存权重参数
model.save_weights('my_model_weight', save_format='tf')

# 输出结果，供后面对比
predictions = model.predict(x_test)
first_batch_loss = model.train_on_batch(x_train[:64], y_train[:64])

# 读取保存的模型参数
new_model = get_model()
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam')

new_model.load_weights('my_model_weight')
new_prediction = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_prediction, atol=1e-6)
new_first_batch_loss = new_model.train_on_batch(x_train[:64], y_train[:64])
# assert first_batch_loss == new_first_batch_loss