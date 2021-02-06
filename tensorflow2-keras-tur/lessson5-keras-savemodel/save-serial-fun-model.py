from __future__ import division, absolute_import, print_function, unicode_literals
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from tensorflow.keras import layers

def prepare(x, y):
    x = tf.cast(tf.reshape(x, shape=(-1, 784)), dtype=tf.float32)/255.
    return x, y

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense1')(inputs)
x = layers.Dense(64, activation='relu', name='dense2')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
traindb = tf.data.Dataset.from_tensor_slices((x_train, y_train))
testdb = tf.data.Dataset.from_tensor_slices((x_test, y_test))
traindb = traindb.shuffle(32000).batch(64).map(prepare)
testdb = testdb.shuffle(10000).batch(64).map(prepare)
x_test = x_test.reshape(-1, 784).astype('float32')/255.

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-3))
history = model.fit(traindb, epochs=10)
predictions = model.predict(x_test)

# 保存全模型
# 可以对整个模型进行保存，其保存的内容包括：
# 该模型的架构模型的权重（在训练期间学到de）
# 模型的训练配置（你传递给编译的），如果有的话
# 优化器及其状态（如果有的话）（这使您可以从中断的地方重新启动训练）

model.save('the_save_model.h5')
new_model = keras.models.load_model('the_save_model.h5')
new_prediction = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_prediction, atol=1e-6) # 预测结果一样

# 1.2 保存为SavedModel文件
# keras.experimental.export_saved_model(model, 'saved_model')
# new_model2 = keras.experimental.load_from_saved_model('saved_model')
# new_prediction2 = new_model2.predict(x_test)
# np.testing.assert_allclose(predictions, new_prediction2, atol=1e-6)

# 1.3仅保存网络结构
# 仅保持网络结构，这样导出的模型并未包含训练好的参数
config = model.get_config()
reinitialized_model = keras.Model.from_config(config)
new_prediction3 = reinitialized_model.predict(x_test)
assert abs(np.sum(predictions-new_prediction3)) >0

# 也可以使用json保存网络结构
json_config = model.to_json()
reinitialized_model2 = keras.models.model_from_json(json_config)
new_prediction4 = reinitialized_model2.predict(x_test)
assert abs(np.sum(predictions-new_prediction4)) >0


# 1.4仅保存网络参数
weights = model.get_weights()
model.set_weights(weights)
# 可以把结构和参数保存结合起来
config = model.get_config()
weights = model.get_weights()
new_model2 = keras.Model.from_config(config)  # config只能用keras.Model的这个api
new_model2.set_weights(weights)
new_prediction5 = new_model2.predict(x_test)
np.testing.assert_allclose(predictions,new_prediction5, atol=1e-6)

# 1.5完整的模型保存方法
json_config2 = model.to_json()
with open('model_config.json', 'w') as json_file:
    json_file.write(json_config)

model.save_weights('path_to_my_weights.h5')

with open('model_config.json') as json_file:
    json_config3 = json_file.read()
new_model3 = keras.models.model_from_json(json_config3)
new_model3.load_weights('path_to_my_weights.h5')

new_prediction6 = new_model3.predict(x_test)
np.testing.assert_allclose(predictions, new_prediction6, atol=1e-6)

# 当然也可以一步到位
model.save('path_to_my_model.h5')
del model
model = keras.models.load_model('path_to_my_model.h5')

# 1.6保存网络权重为SavedModel格式
model.save_weights('weight_tf_savedmodel')
model.save_weights('weight_tf_savedmodel_h5', save_format='h5')
