from __future__ import division, absolute_import, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32')/255.
x_test = x_test.reshape(-1, 784).astype('float32')/255.

model = tf.keras.models.load_model('the_save_model.h5')
prediction = model.predict(x_test[:2000])
y_test_top20 = y_test[:2000]
y_pred=  tf.argmax(prediction, axis=1)
correct_num = tf.reduce_sum(tf.cast(tf.equal(y_test_top20, y_pred), dtype=tf.int32))

