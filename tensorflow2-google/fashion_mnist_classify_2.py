from __future__ import print_function, division, unicode_literals, absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras


def prepare(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batch_size = 128

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_db = train_db.map(prepare).shuffle(60000).batch(batch_size)
test_db = test_db.map(prepare).batch(batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

epochs = 10

for epoch in range(epochs):
    for i, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            prediciton = model(x)
            # loss = loss_fn(prediciton, y)
            y = tf.one_hot(y, depth=10)
            loss_mse = tf.reduce_mean(tf.losses.MSE(y, prediciton))
            loss_ce = tf.losses.categorical_crossentropy(y, prediciton, from_logits=True)
            loss_ce = tf.reduce_mean(loss_ce)
        gradients = tape.gradient(loss_ce, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    total, correct = 0, 0
    for i, (x_test, y_test) in enumerate(test_db):
        y_predict = model(x_test)
        pred = tf.argmax(y_predict, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        # print('pred: ', pred)
        # print('y_test: ', y_test)
        correct += tf.reduce_sum(tf.cast(tf.equal(pred,y_test), dtype=tf.int32))
        total += x_test.shape[0]
    print('test correct: ', correct/total)





