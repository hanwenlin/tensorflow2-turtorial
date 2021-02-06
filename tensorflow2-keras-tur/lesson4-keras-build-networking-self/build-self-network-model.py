from __future__ import division, absolute_import, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from tensorflow.keras import layers

# 通常，我们使用Layer类来定义内部计算块，
# 并使用Model类来定义外部模型 - 即要训练的对象。

# Model类与Layer的区别：
# 它公开了内置的训练，评估和预测循环（model.fit(),model.evaluate(),model.predict()）。
# 它通过model.layers属性公开其内层列表。
# 它公开了保存和序列化API。

# 下面通过构建一个变分自编码器（VAE），来介绍如何构建自己的网络。
# sampleing network
class Sampleing(layers.Layer):
    def call(self, inputs, *args, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon

# encoder
class Encoder(layers.Layer):
    def __init__(self,laten_dim=32, intermediate_dim=64,
                 name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = layers.Dense(laten_dim)
        self.dense_log_var = layers.Dense(laten_dim)
        self.sampling = Sampleing()
    def call(self, inputs, *args, **kwargs):
        h1 = self.dense_proj(inputs)
        z_mean = self.dense_mean(h1)
        z_log_var = self.dense_log_var(h1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

# decoder
class Decoder(layers.Layer):
    def __init__(self, original_dim,
                 intermediate_dim=64, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')
    def call(self, inputs, *args, **kwargs):
        h1 = self.dense_proj(inputs)
        return self.dense_output(h1)

# 变分自编码器
class VAE(keras.Model):
    def __init__(self, original_dim, laten_dim=32,
                 intermediate_dim=64, name='encoder', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(laten_dim=laten_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_sum(z_log_var- tf.square(z_mean)-tf.exp(z_log_var)+1)
        self.add_loss(kl_loss)
        return reconstructed

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
vae = VAE(784,32,64)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
# vae.fit(x_train, x_train, epochs=3, batch_size=64)


# 自己编写训练方法
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset  =train_dataset.shuffle(buffer_size=1024).batch(64)
original_dim  =784
vae = VAE(original_dim, 64, 32)
optimizer = tf.keras.optimizers.Adam(1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

# every epoch iteration
for epoch in range(4):
    print('Start of epoch %d' %(epoch))
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            loss = mse_loss_fn(reconstructed, x_batch_train)
            loss += sum(vae.losses)

        grads = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(grads, vae.trainable_variables))

        loss_metric(loss)

        if step % 200 == 0:
            print('step %s: mean loss = %s' % (step, loss_metric.result()))