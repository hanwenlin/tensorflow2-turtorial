import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from IPython.display import  SVG

# 自动编码器的两个主要组成部分; 编码器和解码器
# 编码器将输入压缩成一小组“编码”（通常，编码器输出的维数远小于编码器输入）
# 解码器然后将编码器输出扩展为与编码器输入具有相同维度的输出
# 换句话说，自动编码器旨在“重建”输入，同时学习数据的有限表示（即“编码”）

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28*28))/255.
x_test = x_test.reshape((-1, 28*28))/255.

print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

# 2.简单的自编码器
code_dim = 32
inputs = layers.Input(shape=(784,), name='inputs')
code = layers.Dense(code_dim, activation='relu', name='code')(inputs)
output = layers.Dense(x_train.shape[1], activation='softmax', name='output')(code)

auto_encoder = keras.Model(inputs=inputs, outputs=output)
auto_encoder.summary()
keras.utils.plot_model(auto_encoder, show_shapes=True)

encoder = keras.Model(inputs,code)
keras.utils.plot_model(encoder, show_shapes=True)


decoder_input = keras.Input((code_dim,))
decoder_output = auto_encoder.layers[-1](decoder_input)
decoder = keras.Model(decoder_input, decoder_output)
keras.utils.plot_model(decoder, show_shapes=True)


auto_encoder.compile(optimizer='adam',
                    loss='binary_crossentropy')

early_stop = keras.callbacks.EarlyStopping(patience=2, monitor='loss')
history = auto_encoder.fit(x_train, x_train, batch_size=64,
                           epochs=100,validation_split=0.2, callbacks=[early_stop])


encoded = encoder.predict(x_test)
decoded = decoder.predict(encoded)



plt.figure(figsize=(10,4))
n = 5
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n+i+1)
    plt.imshow(decoded[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()