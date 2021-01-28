from __future__ import absolute_import, print_function, division, unicode_literals
import  tensorflow as tf
from tensorflow  import keras

import numpy as np

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# 参数 num_words=10000 保留训练数据中最常出现的10,000个单词，丢弃罕见的单词

# 将整数变为文本
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(v, k) for k,v in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# 预处理数据
# 使用pad_sequencets函数来标准化长度
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                           value=word_index['<PAD>'],
                                                           padding='post', maxlen=256)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index['<PAD>'],
                                                           padding='post', maxlen=256)

# 构建模型
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train, partial_y_train,
                    batch_size=512,
                    epochs=40, validation_data=(x_val, y_val), verbose=1)

# 评估模型
results = model.evaluate(test_data, test_labels)
print(results)

# 创建准确性和损失随时间变化的图标
history_dict = history.history
print(history_dict.keys())

import matplotlib.pyplot as plt
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()