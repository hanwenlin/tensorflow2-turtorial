from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


embedding_layer = layers.Embedding(1000, 32)

vocab_size = 10000
imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels)  = imdb.load_data(num_words=vocab_size)

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# decode_review(train_data[0])

maxlen = 500
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'],
                                                        padding='post', maxlen=maxlen)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'],
                                                       padding='post', maxlen=maxlen)


# 创建一个简单的模型
embedding_dim = 16
model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

# 编译和训练模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    epochs=30, batch_size=512, validation_split=0.2)

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))

plt.show()


# 检索学习的嵌入
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()