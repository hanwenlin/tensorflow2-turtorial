import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1.载入数据
vocab_size = 10000
(train_x, train_y), (test_x, test_y) = keras.datasets.imdb.load_data(num_words=vocab_size)
# print(train_x[0])
# print(train_x[1])
word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
reserve_word_index = {v:k for k, v in word_index.items()}

def decode_review(text):
    return ' '.join([reserve_word_index.get(i, '?') for i in text])

print(decode_review(train_x[0]))

maxlen = 500
train_x = keras.preprocessing.sequence.pad_sequences(train_x, value=word_index['<PAD>'], padding='post', maxlen=maxlen)
test_x = keras.preprocessing.sequence.pad_sequences(test_x, value=word_index['<PAD>'], padding='post', maxlen=maxlen)

# 2.构建模型
embedding_dim = 200
model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(optimizer='adam',
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=20, batch_size=512, validation_split=0.2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Triaing and Validation accuracy')
plt.xlabel('EPochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
# plt.figure(figsize=(16, 9))
plt.show()

# 3.导出词嵌入
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)
out_v = open('vecs.tsv', 'w')
out_m = open('meta.tsv', 'w')
for word_num in range(vocab_size):
    word = reserve_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + '\n')
    out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
out_v.close()
out_m.close()