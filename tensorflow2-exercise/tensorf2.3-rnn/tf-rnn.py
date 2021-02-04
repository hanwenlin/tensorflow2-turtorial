from __future__ import division, unicode_literals,print_function, absolute_import
import tensorflow as tf
from tensorflow import keras

imdb = keras.datasets.imdb
vocab_size = 10000
index_from = 3
max_length = 500
embedding_dim = 16
batch_size = 128

word_index = imdb.get_word_index()

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size, index_from=index_from)
word_index = {k:(v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<END>'] = 3

reverse_word_index = dict([(value, key) for key, value in word_index.items()])

def decode_review(text_ids):
    return ' '.join([reverse_word_index(word_id, '<UNK>') for word_id in text_ids])


# padding
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=max_length
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=max_length
)

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=5,
                    batch_size=batch_size, validation_split=0.2)



# 普通rnn
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.SimpleRNN(units=64, return_sequences=False),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

bi_rnn_model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.SimpleRNN(units=32, return_sequences=False)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, 'sigmoid')
])

