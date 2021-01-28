from __future__ import absolute_import, print_function, division, unicode_literals
import tensorflow as tf
from tensorflow  import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


# 使用tfds下载数据集，数据集附带一个内置的字字标记器
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

tokenizer = info.features['text'].encoder

