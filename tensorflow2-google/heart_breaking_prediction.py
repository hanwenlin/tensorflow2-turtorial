from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
column_names = ['age', 'sex', ]
dataframe = pd.read_csv('processed.cleveland.data', )
print(dataframe.head())

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# 使用 tf.data创建输入管道
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# 理解输入管道
for feature_batch, label_batch in train_ds.take(1):
    print('Evary feature:', list(feature_batch.keys()))
    print('A batch of ages: ', feature_batch['age'])
    print('A batch of targets: ', label_batch)


# 几种类型的特征列
example_batch = next(iter(train_ds))[0]

# 用于创建特征列和转换批量数据
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


# 数字列
age = feature_column.numeric_column('age')
demo(age)

# 桶列
# 不希望直接将数字输入模型，而是将数值范围分成不同的类型,有点类似柱状图
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

# 分类列
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

# 嵌入列
# 不使用独热，将数据表示位低维密集向量
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

# 哈希特征列
thal_hashed = feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=1000)
# 可以不用管分成几类，这个hash_bucket_size一般大于类别数，可能会有冲突，不同字符串映射到一个桶中
demo(thal_hashed)


# 交叉特征列
# 多个特征组合成单个特征
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(crossed_feature)

# 选择要使用的列
feature_columns = []

# 数字列numeric
for  header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

# bucketized 分桶列
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator 指示符列
thal = feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding 嵌入列
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed 交叉列
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)
# -----------------------------上面很多是练习-------------------------------------------
# 创建特征层
feature_layer = tf.keras.layers.DenseFeatures(feature_column)
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# 创建 编译和训练模型
model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10)

loss, accuracy = model.evaluate(test_ds)
print('Accuracy', accuracy)

