from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:10000]
test_labels = test_labels[:10000]

train_images = train_images[:10000].reshape(-1, 28*28)/255.0
test_images = test_images[:10000].reshape(-1, 28*28)/255.0

def create_model():
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation='relu',input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

#
# model = create_model()
# model.summary()

checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True, verbose=2)

model = create_model()
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels),
          callbacks=[cp_callback])
# model.fit(train_images, train_labels,  epochs = 10,
#           validation_data = (test_images,test_labels),
#           callbacks = [cp_callback])

model2 = create_model()
loss, acc = model2.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# 从检查点加载权重，并重新评估
model3 = create_model()
model3.load_weights(checkpoint_path)
loss, acc = model3.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# 检查点选项，每5个周期保存一次唯一命名检查点
# checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     checkpoint_path, verbose=1, save_weights_only=True, period=5
# )
#
# model =  create_model()
# model.save_weights(checkpoint_path.format(epoch=0))
# model.fit(train_images, train_labels, validation_data=(test_images, test_labels),
#           epochs=20, callbacks=[cp_callback], verbose=0)


# 保存整个模型
# 作为HDF5文件
model = create_model()
model.fit(train_images, train_labels,epochs=5)
model.save('my_model.h5')

new_model = keras.models.load_model('my_model.h5')
new_model.summary()

# 此方法可以保存模型的所有东西
# 权重值
# 模型的配置（架构）
# 优化器配置


# 作为saved_model
model = create_model()

model.fit(train_images, train_labels, epochs=5)
import time
saved_model_path = "./saved_models/{}".format(int(time.time()))

# tf.keras.experimental.export_saved_model(model, saved_model_path)
# new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
# new_model.summary()