# IMDB 分类
from __future__ import absolute_import, division, print_function, unicode_literals
# import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print('Version:', tf.__version__)  # 2.0.0
print('Eager mode:', tf.executing_eagerly())  # True
print('Hub Version:', hub.__version__)  # 0.7.0
print("GPU is",
      "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")  # NOT AVAILABLE

# 下载IMDB数据集
# -> 将训练集按照 6:4 比例切割，15000个训练样本，10000个验证样本，25000个测试样本
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
(train_data, validation_data), test_data = tfds.load(
    name='imdb_reviews',
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True
)

# 探索数据
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

# ->训练例子的数据
print('examples => ', train_examples_batch)

print('labels => ', train_labels_batch)

# 构建模型
# ->预训练文本嵌入模型，查看下嵌入文本形状
embedding = 'https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1' # 模型1 loss: 0.325,accuracy: 0.861
# embedding = 'https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim-with-oov/1' # 模型2 但 2.5%的词汇转换为未登录词桶（OOV buckets）
# embedding = 'https://hub.tensorflow.google.cn/google/tf2-preview/nnlm-en-dim50/1' # 模型3 拥有约 1M 词汇量且维度为 50 的更大的模型。
# embedding = 'https://hub.tensorflow.google.cn/google/tf2-preview/nnlm-en-dim128/1'  # 模型4 拥有约 1M 词汇量且维度为128的更大的模型。

hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
shape = hub_layer(train_examples_batch[:3])
print(shape)

# ->构建完整模型
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()  # 打印下图层统计

# ->损失函数和优化器，编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              # 损失函数：binary_crossentropy和mean_squared_error，前者更适合处理概率
              metrics=['accuracy'])
# 训练模型
# ->loss 值越小越好
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,  # 20次迭代
                    validation_data=validation_data.batch(512),
                    verbose=1)

# 评估模型
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

#
