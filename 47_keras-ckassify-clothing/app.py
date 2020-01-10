from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

print('TensorFlow version：', tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 对标签进行命名
class_names = ['T恤/上衣', '裤子', '毛衣', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包包', '短靴']
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 探索模型
print(train_images.shape)  # (60000, 28, 28)

# 训练集标签的长度
# print(len(train_labels))  # 60000 标签

# 打印标签的值
# print(train_labels)  # [9 0 0 ... 3 0 5]

# 打印测试图片格式
# print(test_images.shape)  # (10000,28,28)

# 打印测试集的长度
# print(len(test_labels))  # 10000

# 预处理数据
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(True)
plt.show()

# 将值缩小到0-1之间，为此需要除以225，训练集和测试集一样，这样0-255会被缩小到 0-1的范围内
train_images = train_images / 255.0
test_images = test_images / 255.0

# 验证格式，显示训练集前25个图像
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]], fontproperties=font)
plt.show()

# 建立模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型>
# 喂给模型,会打印损失和进度
model.fit(train_images, train_labels, epochs=10)
# 结果=> 60000/60000 [==============================] - 8s 138us/sample - loss: 0.2373 - accuracy: 0.9113

# 评估
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('测试精度：', test_acc)

# 预测
predictions = model.predict(test_images)
print(len(predictions))
print(predictions[999])

# 最高预测度
print(np.argmax(predictions[999]))  # 9 获取数组中最大值的索引值

# 检查这个图片打印的结果
print(test_labels[999], class_names[test_labels[999]])  # 9


# 选10个图片做验证

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100 * np.max(predictions_array),
        class_names[true_label]),
        color=color,
        fontproperties=font
    )


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 验证预测1
# i = 12  # 0-10000随便改
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i], test_labels)
# plt.show()

# 预测测试集前25张图片,蓝色正确，红色错误，当然模型可能是错误的
num_rows = 5
num_cols = 5
num_images = num_rows * num_cols

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 使用经过训练的模型去测试单个图片做预测
img = test_images[1]
print(img.shape)

# 优化处理一批
img = (np.expand_dims(img, 0))
print(img.shape)

# 现在预测图片的正确标签
predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
plt.xticks(range(10), class_names, rotation=45)

print(np.argmax(predictions_single[0]))  # 2
