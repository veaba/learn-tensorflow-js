from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

