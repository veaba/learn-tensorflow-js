# IMDB 分类
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print('Version:', tf.__version__)  # 2.0.0
print('Eager mode:', tf.executing_eagerly())  # True
print('Hub Version:', hub.__version__)  # 0.7.0
print("GPU is",
      "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")  # NOT AVAILABLE
