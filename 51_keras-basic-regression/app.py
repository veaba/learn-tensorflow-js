# 回归

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

print(tf.__version__)  # 2.0.0

# 下载dataset 数据集
# dataset_path=keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

# print("==>",dataset_path) # C:\Users\Administrator\.keras\datasets\auto-mpg.data

# 使用pandas 导入数据

dataset_path = "auto-mpg.data"

column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]

raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment="\t", sep=" ", skipinitialspace=True
                          )
dataset = raw_dataset.copy()
dataset.tail()

print("dataset===>\n", dataset)
"""
dataset===>       MPG  Cylinders  Displacement  Horsepower  Weight  Acceleration  Model Year  Origin
0    18.0          8         307.0       130.0  3504.0          12.0          70       1
1    15.0          8         350.0       165.0  3693.0          11.5          70       1
2    18.0          8         318.0       150.0  3436.0          11.0          70       1
3    16.0          8         304.0       150.0  3433.0          12.0          70       1
4    17.0          8         302.0       140.0  3449.0          10.5          70       1
..    ...        ...           ...         ...     ...           ...         ...     ...
393  27.0          4         140.0        86.0  2790.0          15.6          82       1
394  44.0          4          97.0        52.0  2130.0          24.6          82       2
395  32.0          4         135.0        84.0  2295.0          11.6          82       1
396  28.0          4         120.0        79.0  2625.0          18.6          82       1
397  31.0          4         119.0        82.0  2720.0          19.4          82       1
"""

# 数据清洗，包含一些未知值
print("dataset.isna().sum()==>\n", dataset.isna().sum())

"""
MPG             0
Cylinders       0
Displacement    0
Horsepower      6
Weight          0
Acceleration    0
Model Year      0
Origin          0
dtype: int64
"""
# # 删除一些干扰的数据
dataset = dataset.dropna()
print("dataset.dropna()===>\n", dataset)

# # Origin 转为独热码
origin = dataset.pop("Origin")
print("dataset.pop('Origin')===>\n", origin)

dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

dataset.tail()

print("Origin增加行USA、Europe、Japan===>\n", dataset)

"""
Origin增加行USA、Europe、Japan===>
       MPG  Cylinders  Displacement  Horsepower  Weight  Acceleration  Model Year  USA  Europe  Japan
0    18.0          8         307.0       130.0  3504.0          12.0          70  1.0     0.0    0.0
1    15.0          8         350.0       165.0  3693.0          11.5          70  1.0     0.0    0.0
2    18.0          8         318.0       150.0  3436.0          11.0          70  1.0     0.0    0.0
3    16.0          8         304.0       150.0  3433.0          12.0          70  1.0     0.0    0.0
4    17.0          8         302.0       140.0  3449.0          10.5          70  1.0     0.0    0.0
..    ...        ...           ...         ...     ...           ...         ...  ...     ...    ...
393  27.0          4         140.0        86.0  2790.0          15.6          82  1.0     0.0    0.0
394  44.0          4          97.0        52.0  2130.0          24.6          82  0.0     1.0    0.0
395  32.0          4         135.0        84.0  2295.0          11.6          82  1.0     0.0    0.0
396  28.0          4         120.0        79.0  2625.0          18.6          82  1.0     0.0    0.0
397  31.0          4         119.0        82.0  2720.0          19.4          82  1.0     0.0    0.0

[392 rows x 10 columns]
"""

# 拆分数据集和测试集

train_dataset = dataset.sample(frac=0.8, random_state=0)  # 这段是什么意思？
test_dataset = dataset.drop(train_dataset.index)  # 为什么要drop呢？，

print("train_dataset===>\n", train_dataset)
print("test_dataset===>\n", test_dataset)
print("sns.pairplot===>\n", train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]])
# 数据检查

# #快速查看训练集中几队列的联合分布
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# plt.show()

# # 查看总体的数据统计

train_stats = train_dataset.describe()
train_dataset.pop("MPG")
train_stats = train_dataset.transpose()
print("train_stats===>\n", train_stats)  # 和原文不太一样的数据

# #从标签中分离特征
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")


# # 数据规范化
def norm(x):
    return (x - train_stats["mean"]) / train_stats["std"]


# 归一化的数据来训练模型
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# 模型

# #构建模型

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=["mae", "mse"])
    return model


# # 检查模型，使用.summary方法来打印该模型的简单描述
model = build_model()
model.summary()
