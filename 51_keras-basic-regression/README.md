# 基本回归：预测燃油效率

在 回归 (regression) 问题中，我们的目的是预测出如价格或概率这样连续值的输出。相对于分类(classification) 问题，分类(classification) 的目的是从一系列的分类出选择出一个分类 （如，给出一张包含苹果或橘子的图片，识别出图片中是哪种水果）。

本 notebook 使用经典的 [Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg) 数据集，构建了一个用来预测70年代末到80年代初汽车燃油效率的模型。为了做到这一点，我们将为该模型提供许多那个时期的汽车描述。这个描述包含：气缸数，排量，马力以及重量。

本示例使用 [tf.keras API](https://tensorflow.google.cn/api_docs/python/tf/keras)，相关细节请参阅 [本指南](https://tensorflow.google.cn/guide/keras)。

## 引用

- [回归原文](https://tensorflow.google.cn/tutorials/keras/regression)

## 依赖

```shell
 pip install -q seaborn # conda 自带了这些依赖
```

版本检查

```python
from __future__ import absolute_import,division,print_function,unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

import tensorflow as tf 
from tensorflow import keras
from tensorlfow.keras import layers

```

## Auto MPG 数据集

该数据集可以从[UCI机器学习库](https://archive.ics.uci.edu/ml/)中获取

### 获取数据

首先下载数据集

```shell
dataset_path=keras.utils.get_file("auto-mpg.data","")
```

上面的数据结构是这样

```text
MPG   气缸 排量        马力      重量        加速   车型 起源
18.0   8   307.0      130.0      3504.      12.0   70  1	"chevrolet chevelle malibu"
15.0   8   350.0      165.0      3693.      11.5   70  1	"buick skylark 320"
18.0   8   318.0      150.0      3436.      11.0   70  1	"plymouth satellite"
16.0   8   304.0      150.0      3433.      12.0   70  1	"amc rebel sst"
...

```

### 数据清洗
数据集中包括一些未知值，需要处理下


```python
dataset.isna().sum()
```

```text
MPG             0
Cylinders       0
Displacement    0
Horsepower      6
Weight          0
Acceleration    0
Model Year      0
Origin          0
dtype: int64
```

为了保证这个初始示例的简单性，删除这些行

```txt
dataset=dataset.dropna()
```

`Origin` 列实际上代表分类，而不仅仅是一个数字，所以把它转换为`独热码`(one-hot):

```python
origin=dataset.pop('Origin')
```

```python
dataset['USA']=(origin==1)*1.0
dataset['Europe']=(origin==2)*1.0
dataset['Japan']=(origin==3)*1.0
dataset.tail()

```

||MPG|气缸|排量|马力|重量|加速|车型|USA|Europe|Japan|
|---|---|---|---|---|---|---|---|---|---|---|
393	|27.0| 4 | 140.0 |	86.0 |	2790.0|	15.6|	82|	1.0	|1.0|1.0
394	|44.0| 4 | 97.0	 |  52.0 |  2130.0| 24.6|   82| 0.0	|1.0|1.0
395	|32.0| 4 | 135.0 |	84.0 |	2295.0|	11.6|	82|	1.0	|1.0|1.0
396	|28.0| 4 | 120.0 |	79.0 |	2625.0|	18.6|	82|	1.0	|1.0|1.0
397	|31.0| 4 | 119.0 |	82.0 |	2720.0|	19.4|	82|	1.0	|1.0|1.0

### 拆分训练数据集和测试数据集

现在需要将数据集拆分一个训练数据集和一个测试数据集。

我们最后将使用测试数据集对模型进行评估。


```python

```

### 数据检查

### 从标签中分离特征

### 数据规范化

## 模型

### 构建模型

### 检查模型

### 训练模型

### 做预测

## 结论