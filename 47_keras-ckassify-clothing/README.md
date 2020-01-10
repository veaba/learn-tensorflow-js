# Keras 机器学习基础知识

## 基本分类：衣服图片分类
- [原文](https://tensorflow.google.cn/tutorials/keras/classification)
- [Github源码](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb0)
- [下载训练集](https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/keras/classification.ipynb)
- [matplotlib](https://www.matplotlib.org.cn/) 一个绘图库
- numpy 一个python 基本数学库
- tensorflow 大数据框架


本指南训练一个神经网络模型来分类像运动鞋和衬衫衣服的图片。如果你不了解所有的细节没关系；这是一个完整的TensorFlow程序的快节奏概述，开始后会有解释细节。

本指南用到 [tf.keras](https://tensorflow.google.cn/guide/keras) ，一个在TensorFlow中构建和循环模型高等级的API

```python
from __future__ import  absolute_import,division,print_function,unicode_literals

# TensorFlow 和 tf.keras
import tensorflow as tf 
from tensorflow import keras 

# 辅助库

import numpy as np

import matplotlib.pyplot as plt

print(tf.__version__) # 先查看一下tensorflow 版本


```

## pip 安装依赖

> pip install tensorflow 


升级tensorflow
>conda install --channel https://conda.anaconda.org/anaconda tensorflow=2.0.0


## 导入服装 MNIST 数据集

本指南使用[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)数据集，其中包含10个类别的70000个灰度图像。这些图像以低分辨率（28 x 28像素）显示衣服的各个物品，如下所示：

![](https://tensorflow.google.cn/images/fashion-mnist-sprite.png)

图1 是[Fashion-MNIST samples](https://github.com/zalandoresearch/fashion-mnist) (by Zalando, MIT License).

时尚MNIST旨在取代经典MNIST数据集，该数据集通常用作计算机视觉机器学习程序的“Hello，World”。MNIST数据集包含手写数字（0、1、2等）的图像，其格式与您将在此处使用的衣服相同。


本指南使用时尚MNIST的多样性，因为这是一个比普通MNIST稍微更具挑战性的问题。这两个数据集都相对较小，用于验证算法是否按预期工作。它们是测试和调试代码的良好起点。

这里，使用60000个图像来训练网络，使用10000个图像来评估网络学习如何准确地分类图像。您可以直接从TensorFlow访问时尚列表。直接从TensorFlow导入和加载时尚MNIST数据：

继续往下添加下面代码，则如下：

```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```
执行一次会自动加载相关资源

加载数据集返回四个NumPy数组：

- `train_images` 和 `train_lables`数组是模型用来学习的数据训练集
- 该模型通过测试集对测试图像和测试标签数组进行测试。

图像是28x28numpy数组，像素值从0到255不等。标签是一个整数数组，范围从0到9。这些对应于图像所代表的服装类别：

|Label|Class|
|-----|-----|
|0|T恤/上衣|
|1|裤子|
|2|毛衣、套衫|
|3|连衣裙|
|4|外套、大衣|
|5|凉鞋|
|6|衬衫|
|7|运动鞋|
|8|包|
|9|短靴|
每个图像都映射到一个标签。由于类名不包含在数据集中，请将它们存储在此处，以便以后打印图像时使用：


```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

## 探索数据

在训练模型之前，让我们研究一下数据集的格式。以下显示训练集中有60000个图像，每个图像表示为28 x 28像素：


```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
```


同样，在训练集中有60000个标签：

```shell script
len(train_labels) #60000
```
每个标签都是0到9之间的整数：

```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
```

测试集中有10000个图像。同样，每个图像表示为28 x 28像素：

```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
print(test_images.shape) #(10000, 28, 28)
```
测试集包含10000个图像标签：

```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
len(test_labels) #10000
```

## 预处理数据
在训练网络之前，必须对数据进行预处理。如果检查训练集中的第一个图像，将看到像素值在0到255的范围内：

```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```
![](https://tensorflow.google.cn/tutorials/keras/classification_files/output_m4VEw8Ud9Quh_0.png)

在将这些值输入到神经网络模型之前，将其缩放到0到1的范围。为此，请将这些值除以255。训练集和测试集必须以相同的方式进行预处理：


```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images=train_images/255.0
test_images=test_images/255.0
```

为了验证数据的格式是否正确，以及您是否准备好构建和训练网络，让我们显示训练集中的前25个图像，并在每个图像下面显示类名。

```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images=train_images/255.0
test_images=test_images/255.0

# 本次添加的
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```


如图所示：

![](https://tensorflow.google.cn/tutorials/keras/classification_files/output_oZTImqg_CaW1_0.png)

## 建立模型

建立神经网络需要配置模型的`layers`，然后编译模型。

### 配置`layers`

神经网络的基本组成部分为`layers`，`layers`从输入到它们的数据中提取表示，希望这些对手头的问题有意义

大部分的深度学习都是把简单的`layers` 连在一起。大多数`layers` ，比如[`tf.keras.layers.Dense`](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Dense)，都有在训练中学习的参数。

```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images=train_images/255.0
test_images=test_images/255.0

# 本次添加的
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')        
])
```

该网络的第一层[`tf.keras.layers.Flatten`](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Flatten)将图像的格式从二维数组（28×28像素）转换为一维数组（28×28=784像素）。把这一层想象成图像中一行行的像素并将它们排列起来。此层没有要学习的参数；它只重新格式化数据。

扁平化像素后，网络由两个[`tf.keras.layers.Dense`](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Dense)层组成。这些神经层紧密相连，或完全相连。第一致密层有128个节点（或神经元）。第二层（也是最后一层）是一个10节点的`softmax`层，它返回一个由10个概率得分组成的数组，其总和为1。每个节点包含一个分数，表示当前图像属于10个类之一的概率。

### 编译模型

在模型准备好进行训练之前，它还需要一些设置。这些是在模型的编译步骤中添加的：

- **损失函数** - 这可以测量模型在训练期间的准确性。您希望最小化此函数，以便将模型`转向`到正确的方向。
- **优化器** - 这是如何根据模型看到的数据和它的损失函数来更新模型的。
- **指标** - 用于监控训练和测试步骤。下面的例子使用了精确性，即正确分类的图像的分数。

```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images=train_images/255.0
test_images=test_images/255.0

# 本次添加的
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
# 配置
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')        
])

# 编译模型
model.compile(
    optimizer='adam',                       # 优化
    loss='sparse_categorical_crossentropy', # 损失
    metrics=['accuracy']                    # 指标
)
```
### 训练模型

训练神经网络模型需要以下步骤：

1. 将训练数据输入模型。在本例中，训练数据位于`train_images`和`train_labels`数组中。
2. 模型学习关联图像和标签。
3. 在本例中，您要求模型对测试集`test_images`数组进行预测。
4. 验证预测是否与`test_labels`数组中的标签匹配。


### 喂给model
要开始训练，请调用`model.fit`方法，因为它将模型“拟合”到培训数据：

```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images=train_images/255.0
test_images=test_images/255.0

# 本次添加的
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
# 配置
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')        
])

# 编译模型
model.compile(
    optimizer='adma',                       # 优化
    loss='sparse_categorical_crossentropy', # 损失
    metrics=['accuracy']                    # 指标
)

# 喂给模型
model.fit(train_images,train_labels,epochs=10)
```
当模型训练时，将显示损失和精度度量。该模型对训练数据的精度达到0.91（91%）。

### 评估准确性

接下来，比较模型在测试数据集上的执行情况：
```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images=train_images/255.0
test_images=test_images/255.0

# 本次添加的
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
# 配置
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')        
])

# 编译模型
model.compile(
    optimizer='adma',                       # 优化
    loss='sparse_categorical_crossentropy', # 损失
    metrics=['accuracy']                    # 指标
)

# 喂给模型
model.fit(train_images,train_labels,epochs=10)

# 评估

test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)

# 10000/10000 - 1s - loss: 0.3499 - accuracy: 0.8811
print('Test accuracy：',test_acc) # Test accuracy: 0.8811

```

结果表明，测试数据集的准确性略低于训练数据集的准确性。训练精度和测试精度之间的这种差距代表了过度拟合。过度拟合是指机器学习模型在新的、以前看不见的输入上的性能比在训练数据上的性能差。


### 作出预测

通过训练模型，您可以使用它对一些图像进行预测。

```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images=train_images/255.0
test_images=test_images/255.0

# 本次添加的
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
# 配置
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')        
])

# 编译模型
model.compile(
    optimizer='adma',                       # 优化
    loss='sparse_categorical_crossentropy', # 损失
    metrics=['accuracy']                    # 指标
)

# 喂给模型
model.fit(train_images,train_labels,epochs=10)

# 评估
test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)
# 10000/10000 - 1s - loss: 0.3499 - accuracy: 0.8811
print('Test accuracy：',test_acc) # Test accuracy: 0.8811

# 预测 
predictions= model.predict(test_images)

```
在这里，模型已经为测试集中的每个图像预测了标签。让我们看看第一个预测：

```text
print(predictions[0])
```
结果：

```text
array([1.4711586e-08, 7.2586021e-10, 3.9839532e-10, 5.7181171e-09,
       1.5156877e-09, 3.4303256e-04, 2.1361195e-07, 4.7104005e-03,
       8.6297112e-09, 9.9494630e-01], dtype=float32)
```

预测是由10个数字组成的数组。它们代表了模型的“自信”，即形象对应于10种不同的服装。您可以看到哪个标签具有最高的信任值：


```text
np.argmax(predictions[0]) #9
```

因此，该模型最有信心的是，这个图像是一个踝靴，或类名称[9]。检查测试标签表明此分类正确：

```text
print(test_labels[0])
```
用这个图表来查看10个类预测的完整集合。

```python
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np  # python 基础数学库
import matplotlib.pyplot as plt  # python 绘图库
fashion_mnist = keras.datasets.fashion_mnist
# 获取训练集、测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape) # (60000, 28, 28)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images=train_images/255.0
test_images=test_images/255.0

# 本次添加的
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
# 配置
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')        
])

# 编译模型
model.compile(
    optimizer='adma',                       # 优化
    loss='sparse_categorical_crossentropy', # 损失
    metrics=['accuracy']                    # 指标
)

# 喂给模型
model.fit(train_images,train_labels,epochs=10)

# 评估
test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)
# 10000/10000 - 1s - loss: 0.3499 - accuracy: 0.8811
print('Test accuracy：',test_acc) # Test accuracy: 0.8811

# 预测 
predictions= model.predict(test_images)

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
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
```

### 验证预测

通过训练模型，您可以使用它对一些图像进行预测。

让我们看看第0幅图像、预测和预测数组。正确的预测标签为蓝色，错误的预测标签为红色。这个数字给出了预测标签的百分比（100分之一）。

```python
i=0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions[i],test_labels,test_images)
plt.subplot(1,2,2)
plot_value_array(i,predictions[i],test_labels)
plt.show()
```
![](https://tensorflow.google.cn/tutorials/keras/classification_files/output_HV5jw-5HwSmO_0.png)

```python
i=12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions[i],test_labels,test_images)
plt.subplot(1,2,2)
plot_value_array(i,predictions[i],test_labels)
plt.show()
```
![](https://tensorflow.google.cn/tutorials/keras/classification_files/output_Ko-uzOufSCSe_0.png)

让我们用他们的预测来绘制几个图像。请注意，即使在非常自信的情况下，模型也可能是错误的。

```python
# 绘制第一个X测试图像、它们的预测标签和真实标签。
# 颜色正确预测为蓝色，错误预测为红色。

num_rows=5
num_cols=3
num_images=num_rows*num_cols

plt.figure(figsize=(2*2*num_cols,2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows,2*num_cols,2*i+1)
    plot_image(i,predictions[i],test_labels,test_images)
    plt.subplot(num_rows,2*num_cols,2*i+2)
    plo_value_array(i,predictions[i],test_labels)
plg.tight_layout()
plg_show()
```
![](https://tensorflow.google.cn/tutorials/keras/classification_files/output_hQlnbqaw2Qu__0.png)

### 使用经过训练的模型

最后，利用训练后的模型对单个图像进行预测。

```python
# 从测试数据集中获取图像。

img=test_images[1]
print(img.shape) #(28, 28)
```

对[`tf.keras`](https://tensorflow.google.cn/api_docs/python/tf/keras)模型进行了优化，可以同时对一批或一组示例进行预测。因此，即使使用单个图像，也需要将其添加到列表中：

```python
# 将图像添加到只有它一个成员的批处理中。

img=(np.expand_dims(img,0))
print(img.shape) # (1,28,28)
```

现在预测此图像的正确标签：

```python
predictions_single=model.predict(img)
print(predictions_single)
```
结果

```text
[[2.3515991e-05 1.8022357e-13 9.8655802e-01 1.3430835e-13 1.3296051e-02
  5.1873236e-13 1.2246115e-04 1.5213916e-23 8.6392804e-09 1.0808098e-15]]
```

```python
plot_value_array(1,predictions_single[0],test_lables)
- =plt.xticks(range(10),class_names,rotation=45)
```

![](https://tensorflow.google.cn/tutorials/keras/classification_files/output_6Ai-cpLjO-3A_0.png)


`model.predict`返回一个列表列表，一个列表用于批数据中的每个图像。获取批次中我们（仅）图像的预测：

```python
print(np.argmax(predictions_single[0])) #2
```

模型预测了一个标签。


但是这个指南，并无涉及到如何保存模型：
将的通过训练集、测试集来训练模型，并最终验证模型的过程。
