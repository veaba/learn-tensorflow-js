# 识别手写灰度化图片新手项目-python版本

## tensorflow 
 
 分为CPU 版本和GPU(需要显卡支持) 版本


- [Windows10下搭建TensorFlow环境（GPU版本）](https://www.jianshu.com/p/6f34945020f6)
- [安装cuda，如果需要安装TensorFlow GPU需要支持的话](https://developer.nvidia.com/cuda-toolkit-archive)
- tensorflow 不支持 python 3.7~~
- [解决Python3.7不能安装tensorflow<1.13的问题](https://www.jianshu.com/p/1a3e194886b4)
- numpy Python进行科学计算的基础软件包
- [tensorflow demo](https://tensorflow.google.cn/js/demos/?hl=zh_cn)
- [❤ tensorflow.js 更多的demo](https://github.com/tensorflow/tfjs-examples/)
- [python+opencv图像处理（一）](https://www.cnblogs.com/qiangayz/p/9569967.html)

- [机器学习-MNIST机器学习入门](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html)
- [10分钟教你用python 30行代码搞定简单手写识别！](https://www.cnblogs.com/dengfaheng/p/10959153.html)

```txt
pip install tensorflow 

各种报错：  
ERROR: Could not find a version that satisfies the requirement tensorflow (from versions: none)
ERROR: No matching distribution found for tensorflow

anaconda 安装tensorflow：

conda install --channel https://conda.anaconda.org/conda-forge tensorflow
```

## 第一个tensorflow程序
```python
import tensorflow as tf 
message= tf.constant("welcome to the exciting world of deep neural networks")
with tf.compat.v1.Session() as sess:
    print(sess.run(message).decode())

```
移除警告：
```python
import tensorflow as tf 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
message= tf.constant("welcome to the exciting world of deep neural networks")
with tf.compat.v1.Session() as sess:
    print(sess.run(message).decode())
```


## tensorflow 张量
- 常量：值无法改变
- 变量：
- 占位符：

### tf 常量
t1= tf.constant(444)

tf.zeros()所有元素为0的张量，
tf.zeros([M,N],tf.dtype)

### tf 变量

- 无法被直接用于seesion run
- 需要初始值：sess.run(a1.initializer)

### tf 占位符



## 计算汉明距离
## 平均哈希法
## 感知哈希算法
## dHash算法
## 差值感知算法
## 方差计算

## 手写MNIST 项目结构

[MNIST](http://yann.lecun.com/exdb/mnist/)

```
│  app-tf.py
│  opencv.py
│  README.md
│  scrapy.py
│
└─MNIST_data
        t10k-images-idx3-ubyte.gz
        t10k-labels-idx1-ubyte.gz
        train-images-idx3-ubyte.gz
        train-labels-idx1-ubyte.gz

```

> python mnist.py 就可以走了


## 数学模型：softmax Regression 

## RNN：递归神经网络
## CNN：卷积神经网络
## DBN：深度置信网络

## 遗传式神经网络/继承式神经网络/家族式神经网络/守望者神经网络（瞎扯的，2019年9月26日16:36:52）
- 训练100次
- 第一次的结果指导第二次
- 就像家族一样，爸爸(B)指导儿子(C)，爷爷(A)指导爸爸的同时也指导儿子，此时（A）成功率只有0.000001%，但C的结果有很大的一份继承上一辈的判断为依据
- 训练100次后，就有了一个很大的庞大家族，假如第一代到第100代都没死，然后A生了B之后，再生了B-2,B-2又生了C-2，这样，就有1**100 的层级关系网络，哇，这个关系就复杂了，H-N可以知道任意的Z-N的一代，当然会分配一定的参数在里面作为下一代的判断依据。
- 把训练次数提高到10000，...，1000w，看最后一代的判断结果是否正确
- 然后把1000w的最后一代作为新的一次轮回的第一代，再传承100代、
- 任何上一辈都可以指导下一辈任意代的结果
- 然后用最后一代或者增加人为调参，筛选出来几率高的辈数，和传承节点，开启新一轮的迭代更新。

