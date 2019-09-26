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

```txt
pip install tensorflow 

各种报错：  
ERROR: Could not find a version that satisfies the requirement tensorflow (from versions: none)
ERROR: No matching distribution found for tensorflow

anaconda 安装tensorflow：

conda install --channel https://conda.anaconda.org/conda-forge tensorflow
```

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