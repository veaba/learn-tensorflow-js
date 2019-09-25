# Tensorflow.js 循环手写字体

- [演示代码教程](https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html#1)



## 创建函数
### tf.tensor(values, shape?, dtype?)
```js
// 创建向量 tensor
tf.tensor([1,2,3,4]).print()

/*
[1, 2, 3, 4]
*/

// 向量纬度?
tf.tensor([[1,2],[3,4]]).print()
/*
    [[1, 2],
    [3, 4]]
*/

// 传递平面数组并自己指定形状。
tf.tensor([1,2,3,4],[2,2]).print()
/*
[
    [1, 2],
    [3, 4]
]
*/

/*
tensor(values,shape,dtype)

- dtype ('float32'|'int32'|'bool'|'complex64'|'string')


*/
```
### tf.scalar(value,dtype?)
```js
tf.scalar(3.14).print()
//  3.140000104904175
```

### tf.tensor1d(values,dtype?)

```js
// 使用tf.tensor（）也可以实现相同的功能，但通常我们建议使用tf.tensor1d（），因为它使代码更具可读性。
tf.tensor2d([[1,2,3],[4,5,6]]).print()
/*
[
    [1, 2, 3],
    [4, 5, 6]
]

*/ 

```

### tf.tensor2d(values,shape?,dtype?)

```js

tf.tensor2d([1, 2, 3, 4], [2, 2]).print();

/*
[
    [1, 2],
    [3, 4]
]
  
*/
```

### tf.tensor3d (values, shape?, dtype?) 

```js
tf.tensor3d([[[1], [2]], [[3], [4]]]).print();

/*
    [
        [
            [1],
            [2]
        ],
        [
            [3],
            [4]
        ]
    ]
*/
```

### tf.tensor5d (values, shape?, dtype?) 

### tf.tensor6d (values, shape?, dtype?)

### tf.buffer (shape, dtype?, values?)

### tf.clone

### tf.complex

### tf.eye

### tf.fill

### tf.imag

### tf.linspace

### tf.oneHot

### tf.ones
### tf.onesLike
### tf.print
### tf.range
### tf.real
### tf.truncatedNormal
### tf.variable
### tf.zeros
### tf.zerosLike

## tf.Tensor 类
### .flatten
### .asScalar
### .as1D
### .as2D
### .as3D
### .as4D
### .as5D
### .asType
### .buffer
### .bufferSync
### .array
### .arraySync
### .data
### .dataSync
### .dispose
### .toFloat
### .toInt
### .toBool
### .print
### .reshape
### .reshapeAs
### .expandDims
### .cumsum
### .squeeze
### .clone
### .toString

## tf.Variable 类
### .assign

## tf.TensorBuffer
### .set
### .get
### toTensor

## 转换
### tf.batchToSpaceND
### tf.cast
### tf.depthToSpace
### tf.expandDims
### tf.pad
### tf.reshape
### tf.setdiff1dAsync
### tf.spaceToBatchND
### tf.squeeze

## 切割和连接(slicing and joining)
### tf.concat
### tf.gather
### tf.reverse
### tf.slice
### tf.split
### tf.stack
### tf.tile
### tf.unstack

## 随机
### tf.multinomial
### tf.randomNormal
### tf.randomUniform

## 模型-创建
### tf.sequential
### tf.model

## 模型-input
### tf.input

## 模型-加载
### tf.loadGraphModel
### tf.loadLayersModel
### tf.io.browserDownloads
### tf.io.browserFiles
### tf.io.http;

## 模型-管理
### tf.io.copyModel
### tf.io.listModels
### tf.io.moveModel
### tf.io.removeModel

## 模型-序列化

### tf.registerClass

## 模型-类-tf.GraphModel
### .predict
### .execute
### .executeAsync
### .dispose

## 模型：类-tf.LayersModel

### .summary
### .compile
### .evaluate
### .evaluateDataset
### .predict
### .predictOnBatch
### .save
### .geLayer

## 模型：类-tf.Sequential 

### .add
### .summary
### .evaluate
### .evaluateDataset
### .predict
### .fit
### .fitDateset
### .trainOnBatch

## 模型：类-tf.SymbolicTensor

## 模型：运算寄存器
### tf.deregisterOp
### tf.getRegisteredOp
### tf.registerOp

## Layers（图层）：高级激活
### tf.layers.elu
### tf.layers.leakyReLU
### tf.layers.prelu
### tf.layers.reLU
### tf.layers.softmax
### tf.layers.thresholdedReLU

## Layers（图层）：基础
### tf.layers.activation
### tf.layers.dense
### tf.layers.dropout
### tf.layers.embedding
### tf.layers.flatten
### tf.layers.permute
### tf.layers.repeatVector
### tf.layers.reshape

## Layers（图层）：卷积
### tf.layers.conv1d
### tf.layers.conv2d
### tf.layers.conv2dTranspose
### tf.layers.conv3d
### tf.layers.cropping2D
### tf.layers.depthwiseConv2d
### tf.layers.separableConv2d
### tf.layers.upSampling2d

## Layers（图层）：合并

### tf.layers.add
### tf.layers.average
### tf.layers.concatenate
### tf.layers.dot
### tf.layers.maximum
### tf.layers.minimum
### tf.layers.multiply

## Layers（图层）：标准化
### tf.layers.batchNormlization

## Layers（图层）：池
### tf.layers.averagePooling1d
### tf.layers.averagePooling2d
### tf.layers.globalAveragePooling1d
### tf.layers.globalAveragePooling2d
### tf.layers.globalMaxPooling1d
### tf.layers.globalMaxPooling2d
### tf.layers.maxPooling1d
### tf.layers.maxPooling2d


## Layers（图层）：复发 Recurrent
### tf.layers.gru
### tf.layers.gruCell
### tf.layers.lstm
### tf.layers.lstmCel
### tf.layers.rnn
### tf.layers.simpleRNN
### tf.layers.simpleRNNCel
### tf.layers.stackedRNNCells


## Layers（图层）：包装器
### tf.layers.bidirectional
### tf.layers.timeDistributed

## Layers（图层）：tf.layers
### .apply
### .countParams
### .build
### .getWeights
### .setWeights
### .addWeight
### .addLoss
### .computeOutputShape
### .getConfig
### .dispose

## Layers（图层）：tf.RNNCell

## Layers（图层）：输入
### tf.layers.inputLayer


## Layers（图层）：内边距/填充
### tf.layers.zeroPadding2d


##  Layers（图层）：噪声

### tf.layers.alphaDropout
### tf.layers.gaussianDropout
### tf.layers.gaussianNoise

## Layers（图层）：遮罩
### tf.layers.masking

## Operations（操作）:算术

### tf.add
### tf.sub
### tf.mul
### tf.div
### tf.addN
### tf.floorDiv
### tf.maximum
### tf.minimum
### tf.mod
### tf.pow
### tf.squaredDifference

## Operations（操作）:基础数学

### tf.abs
### tf.acos
### tf.acosh
### tf.asin
### tf.asinh
### tf.atan
### tf.atan2
### tf.atanh
### tf.ceil
### tf.clipByValue
### tf.cos
### tf.cosh
### tf.elu
### tf.erf
### tf.exp
### tf.expm1
### tf.floor
### tf.isFinite
### tf.islnf
### tf.isNaN
### tf.leakyRelu
### tf.log
### tf.log1p
### tf.logSigmoid
### tf.neg
### tf.prelu
### tf.reciprocal
### tf.relu
### tf.round
### tf.rsqrt
### tf.selu
### tf.sigmoid
### tf.sign
### tf.sin
### tf.sinh
### tf.softplus
### tf.sqrt
### tf.square
### tf.step
### tf.tan
### tf.tanh

## Operations（操作）:矩阵
### tf.dot
### tf.matMul
### tf.fused.matMul
### tf.norm
### tf.outerProduct
### tf.transpose

## Operations（操作）:卷积
### tf.avgPool
### tf.conv1d
### tf.conv2d(官方目录，这个重复两次))
### tf.conv2dTranspose
### tf.conv3d
### tf.depthwiseConv2d
### tf.maxPool
### tf.pool
### tf.separableConv2d

## Operations（操作）:还原
### tf.all
### tf.any
### tf.argMax
### tf.argMin
### tf.logSumExp
### tf.max
### tf.mean
### tf.min
### tf.prod
### tf.sum

## Operations（操作）:标准化
### tf.batchNorm
### tf.localResponseNormalization
### tf.logSoftmax
### tf.moments
### tf.softmax
### tf.sparseToDense

## Operations（操作）:图片/镜像
### tf.image.cropAndResize
### tf.image.nonMaxSuppression
### tf.image.resizeBilinear
### tf.image.resizeNearestNeighbor

## Operations（操作）:RNN

### tf.basicLSTMCell
### tf.multiRNNCell

## Operations（操作）:逻辑操作

### tf.equal
### tf.greater
### tf.greaterEqual
### tf.less
### tf.lessEqual
### tf.logicalAnd
### tf.logicalNot
### tf.logicalOr
### tf.logicalXor
### tf.notEqual
### tf.where
### tf.whereAsync

## Operations（操作）:扫描

### tf.cumsum

## Operations（操作）:Segment 片段
### tf.unsortedSegmentSum

## Operations（操作）:移动平均
### tf.movingAverage

## Operations（操作）:分割和插入
### tf.gatheND
### tf.scatterND
### tf.stridedSlice

## Operations（操作）:估价
### tf.confusionMatrix
### tf.topk

## Operations（操作）:频谱 Spectral
### tf.spectral.fft
### tf.spectral.ifft
### tf.spectral.irfft
### tf.spectral.rfft

## Operations（操作）:信号
### tf.signal.frame
### tf.signal.hammingWindow
### tf.signal.hannWindow

## Operations（操作）:线性代数
### tf.linalg.gramSchmidt
### tf.linalg.qr

## Traning（训练）：梯度
## Traning（训练）：优化器
## Traning（训练）：亏损
## Traning（训练）：tf.train.Optimizer类
### .minimize


## Performance(性能)：内存
### tf.tidy
### tf.dispose
### tf.keep
### tf.memory

## Performance(性能)：计时
### tf.time
### tf.nextFrame

## Performance(性能)：简介
### tf.profile

## Environment(环境)
### tf.disposeVariables
### tf.enableDebugMode
### tf.enableProdMode

## Performance(约束条件):类
### tf.constraints.Constraint

### tf.constraints.maxNorm
### tf.constraints.minMaxNorm
### tf.constraints.nonNeg
### tf.constraints.unitNorm

## Initializers(初始化器)：类
### tf.initializers.Initializer
### tf.initializers.constant
### tf.initializers.glorotNormal
### tf.initializers.glorotUniform
### tf.initializers.heNormal
### tf.initializers.heUniform
### tf.initializers.identity
### tf.initializers.leCunNormal
### tf.initializers.leCunUniform
### tf.initializers.ones
### tf.initializers.orthogonal
### tf.initializers.randomNormal
### tf.initializers.randomUniform
### tf.initializers.truncatedNormal
### tf.initializers.varianceScaling
### tf.initializers.zeros

## Regularizers(正则)
### tf.regularizers.l1
### tf.regularizers.l1l2
### tf.regularizers.l2


## DATA(数据)：创建
### tf.data.array
### tf.data.csv
### tf.data.generator
### tf.data.microphone
### tf.data.webcam

## DATA(数据)：操作
### tf.data.zip

## DATA(数据)：tf.data.CSVDataset 类
### .columnNames

## DATA(数据)：tf.data.Dataset 类
### .batch
### .concatenate
### .filter
### .forEachAsync
### .map
### .mapAsync
### .prefetch
### .repeat
### .skip
### .shuffle
### .take
### .toArray

## DATA(数据)：
## Visualization(可视化)：见
[ API docs for tfjs-vis are available here](https://js.tensorflow.org/api_vis/latest/)


## UTIL（工具类库）
### tf.util.assert
### tf.decodeString
### tf.encodeString
### tf.fetch
### tf.util.flatten
### tf.util.now
### tf.util.shuffle
### tf.util.sizeFromShape


## Backends（后端）
### tf.backend
### tf.getBackend
### tf.ready
### tf.registerBackend
### tf.removeBackend
### tf.setBackend

## Browser（浏览器）
### tf.browser.fromPixels
### tf.browser.toPixels

## Metrics（矩阵）
### tf.metrics.binaryAccuracy
### tf.metrics.binaryCrossentropy
### tf.metrics.categoricalAccuracy
### tf.metrics.categoricalCrossentropy
### tf.metrics.cosineProximity
### tf.metrics.meanAbsoluteError
### tf.metrics.meanAbsolutePercentageError
### tf.metrics.meanSquaredError
### tf.metrics.precision
### tf.metrics.recall
### tf.metrics.sparseCategoricalAccuracy

## Callbacks（回调）
### tf.callbacks.earlyStopping




