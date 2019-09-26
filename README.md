# learn-tensorflow-js

启用域名：https://datav.ai/mhw

或者名字：MHWAI

## 计划：每一天或者每一周 checkout 出来一个分支——预计100周~啊哈哈
- 一个分支一周的时间
- 这样的话就可以根据分支的变化日志来追踪整个项目的变更了（此前一直想做的）
- 先完成手写玩的demo联系
- 再此基础上做识别怪物

## Tracking Project：TODO 
- 最新的进度在最新分分支里面



## 相关学习链接
- [极客学院——TensorFlow 官方文档中文版](http://wiki.jikexueyuan.com/project/tensorflow-zh/)
- [谷歌全新开源人工智能系统TensorFlow官方文档中文版 GitHub仓库，内容同上](https://github.com/jikexueyuanwiki/tensorflow-zh)
- [TensorFlow JS版官网](https://www.tensorflow.org/js/)
- [3行代码就能通过浏览器从任意方向跟踪图像中的手的动作Handtrack.js](https://towardsdatascience.com/handtrackjs-677c29c1d585)
- [handtrack.js github仓库](https://github.com/victordibia/handtrack.js)
- [使用TensorFlow进行训练识别视频图像中物体-文章展示](http://ask.ainoob.cn/article/69)
- [python opencv 教程](https://docs.opencv.org/3.4/d6/d00/tutorial_py_root.html)
- [ffmpeg.win32，!!!暂无地址]()

- [[AI开发]Python+Tensorflow打造自己的计算机视觉API服务-文章](https://www.cnblogs.com/xiaozhi_5638/p/9268715.html)
- [vision-web-service-上面文章配套的源码](https://github.com/sherlockchou86/vision-web-service)

- [三分钟训练眼球追踪术，AI就知道你在盯着哪个妹子 | TensorFlow.js代码](https://cloud.tencent.com/developer/article/1349501)

- [tfjs-vis 是一个用于在浏览器内实现可视化的小型库，用于TensorFlow.js。](https://github.com/tensorflow/tfjs-vis)
- [tfjs-demo](https://github.com/tensorflow/tfjs-examples)

- [tfjs：Real-time Human Pose Estimation in the Browser with TensorFlow.js-文章](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5)


- [Go语言跨平台自动化Gui系统,控制鼠标和键盘等](https://github.com/go-vgo/robotgo)

- [Spy++ —— 窗口、消息查看分析利器-文章](https://blog.csdn.net/qq_25408423/article/details/80884114)

- [ PE Explorer 可以查看DLL 有哪些输出函数]()

- [用Go语言写一个Windows的外挂 (上)](https://studygolang.com/articles/9862)

- [Python去控制鼠标键盘：PyAutoGUI 文章](https://www.cnblogs.com/fanghao/p/8453207.html)

- ❤[tensorflow.js 更多的demo](https://github.com/tensorflow/tfjs-examples/)

- [基于Anaconda 安装tensorflow](https://www.jianshu.com/p/42a939285d7a)

- [Anaconda 更改源](https://jingyan.baidu.com/article/1876c8527be1c3890a137645.html)

- [scrapy 爬虫工具，python 依赖包]()

- 国内镜像安装opencv-python
> pip install opencv-python -i  https://pypi.douban.com/simple

## 项目背景

> 来源，最近在玩一款单机游戏——《怪物猎人：世界》，水平很菜，在肝中。挺好玩，但是呢，这游戏没显示怪物血量，每次都得会营地去更换装备。假如，存在一款辅助性工具，狩猎怪物,在打斗的时候，根据玩家的语音、手势即时的更换装备，这样是不是很酷呢？

- 首先更换装备，其实使用CE去更改内存的方式应该可以实现的吧，这个没有试过
- “剑来”、“刀来”，“一只穿云箭”，“虫棍”......，突然就更换装备（语音控制）
- “换装”（类似《妖精的尾巴》里面的妖精女王的技能），想想都很酷（语音控制）
- 对着摄像头去对比下手势，就换下装备，呼呼。。不敢想了。

既然，理论上能够实现，那为什么不去试一下呢？哪怕失败~~~，我又不恐惧它。

这个部位，可能用于项目起初的控制装备的方式之一

## Tensorflow 关注的几个模块

- 模型 https://tensorflow.google.cn/js/models?hl=zh_cn


## tensorflow.js 的API

- AdadeltaOptimizer: ƒ e(e,n,r)
- AdagradOptimizer: ƒ e(e,n)
- AdamOptimizer: ƒ e(e,n,r,i)
- AdamaxOptimizer: ƒ e(e,n,r,i,o)
- Callback: ƒ e()
- CallbackList: ƒ t(t,e)
- CustomCallback: ƒ e(e)
- DataStorage: ƒ t(t)
- ENV: t {features: {…}, registry: {…}, backendName: "webgl", globalEngine: t}
- Environment: ƒ t(t)
- GraphModel: ƒ t(t,e)
- History: ƒ e()
- InputSpec: ƒ (t)
- KernelBackend: ƒ t()
- LayerVariable: ƒ t(t,e,n,r,i)
- LayersModel: ƒ e(e)
- MomentumOptimizer: ƒ e(e,n,r)
- Optimizer: ƒ e()
- RMSPropOptimizer: ƒ e(e,n,r,i,o)
- RNN: ƒ e(e)
- Rank: {R0: "R0", R1: "R1", R2: "R2", R3: "R3", R4: "R4", …}
- Reduction: {0: "NONE", 1: "MEAN", 2: "SUM", 3: "SUM_BY_NONZERO_WEIGHTS", NONE: 0, MEAN: 1, SUM: 2, SUM_BY_NONZERO_WEIGHTS: 3}
- SGDOptimizer: ƒ e(e)
- Sequential: ƒ e(e)
- SymbolicTensor: ƒ (t,e,n,r,i,o,a)
- Tensor: ƒ t(t,e,n,r,i)
- TensorBuffer: ƒ t(t,e,n)
- Variable: ƒ e(e,n,r)
- abs: ƒ ()
- acos: ƒ ()
- acosh: ƒ ()
- add: ƒ ()
- addN: ƒ ()
- addStrict: ƒ ()
- all: ƒ ()
- any: ƒ ()
- argMax: ƒ ()
- argMin: ƒ ()
- asin: ƒ ()
- asinh: ƒ ()
- atan: ƒ ()
- atan2: ƒ ()
- atanh: ƒ ()
- avgPool: ƒ ()
- basicLSTMCell: ƒ ()
- batchNorm: ƒ ()
- batchNorm2d: ƒ ()
- batchNorm3d: ƒ ()
- batchNorm4d: ƒ ()
- batchNormalization: ƒ ()
- batchNormalization2d: ƒ ()
- batchNormalization3d: ƒ ()
- batchNormalization4d: ƒ ()
- batchToSpaceND: ƒ ()
- browser: {toPixels: ƒ, fromPixels: ƒ}
- buffer: ƒ zo(t,e,n)
- cast: ƒ ()
- ceil: ƒ ()
- clipByValue: ƒ ()
- clone: ƒ ()
- complex: ƒ ()
- concat: ƒ ()
- concat1d: ƒ ()
- concat2d: ƒ ()
- concat3d: ƒ ()
- concat4d: ƒ ()
- constraints: {maxNorm: ƒ, unitNorm: ƒ, nonNeg: ƒ, minMaxNorm: ƒ}
- conv1d: ƒ ()
- conv2d: ƒ ()
- conv2dDerFilter: ƒ ()
- conv2dTranspose: ƒ ()
- conv3d: ƒ ()
- cos: ƒ ()
- cosh: ƒ ()
- cumsum: ƒ ()
- customGrad: ƒ Vt(t)
- data: {array: ƒ, Dataset: ƒ, zip: ƒ, CSVDataset: ƒ, TextLineDataset: ƒ, …}
- deprecationWarn: ƒ Pt(t)
- depthToSpace: ƒ ()
- depthwiseConv2d: ƒ ()
- disableDeprecationWarnings: ƒ zt()
- dispose: ƒ (t)
- disposeVariables: ƒ ()
- div: ƒ ()
- divStrict: ƒ ()
- dot: ƒ ()
- elu: ƒ ()
- enableDebugMode: ƒ Ft()
- enableProdMode: ƒ Lt()
- environment: {EPSILON_FLOAT16: 0.0001, EPSILON_FLOAT32: 1e-7, Environment: ƒ, enableProdMode: ƒ, enableDebugMode: ƒ, …}
- equal: ƒ ()
- equalStrict: ƒ ()
- erf: ƒ ()
- exp: ƒ ()
- expandDims: ƒ ()
- expm1: ƒ ()
- eye: ƒ ()
- fft: ƒ ()
- fill: ƒ Be(t,e,n)
- floor: ƒ ()
- floorDiv: ƒ ()
- fused: {matMul: ƒ}
- gather: ƒ ()
- gatherND: ƒ ()
- getBackend: ƒ ()
- grad: ƒ (t)
- grads: ƒ (t)
- greater: ƒ ()
- greaterEqual: ƒ ()
- greaterEqualStrict: ƒ ()
- greaterStrict: ƒ ()
- ifft: ƒ ()
- imag: ƒ ()
- image: {resizeBilinear: ƒ, resizeNearestNeighbor: ƒ, nonMaxSuppression: ƒ, nonMaxSuppressionAsync: ƒ, cropAndResize: ƒ}
- initializers: {zeros: ƒ, ones: ƒ, constant: ƒ, randomUniform: ƒ, randomNormal: ƒ, …}
- input: ƒ Pd(t)
- io: {browserFiles: ƒ, browserHTTPRequest: ƒ, concatenateArrayBuffers: ƒ, decodeWeights: ƒ, encodeWeights: ƒ, …}
- irfft: ƒ ()
- keep: ƒ (t)
- layers: {inputLayer: ƒ, elu: ƒ, reLU: ƒ, leakyReLU: ƒ, prelu: ƒ, …}
- leakyRelu: ƒ ()
- less: ƒ ()
- lessEqual: ƒ ()
- lessEqualStrict: ƒ ()
- lessStrict: ƒ ()
- linalg: {gramSchmidt: ƒ, qr: ƒ}
- linspace: ƒ Ue(t,e,n)
- loadGraphModel: ƒ (t,e)
- loadLayersModel: ƒ (t,e)
- localResponseNormalization: ƒ ()
- log: ƒ ()
- log1p: ƒ ()
- logSigmoid: ƒ ()
- logSoftmax: ƒ ()
- logSumExp: ƒ ()
- logicalAnd: ƒ ()
- logicalNot: ƒ ()
- logicalOr: ƒ ()
- logicalXor: ƒ ()
- losses: {absoluteDifference: ƒ, computeWeightedLoss: ƒ, cosineDistance: ƒ, hingeLoss: ƒ, …}
- matMul: ƒ ()
- math: {confusionMatrix: ƒ}
- max: ƒ ()
- maxPool: ƒ ()
- maximum: ƒ ()
- maximumStrict: ƒ ()
- mean: ƒ ()
- memory: ƒ ()
- metrics: {binaryAccuracy: ƒ, binaryCrossentropy: ƒ, sparseCategoricalAccuracy: ƒ, categoricalAccuracy: ƒ, categoricalCrossentropy: ƒ, …}
- min: ƒ ()
- minimum: ƒ ()
- minimumStrict: ƒ ()
- mod: ƒ ()
- modStrict: ƒ ()
- model: ƒ (t)
- models: {modelFromJSON: ƒ}
- moments: ƒ ()
- movingAverage: ƒ ()
- mul: ƒ ()
- mulStrict: ƒ ()
- multiRNNCell: ƒ ()
- multinomial: ƒ ()
- neg: ƒ ()
- nextFrame: ƒ Fl()
- norm: ƒ ()
- notEqual: ƒ ()
- notEqualStrict: ƒ ()
- oneHot: ƒ ()
- ones: ƒ ze(t,e)
- onesLike: ƒ ()
- op: ƒ Se(t)
- outerProduct: ƒ ()
- pad: ƒ ()
- pad1d: ƒ ()
- pad2d: ƒ ()
- pad3d: ƒ ()
- pad4d: ƒ ()
- pool: ƒ ()
- pow: ƒ ()
- powStrict: ƒ ()
- prelu: ƒ ()
- print: ƒ Po(t,e)
- prod: ƒ ()
- profile: ƒ (t)
- rand: ƒ ()
- randomNormal: ƒ ()
- randomUniform: ƒ ()
- range: ƒ We(t,e,n,r)
- real: ƒ ()
- reciprocal: ƒ ()
- registerCallbackConstructor: ƒ (t,e)
- regularizers: {l1l2: ƒ, l1: ƒ, l2: ƒ}
- relu: ƒ ()
- reshape: ƒ ()
- reverse: ƒ ()
- reverse1d: ƒ ()
- reverse2d: ƒ ()
- reverse3d: ƒ ()
- reverse4d: ƒ ()
- rfft: ƒ ()
- round: ƒ ()
- rsqrt: ƒ ()
- scalar: ƒ Re(t,e)
- scatterND: ƒ ()
- selu: ƒ ()
- separableConv2d: ƒ ()
- sequential: ƒ (t)
- serialization: {Serializable: ƒ, SerializationMap: ƒ, registerClass: ƒ}
- setBackend: ƒ (t,e)
- setdiff1dAsync: ƒ (t,e)
- sigmoid: ƒ ()
- sign: ƒ ()
- sin: ƒ ()
- sinh: ƒ ()
- slice: ƒ ()
- slice1d: ƒ ()
- slice2d: ƒ ()
- slice3d: ƒ ()
- slice4d: ƒ ()
- softmax: ƒ ()
- softplus: ƒ ()
- spaceToBatchND: ƒ ()
- sparseToDense: ƒ ()
- spectral: {fft: ƒ, ifft: ƒ, rfft: ƒ, irfft: ƒ}
- split: ƒ ()
- sqrt: ƒ ()
- square: ƒ ()
- squaredDifference: ƒ ()
- squaredDifferenceStrict: ƒ ()
- squeeze: ƒ ()
- stack: ƒ ()
- step: ƒ ()
- stridedSlice: ƒ ()
- sub: ƒ ()
- subStrict: ƒ ()
- sum: ƒ ()
- tan: ƒ ()
- tanh: ƒ ()
- tensor: ƒ Te(t,e,n)
- tensor1d: ƒ De(t,e)
- tensor2d: ƒ Oe(t,e,n)
- tensor3d: ƒ _e(t,e,n)
- tensor4d: ƒ Me(t,e,n)
- tensor5d: ƒ Le(t,e,n)
- tensor6d: ƒ Fe(t,e,n)
- test_util: {WEBGL_ENVS: {…}, PACKED_ENVS: {…}, NODE_ENVS: {…}, CHROME_ENVS: {…}, BROWSER_ENVS: {…}, …}
- tidy: ƒ (t,e)
- tile: ƒ ()
- time: ƒ (t)
- topk: ƒ ()
- train: {sgd: ƒ, momentum: ƒ, adadelta: ƒ, adagrad: ƒ, rmsprop: ƒ, …}
- transpose: ƒ ()
- truncatedNormal: ƒ ()
- unsortedSegmentSum: ƒ ()
- unstack: ƒ ()
- util: {shuffle: ƒ, clamp: ƒ, nearestLargerEven: ƒ, sum: ƒ, randUniform: ƒ, …}
- valueAndGrad: ƒ (t)
- valueAndGrads: ƒ (t)
- variable: ƒ (t,n,r,i)
- variableGrads: ƒ Wt(t,e)
- version: {tfjs-core: "1.0.0", tfjs-data: "1.0.0", tfjs-layers: "1.0.0", tfjs-converter: "1.0.0", tfjs: "1.0.0"}
- version_converter: "1.0.0"
- version_core: "1.0.0"
- version_layers: "1.0.0"
- webgl: {gpgpu_util: {…}, webgl_util: {…}, MathBackendWebGL: ƒ, GPGPUContext: ƒ}
- where: ƒ ()
- whereAsync: ƒ (t)
- zeros: ƒ Pe(t,e)
- zerosLike: ƒ ()

## 模块化开发：todo
|优先级|模块化|测试的仓库|描述||
| --- | --- | --- | --- | --- |
|0|训练Tensorflow识别手写图片||||
|1|汇编去捕获内存和更改内存的值||这是直接关系到后续工作的核心部位||
|2|《怪物猎人：世界》——怪物识别码||||
|3|《怪物猎人：世界》——道具识别码||||
|4|《怪物猎人：世界》——技能识别码||||
|5|《怪物猎人：世界》——素材识别码||药草之类||
|6|《怪物猎人：世界》——合成识别码||||
|7|《怪物猎人：世界》——武器识别码||||
|8|《怪物猎人：世界》——装备识别码||||
|9|《怪物猎人：世界》——地图识别码||||
|10|《怪物猎人：世界》——任务识别码||||
|11|《怪物猎人：世界》——攻击姿势识别码||||
|12|《怪物猎人：世界》——防御姿势识别码||||
|13|《怪物猎人：世界》——一般姿势识别码||逃跑||
|14|《怪物猎人：世界》——小型怪物识别码||||
|14|《怪物猎人：世界》——生物分类识别码||古龙、还是牙龙分类||
|15|《怪物猎人：世界》——怪物的技能数识别码||||
|16|《怪物猎人：世界》——护石识别码||||
|17|《怪物猎人：世界》——宝珠识别码||||
|18|《怪物猎人：世界》——计算伤害识别码||||
|19|《怪物猎人：世界》——营地识别码||||
|20|《怪物猎人：世界》——属性识别码||怪物什么属性||
|21|《怪物猎人：世界》——营地识别码||||
|22|《怪物猎人：世界》——联机下，团队输出识别码||||
|23|《怪物猎人：世界》———联机下，投掷器识别码||||
|24|《怪物猎人：世界》———联机下，异常状态识别码||||
|25|《怪物猎人：世界》———联机下，发现识别码||||
|26|《怪物猎人：世界》———地图药草采集区域||||
|26|《怪物猎人：世界》———地图矿石采集区域||||
|26|《怪物猎人：世界》———地图陷阱采集区域||||
|26|《怪物猎人：世界》———地图水域采集区域||哪个地图哪个水域出现哪种鱼||
|27|《怪物猎人：世界》———投掷器识别码||||
|28|《怪物猎人：世界》———血量预估识别码||||
|30|程序控制键盘||pyautogui：[Python直接控制鼠标键盘](https://www.cnblogs.com/fanghao/p/8453207.html)||
|31|语音识别||通过语音得到预设的值，语音->识别->归类为某个指令||
|32|怪物图片分类识别||机器学习怪物的模型，||
|33|从视频上学习怪物的姿态||||
|34|计算技能分析||||
|35|如何去识别人体的身姿||||
|36|如何去识别手势||||
|37|摄像头识别||||
|38|opencv相关||||
|39|通过AI识别怪物种类|Kuril yak|||
|40|基于Tensorflow.js Node,.js 音高预测||||
|41|MINIST数字识别器|https://github.com/tensorflow/tfjs-examples/tree/master/mnist|||
|42|加法RNN，训练模型从文本样本中学习加法|https://github.com/tensorflow/tfjs-examples/tree/master/addition-rnn|||
|43|鸢尾花分类,使用表格式数据对花朵进行分类|https://github.com/tensorflow/tfjs-examples/tree/master/iris|||
|44|TensorFlow.js 训练图片分类|||使用mobilenet API ，效果真渣|
|45|TensorFlow.js 训练手写|||||||
|46|python-opencv 识别图片|https://www.kancloud.cn/aollo/aolloopencv/259610|||
||||||


## 怪物图片/视频素材

### 大型怪物

|种属|名字|级别|属性|有效属性|
| --- | --- | --- | --- | --- |
|牙龙种|大凶豺龙|初阶|
|鸟龙种|骚鸟|初阶/进阶||
|鸟龙种|毒妖鸟|初阶/进阶||
|牙龙种|飞雷龙|初阶/进阶||
||凶鄂龙|初阶||
||雌火龙|初阶/进阶||
||火龙|初阶/进阶||
||土砂龙|初阶||
|鱼龙种|泥鱼龙|初阶/进阶||
|飞龙种|角龙|初阶/进阶||
|飞龙种|黑角龙|进阶||
|飞龙种|苍火龙|进阶||
|飞龙种|樱火龙|进阶||
||骨锤龙|低阶||
||爆锤龙|进阶||
||爆麟龙|||
||大凶颚龙|||
||惨爪龙|||
||眩鸟|||
||风飘龙|||
||浮空龙|||
|古龙|冰麒麟|||
|古龙|麒麟|||
||岩贼龙|||
|鱼龙种|熔岩龙|||
||恐暴龙|||
||灭尽龙|||
||钢龙|||
||炎王龙|||
||尸套龙|||
||冥灯龙|||

### 小型怪物
|分类|名字||
| --- | --- | --- |
|牙龙|大凶豺龙||
||||
||||
||||
||||

## 必备条件

|知识技能|需要掌握程度|目前掌握程度|是否具备条件|描述|
| --- | --- | --- | --- | --- | 
|Opencv|20%|0%|×|有书，没看过...|
|ffmpeg|30%|0%|×|一个转换视频的工具|
|TensorFlow|20-30%|0%|×|贼烦算法！shit| 
|TensorFlow.js|50%|0%|×|js版本的tensorflow|
|Handtrack.js|50%|0%|×|手势追踪的一个库|
|Python|10%-30%|20%|√|能够使用实现一些实践|
|Node.js|30%|60%|√|很熟悉|
|浏览器前端知识|40%|80%|√|差不多精通级|
|汇编|30%|5%|×|最难了|
|计算机原理和内存|10%|0%|×|了解其机制|
|图像识别|30%|0%|×|需要了解原理|
|语音识别|30%|0%|×|需要了解原理|
||||||


## 软件设计

### 更改内存的值去实现变更道具
- 核心

### 怪物分类识别（机器学习）

根据怪物的图片，先简单识别名称

### 怪物姿态识别（机器学习）

- 通过识别怪物的姿态，以及攻击方式 https://experiments.withgoogle.com/move-mirror，这个demo 展示了可以识别人的姿态
    - 类别（鸟类、飞龙、牙龙类、古龙类）
    - 名字（炎王龙、灭尽龙，唯一值）
    - 姿态分析：
        - 第一步：先识别当前什么姿态，这个姿态用什么技能
        - 第二步：预测这个姿态下的下一个技能的倒计时，以及范围
- 姿态识别技能

- 姿态预测下一个技能，以及下下一个技能

- 技能补位检测：预测不受攻击的站位和区域，
- 预测可能出现的漏洞：（这个有点扯，先放着吧）


### 手势识别
- 可以通过已有的类库 handtrack.js实现控制
- 通过手势变更装备

### 语音识别
- 通过识别语音，目前暂定百度语音的API，去更换装备或者道具
- 需要准确识别道具的名称
- 合成指令：通过A+B等合成
- “剑来”、“刀来”，“一只穿云箭”，“虫棍”......，突然就更换装备（语音控制）
- “换装”（类似《妖精的尾巴》里面的妖精女王的技能），想想都很酷（语音控制）

### 指令预设：

- 合成 xx
- 换装 xx
- 刀来
- 太刀
- 弓箭
- 操虫棍
- 使用药水
- 使用大药水
- 使用麻痹陷阱
- 使用麻醉捕获球
- 技能指令：剑气斩巴拉巴拉之类的，


### 高级模式下（终极版概念）
- 1：1 绘制玩家的姿态和游戏画面的怪物身姿比例
- 通过识别玩家的姿态和手拿一把太刀的劈砍攻击姿态和语音辅助，去隔空鼠标来和PC上游戏画面的人物交互
    - 在一个大的空旷空间内，摄像头识别真实玩家的身姿、位置来具象化投影怪物（和AR游戏一样吧，这里设想）
    - 语音为控制指令
- 这一个步骤比较麻烦的一点是：如何通过外部的API去控制鼠标和键盘的参数，因为键盘和鼠标属于外部的IO设备，这个如何控制？现在还是个空白，没什么概念。
- 需要非常专业的设备+场地才能支持到这一步，很麻烦

## 难点：
- 如何隔空或者获取鼠标参数？
- 如何定位全部的内存参数和变更？


