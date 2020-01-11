# IMDB 影评分类

## 疑问（译者）
## 步骤梳理（译者）
- 拉取50000影评，二分的模型，不是好就是坏，所以，好坏一半，为此训练集合测试集是一样的数据

## 相关资源（译者）
- [原文](https://tensorflow.google.cn/tutorials/keras/text_classification_with_hub)
- tensorflow_hub 一个迁移学习的库和平台
> pip install tensorflow_hub
- tensorflow_datasets
> pip install tensorflow_datasets
## 简介

本笔记本根据影评的内容将影评分为正面影评和负面影评。这是一个二类或二类分类的例子，是一类重要且应用广泛的机器学习问题。

本教程演示了使用TensorFlow Hub和Keras进行迁移学习的基本应用。

我们将使用来源于网络电影数据库（Internet Movie Database）的 IMDB 数据集（IMDB dataset），其包含 50,000 条影评文本。从该数据集切割出的 25,000 条评论用作训练，另外 25,000 条用作测试。训练集与测试集是平衡的（balanced），意味着它们包含相等数量的积极和消极评论。

此笔记本（notebook）使用了 tf.keras，它是一个 Tensorflow 中用于构建和训练模型的高级API，此外还使用了 TensorFlow Hub，一个用于迁移学习的库和平台。有关使用 tf.keras 进行文本分类的更高级教程，请参阅 MLCC文本分类指南（MLCC Text Classification Guide）。

```python
from __future__ import  absolute_import,division,print_function,unicode_literals
import numpy as np 

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# 检查版本
print('Version:',tf.__version__)
print('Eager mode:',tf.executing_eagerly())
print('Hub Version:',hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

```

## 下载IMDB数据集
IMDB数据集可以在 Tensorflow 数据集处获取。以下代码将 IMDB 数据集下载至您的机器（或 colab 运行时环境）中：

```python
# 将训练集按照6:4比例划分，得到15000训练样本，10000验证样本，以及25000测试样本
trans_validation_split=tfds.Split3.TRAIN.subsplit([6,4])
(train_data,validation_data),test_data=tfds.load(
    name="imdb_reviews",
    split=(train_validation_split,tfds.Split.TEST),
    as_supervised=True
)

```

等待下载完成...

## 探索数据
花点时间了解下数据格式吧，每个样本都是一个表示电影评论和相应标签的句子，句子不以任何方式进行预处理。标签是一个值为0或1的整数，其中0表示消极评论，1代表积极评论

让我们来打印下前十个样本

```python
train_examples_batch,train_labels_batch=next(iter(train_data.batch(10)))

# 打印下
print(train_examples_batch)
```
结果
```text
<tf.Tensor: id=219, shape=(10,), dtype=string, numpy=
array([b"As a lifelong fan of Dickens, I have invariably been disappointed by adaptations of his novels.<br /><br />Although his works presented an extremely accurate re-telling of human life at every level in Victorian Britain, throughout them all was a pervasive thread of humour that could be both playful or sarcastic as the narrative dictated. In a way, he was a literary caricaturist and cartoonist. He could be serious and hilarious in the same sentence. He pricked pride, lampooned arrogance, celebrated modesty, and empathised with loneliness and poverty. It may be a clich\xc3\xa9, but he was a people's writer.<br /><br />And it is the comedy that is so often missing from his interpretations. At the time of writing, Oliver Twist is being dramatised in serial form on BBC television. All of the misery and cruelty is their, but non of the humour, irony, and savage lampoonery. The result is just a dark, dismal experience: the story penned by a journalist rather than a novelist. It's not really Dickens at all.<br /><br />'Oliver!', on the other hand, is much closer to the mark. The mockery of officialdom is perfectly interpreted, from the blustering beadle to the drunken magistrate. The classic stand-off between the beadle and Mr Brownlow, in which the law is described as 'a ass, a idiot' couldn't have been better done. Harry Secombe is an ideal choice.<br /><br />But the blinding cruelty is also there, the callous indifference of the state, the cold, hunger, poverty and loneliness are all presented just as surely as The Master would have wished.<br /><br />And then there is crime. Ron Moody is a treasure as the sleazy Jewish fence, whilst Oliver Reid has Bill Sykes to perfection.<br /><br />Perhaps not surprisingly, Lionel Bart - himself a Jew from London's east-end - takes a liberty with Fagin by re-interpreting him as a much more benign fellow than was Dicken's original. In the novel, he was utterly ruthless, sending some of his own boys to the gallows in order to protect himself (though he was also caught and hanged). Whereas in the movie, he is presented as something of a wayward father-figure, a sort of charitable thief rather than a corrupter of children, the latter being a long-standing anti-semitic sentiment. Otherwise, very few liberties are taken with Dickens's original. All of the most memorable elements are included. Just enough menace and violence is retained to ensure narrative fidelity whilst at the same time allowing for children' sensibilities. Nancy is still beaten to death, Bullseye narrowly escapes drowning, and Bill Sykes gets a faithfully graphic come-uppance.<br /><br />Every song is excellent, though they do incline towards schmaltz. Mark Lester mimes his wonderfully. Both his and my favourite scene is the one in which the world comes alive to 'who will buy'. It's schmaltzy, but it's Dickens through and through.<br /><br />I could go on. I could commend the wonderful set-pieces, the contrast of the rich and poor. There is top-quality acting from more British regulars than you could shake a stick at.<br /><br />I ought to give it 10 points, but I'm feeling more like Scrooge today. Soak it up with your Christmas dinner. No original has been better realised.",
       b"Oh yeah! Jenna Jameson did it again! Yeah Baby! This movie rocks. It was one of the 1st movies i saw of her. And i have to say i feel in love with her, she was great in this move.<br /><br />Her performance was outstanding and what i liked the most was the scenery and the wardrobe it was amazing you can tell that they put a lot into the movie the girls cloth were amazing.<br /><br />I hope this comment helps and u can buy the movie, the storyline is awesome is very unique and i'm sure u are going to like it. Jenna amazed us once more and no wonder the movie won so many awards. Her make-up and wardrobe is very very sexy and the girls on girls scene is amazing. specially the one where she looks like an angel. It's a must see and i hope u share my interests",
       b"I saw this film on True Movies (which automatically made me sceptical) but actually - it was good. Why? Not because of the amazing plot twists or breathtaking dialogue (of which there is little) but because actually, despite what people say I thought the film was accurate in it's depiction of teenagers dealing with pregnancy.<br /><br />It's NOT Dawson's Creek, they're not graceful, cool witty characters who breeze through sexuality with effortless knowledge. They're kids and they act like kids would. <br /><br />They're blunt, awkward and annoyingly confused about everything. Yes, this could be by accident and they could just be bad actors but I don't think so. Dermot Mulroney gives (when not trying to be cool) a very believable performance and I loved him for it. Patricia Arquette IS whiny and annoying, but she was pregnant and a teenagers? The combination of the two isn't exactly lavender on your pillow. The plot was VERY predictable and but so what? I believed them, his stress and inability to cope - her brave, yet slightly misguided attempts to bring them closer together. I think the characters, acted by anyone else, WOULD indeed have been annoying and unbelievable but they weren't. It reflects the surreality of the situation they're in, that he's sitting in class and she walks on campus with the baby. I felt angry at her for that, I felt angry at him for being such a child and for blaming her. I felt it all.<br /><br />In the end, I loved it and would recommend it.<br /><br />Watch out for the scene where Dermot Mulroney runs from the disastrous counselling session - career performance.",
       b'This was a wonderfully clever and entertaining movie that I shall never tire of watching many, many times. The casting was magnificent in matching up the young with the older characters. There are those of us out here who really do appreciate good actors and an intelligent story format. As for Judi Dench, she is beautiful and a gift to any kind of production in which she stars. I always make a point to see Judi Dench in all her performances. She is a superb actress and a pleasure to watch as each transformation of her character comes to life. I can only be grateful when I see such an outstanding picture for most of the motion pictures made more recently lack good characters, good scripts and good acting. The movie public needs heroes, not deviant manikins, who lack ingenuity and talent. How wonderful to see old favorites like Leslie Caron, Olympia Dukakis and Cleo Laine. I would like to see this movie win the awards it deserves. Thank you again for a tremendous night of entertainment. I congratulate the writer, director, producer, and all those who did such a fine job.',
       b'I have no idea what the other reviewer is talking about- this was a wonderful movie, and created a sense of the era that feels like time travel. The characters are truly young, Mary is a strong match for Byron, Claire is juvenile and a tad annoying, Polidori is a convincing beaten-down sycophant... all are beautiful, curious, and decadent... not the frightening wrecks they are in Gothic.<br /><br />Gothic works as an independent piece of shock film, and I loved it for different reasons, but this works like a Merchant and Ivory film, and was from my readings the best capture of what the summer must have felt like. Romantic, yes, but completely rekindles my interest in the lives of Shelley and Byron every time I think about the film. One of my all-time favorites.',
       b"This was soul-provoking! I am an Iranian, and living in th 21st century, I didn't know that such big tribes have been living in such conditions at the time of my grandfather!<br /><br />You see that today, or even in 1925, on one side of the world a lady or a baby could have everything served for him or her clean and on-demand, but here 80 years ago, people ventured their life to go to somewhere with more grass. It's really interesting that these Persians bear those difficulties to find pasture for their sheep, but they lose many the sheep on their way.<br /><br />I praise the Americans who accompanied this tribe, they were as tough as Bakhtiari people.",
       b'Just because someone is under the age of 10 does not mean they are stupid. If your child likes this film you\'d better have him/her tested. I am continually amazed at how so many people can be involved in something that turns out so bad. This "film" is a showcase for digital wizardry AND NOTHING ELSE. The writing is horrid. I can\'t remember when I\'ve heard such bad dialogue. The songs are beyond wretched. The acting is sub-par but then the actors were not given much. Who decided to employ Joey Fatone? He cannot sing and he is ugly as sin.<br /><br />The worst thing is the obviousness of it all. It is as if the writers went out of their way to make it all as stupid as possible. Great children\'s movies are wicked, smart and full of wit - films like Shrek and Toy Story in recent years, Willie Wonka and The Witches to mention two of the past. But in the continual dumbing-down of American more are flocking to dreck like Finding Nemo (yes, that\'s right), the recent Charlie & The Chocolate Factory and eye-crossing trash like Red Riding Hood.',
       b"I absolutely LOVED this movie when I was a kid. I cried every time I watched it. It wasn't weird to me. I totally identified with the characters. I would love to see it again (and hope I wont be disappointed!). Pufnstuf rocks!!!! I was really drawn in to the fantasy world. And to me the movie was loooong. I wonder if I ever saw the series and have confused them? The acting I thought was strong. I loved Jack Wilde. He was so dreamy to an 10 year old (when I first saw the movie, not in 1970. I can still remember the characters vividly. The flute was totally believable and I can still 'feel' the evil woods. Witchy poo was scary - I wouldn't want to cross her path.",
       b'A very close and sharp discription of the bubbling and dynamic emotional world of specialy one 18year old guy, that makes his first experiences in his gay love to an other boy, during an vacation with a part of his family.<br /><br />I liked this film because of his extremly clear and surrogated storytelling , with all this "Sound-close-ups" and quiet moments wich had been full of intensive moods.<br /><br />',
       b"This is the most depressing film I have ever seen. I first saw it as a child and even thinking about it now really upsets me. I know it was set in a time when life was hard and I know these people were poor and the crops were vital. Yes, I get all that. What I find hard to take is I can't remember one single light moment in the entire film. Maybe it was true to life, I don't know. I'm quite sure the acting was top notch and the direction and quality of filming etc etc was wonderful and I know that every film can't have a happy ending but as a family film it is dire in my opinion.<br /><br />I wouldn't recommend it to anyone who wants to be entertained by a film. I can't stress enough how this film affected me as a child. I was talking about it recently and all the sad memories came flooding back. I think it would have all but the heartless reaching for the Prozac."],
      dtype=object)>
```

再打印下训练集前十个标签

```python
print(train_labels_batch)
```
结果：
```text
<tf.Tensor: id=220, shape=(10,), dtype=int64, numpy=array([1, 1, 1, 1, 1, 1, 0, 1, 1, 0])>
```

## 构建模型
神经网络是又堆叠的层（Layers）来构建的，这需要从三个主要方面来进行体系结构的决策：

- 如何表示文本？
- 模型里有多少层？
- 每个层里有多少隐层单元（Hidden units）？

本示例中，输入数据由句子组成，预测的标签为0或1

表示文档的一种方式是将句子转换为嵌入向量（embeddings vectors），我们可以适用一个预先训练好的文本嵌入（text embeddings）作为首层，这将有以下三个优点：

- 我们不必担心文本预处理
- 我们可以从迁移学习中受益
- 嵌入具有固定长度，更易于处理

针对此示例我们将使用[TensorFlow Hub](https://tensorflow.google.cn/hub) 中名为 [google/tf2-preview/gnews-swivel-20dim/1](https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1) 的一种预训练文本嵌入模型。

为了达到本教程的目的还有其他三种预训练模型可供测试：

- google/tf2-preview/gnews-swivel-20dim-with-oov/1 —— 类似 google/tf2-preview/gnews-swivel-20dim/1，但2.5% 的词汇转为未登录词桶（OOV Buckets）。如果任务的词汇和模型的词汇没有完全重叠，这将会有所帮助。
- google/tf2-preview/nnlm-en-dim50/1 —— 一个拥有约1M词汇量且维度为50的更大的模型。
- google/tf2-preview/nnlm-en-dim128/1 —— 一个拥有约1M词汇量且维度为128的更大的模型

让我们首先创建一个使用 TensorFlow Hub 模型嵌入（embed）语句的Keras层，并在几个输入样本中进行尝试，请注意无论输入文本的长度如何，嵌入输出的形状都是: `(number_examples,embedding_dimension)`

```python
embedding="https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer=hub.KerasLayer(embedding,input_shape=[],dtype=tf.string,trainable=True)
hub_layer(train_examples_batch[:3])
```
结果如下：
```text
<tf.Tensor: id=402, shape=(3, 20), dtype=float32, numpy=
array([[ 3.9819887 , -4.4838037 ,  5.177359  , -2.3643482 , -3.2938678 ,
        -3.5364532 , -2.4786978 ,  2.5525482 ,  6.688532  , -2.3076782 ,
        -1.9807833 ,  1.1315885 , -3.0339816 , -0.7604128 , -5.743445  ,
         3.4242578 ,  4.790099  , -4.03061   , -5.992149  , -1.7297493 ],
       [ 3.4232912 , -4.230874  ,  4.1488533 , -0.29553518, -6.802391  ,
        -2.5163853 , -4.4002395 ,  1.905792  ,  4.7512794 , -0.40538004,
        -4.3401685 ,  1.0361497 ,  0.9744097 ,  0.71507156, -6.2657013 ,
         0.16533905,  4.560262  , -1.3106939 , -3.1121316 , -2.1338716 ],
       [ 3.8508697 , -5.003031  ,  4.8700504 , -0.04324996, -5.893603  ,
        -5.2983093 , -4.004676  ,  4.1236343 ,  6.267754  ,  0.11632943,
        -3.5934832 ,  0.8023905 ,  0.56146765,  0.9192484 , -7.3066816 ,
         2.8202746 ,  6.2000837 , -3.5709393 , -4.564525  , -2.305622  ]],
      dtype=float32)>

```
现在让我们构建完整的模型：

```python
model=tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

model.summary()
```

结果
```text
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
keras_layer (KerasLayer)     (None, 20)                400020    
_________________________________________________________________
dense (Dense)                (None, 16)                336       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17        
=================================================================
Total params: 400,373
Trainable params: 400,373
Non-trainable params: 0
_________________________________________________________________
```

层按顺序堆叠以构建分类器：

1. 第一层是TensorFlow Hub层。这一层使用一个预训练的保存好的模型来将句子映射为嵌入向量。我们使用的预训练文本嵌入模型(google/tf2-preview/gnews-swivel-20dim/1)将句子切割为符号，嵌入每个符号然后进行合并，最终得到的维度是：(num_examples,embedding_dimension)
2. 该定长输出向量通过一个有16个隐层单元的全连接层（Dense）进行管道传输。
3. 最后一层与单个输出结点紧密相连，使用Sigmoid激活含税，其函数值介于0-1之间的浮点数，表示概率或置信水平。

让我们编译模型。

### 损失函数和优化器

一个模型需要损失函数和优化器来训练，由于这是一个二分类问题且模型输出概率值（一个使用sigmoid激活函数单一单元层），我们将使用`binary_crossentropy`损失函数。

这不是损失函数的唯一选择，例如，你可以选择`mean_squared_error`，但是，一般来说，`binary_crossentropy`更适合处理概率问题。它能够度量概率分布之间的距离，或者在我们的示例中，指的度量ground-truth分布与预测值之间的距离

现在，配置模型来使用优化器和损失函数。

```python
model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```
## 训练模型

以512个样本的mini-batch大小迭代20个epoch来训练模型，这是指`x_train`和`y_train`张量中所有样本的20次迭代。在训练过程中，监测来自验证集10000个样本上的损失值(loss)和准确率(accuracy)：

```python
history=model.fit(train_data.shuffle(10000).batch(512),
    epochs=20,
    validation_data=validation_data.batch(512),
    verbose=1
)
```
结果：
```text
Epoch 1/20
30/30 [==============================] - 5s 153ms/step - loss: 0.9062 - accuracy: 0.4985 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 2/20
30/30 [==============================] - 4s 117ms/step - loss: 0.7007 - accuracy: 0.5625 - val_loss: 0.6692 - val_accuracy: 0.6029
Epoch 3/20
30/30 [==============================] - 4s 117ms/step - loss: 0.6486 - accuracy: 0.6379 - val_loss: 0.6304 - val_accuracy: 0.6543
Epoch 4/20
30/30 [==============================] - 4s 117ms/step - loss: 0.6113 - accuracy: 0.6866 - val_loss: 0.5943 - val_accuracy: 0.6966
Epoch 5/20
30/30 [==============================] - 3s 114ms/step - loss: 0.5764 - accuracy: 0.7176 - val_loss: 0.5650 - val_accuracy: 0.7201
Epoch 6/20
30/30 [==============================] - 3s 109ms/step - loss: 0.5435 - accuracy: 0.7447 - val_loss: 0.5373 - val_accuracy: 0.7424
Epoch 7/20
30/30 [==============================] - 3s 110ms/step - loss: 0.5132 - accuracy: 0.7723 - val_loss: 0.5080 - val_accuracy: 0.7667
Epoch 8/20
30/30 [==============================] - 3s 110ms/step - loss: 0.4784 - accuracy: 0.7943 - val_loss: 0.4790 - val_accuracy: 0.7833
Epoch 9/20
30/30 [==============================] - 3s 110ms/step - loss: 0.4440 - accuracy: 0.8172 - val_loss: 0.4481 - val_accuracy: 0.8054
Epoch 10/20
30/30 [==============================] - 3s 112ms/step - loss: 0.4122 - accuracy: 0.8362 - val_loss: 0.4204 - val_accuracy: 0.8196
Epoch 11/20
30/30 [==============================] - 3s 110ms/step - loss: 0.3757 - accuracy: 0.8534 - val_loss: 0.3978 - val_accuracy: 0.8290
Epoch 12/20
30/30 [==============================] - 3s 111ms/step - loss: 0.3449 - accuracy: 0.8685 - val_loss: 0.3736 - val_accuracy: 0.8413
Epoch 13/20
30/30 [==============================] - 3s 109ms/step - loss: 0.3188 - accuracy: 0.8798 - val_loss: 0.3570 - val_accuracy: 0.8465
Epoch 14/20
30/30 [==============================] - 3s 110ms/step - loss: 0.2934 - accuracy: 0.8893 - val_loss: 0.3405 - val_accuracy: 0.8549
Epoch 15/20
30/30 [==============================] - 3s 109ms/step - loss: 0.2726 - accuracy: 0.9003 - val_loss: 0.3283 - val_accuracy: 0.8611
Epoch 16/20
30/30 [==============================] - 3s 111ms/step - loss: 0.2530 - accuracy: 0.9079 - val_loss: 0.3173 - val_accuracy: 0.8648
Epoch 17/20
30/30 [==============================] - 3s 113ms/step - loss: 0.2354 - accuracy: 0.9143 - val_loss: 0.3096 - val_accuracy: 0.8679
Epoch 18/20
30/30 [==============================] - 3s 112ms/step - loss: 0.2209 - accuracy: 0.9229 - val_loss: 0.3038 - val_accuracy: 0.8700
Epoch 19/20
30/30 [==============================] - 3s 112ms/step - loss: 0.2037 - accuracy: 0.9287 - val_loss: 0.2990 - val_accuracy: 0.8736
Epoch 20/20
30/30 [==============================] - 3s 109ms/step - loss: 0.1899 - accuracy: 0.9349 - val_loss: 0.2960 - val_accuracy: 0.8751
```


## 评估模型

我们来看下模型的表现如何，将返回两个值，损失值：一个表示误差的数字，值越低越好；准确率。

```python
results=model.evalute(test_data.patch(512),varbose=2)
for name,value in zip(model.metrics_names,results):
    print("%s: %.3f " % (name,value))
```

结果如下：

```text
49/49 - 2s - loss: 0.3163 - accuracy: 0.8651
loss: 0.316
accuracy: 0.865
```
这种十分朴素的方法得到约87%的准确率，若采用更高的方法，模型的准确率应当接近95%。

## 进一步阅读
有关使用字符串输入的更一般的方法，以及对训练期间的准确率和损失值的更详细的分析，请参阅[此处](https://tensorflow.google.cn/tutorials/keras/basic_text_classification)
