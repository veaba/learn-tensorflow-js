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


### 损失函数和优化器

## 训练模型

## 评估模型

## 进一步阅读