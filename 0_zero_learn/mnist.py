# 循环手写图片

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
mnist= input_data.read_data_sets("MNIST_data/",one_hot=True)

"""
mnist=
 Datasets(
     train=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x0000000002933630>, 
     validation=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x000000001ADB1DA0>, 
     test=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x000000001ADB1DD8>
)
"""


# 定义一个占位符 x，干啥的？
x = tf.compat.v1.placeholder(tf.float32,[None,784]) # 张量tensor 形状是 [None,784],None第一个纬度任意

"""
x =Tensor("Placeholder:0", shape=(?, 784), dtype=float32)
"""

# 定义变量W，b，是可以被修改的丈量，用来存放机器学习模型参数

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
"""
W =
<tf.Variable 'Variable:0' shape=(784, 10) dtype=float32_ref>
b =
<tf.Variable 'Variable_1:0' shape=(10,) dtype=float32_ref>
"""

# 实现模型，y是预测分布
y =tf.nn.softmax(tf.matmul(x,W)+b)
"""
y=
Tensor("Softmax:0", shape=(?, 10), dtype=float32)
"""

# 训练模型，y_ 是实际分布
y_=tf.compat.v1.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_ * tf.math.log(y)) # 交叉嫡？啥玩意？cost function

"""
y_=
Tensorcross_entropyz("Placeholder_1:0", shape=(?, 10), dtype=float32)

cross_entropy =
Tensor("Neg:0", shape=(), dtype=float32)
"""



#使用梯度下降来降低cost，学习速率为0.01
train_step=tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化已经创建的变量
init=tf.compat.v1.global_variables_initializer()

"""
train_step=
name: "GradientDescent"
op: "NoOp"
input: "^GradientDescent/update_Variable/ApplyGradientDescent"
input: "^GradientDescent/update_Variable_1/ApplyGradientDescent"

init =
name: "init"
op: "NoOp"
input: "^Variable/Assign"
input: "^Variable_1/Assign"


"""


# 在一个session 中启动模型，并初始化变量

sess=tf.compat.v1.Session()
sess.run(init)


"""
sess=
<tensorflow.python.client.session.Session object at 0x000000001F5F6F98>

"""


for i in range(1,10000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

# 验证正确率
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

"""
batch_xs:
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
batch_ys:
[[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 ...
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]
"""

"""
1000 0.916
100000 0.098
10000 0.9193


"""