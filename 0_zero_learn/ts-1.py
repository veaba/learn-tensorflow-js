import tensorflow as tf 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def demo():
    a= tf.constant([1,2,3,4])
    b= tf.constant([5,6,7,8])

    print(a) # Tensor("Const:0", shape=(4,), dtype=int32)
    print(b) # Tensor("Const_1:0", shape=(4,), dtype=int32)

    add =tf.add(a,b)

    with tf.compat.v1.Session() as sess:
        print(sess.run(add)) # [6 8 10 12]


"""
with tf.compat.v1.Session() as sess:
    print(sess.run(add)) # [6 8 10 12]

等于下面：

sess =tf.compat.v1.Session()
print(sess.run(add))
sess.close()

"""

def all_zeros():
    zero=tf.zeros([9,10],"int32")
    with tf.compat.v1.Session() as sess1:
        print(sess1.run(zero))
"""
[[0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]]
"""

def all_one():
    one=tf.ones([9,10],tf.int32)
    with tf.compat.v1.Session() as sess:
        print(sess.run(one))


# tensorflow 变量


def var_rand_1():
    rand_t=tf.random.uniform([50,50],0,10,seed=0)
    a1=tf.Variable(rand_t)
    # a2=tf.Variable(rand_t)
    with tf.compat.v1.Session() as sess:
        print(sess.run(a1.initializer))


def placehoder_run():
    x =tf.compat.v1.placeholder("float")
    y=2*x
    data=tf.random.uniform([4,6],10)#10 默认0 0-1，乘以10倍
    with tf.compat.v1.Session() as sess:
         x_data=sess.run(data)
         print(sess.run(y,feed_dict={x:x_data}))

placehoder_run()