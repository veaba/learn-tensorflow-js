import tensorflow as tf 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
message= tf.constant("welcome to the exciting world of deep neural networks")
with tf.compat.v1.Session() as sess: 
    # print(sess.run(message))# b'welcome to the exciting world of deep neural networks'
    print(sess.run(message).decode()) 