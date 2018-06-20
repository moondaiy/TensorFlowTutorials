import tensorflow as tf
import numpy as np



def my_func(x,z):

    print(type(x))
    print(type(z))
    print(z)
    print(x)

    return np.sinh(x), z

inp = tf.placeholder(tf.float32,shape=[2,2])

testVar = tf.constant(12.0)

#tf.py_func可以接收tensor类型的参数,在提供的python 模块内部会被变成 numpy 类型, 输出的类型要指明,在这个函数调用完毕后,会讲 模块内部的输出转换成 tensorflow的类型
y1,y2 = tf.py_func(my_func, [inp, testVar], [tf.float32, tf.float32])


init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    sess.run([y1,y2], feed_dict={inp:[[1,2],[3,4]]})