# import tensorflow as tf
#
# a = tf.get_variable("a", dtype=tf.int32, shape=[], initializer=tf.zeros_initializer())
# b = tf.get_variable("b", dtype=tf.float32,shape=[], initializer=tf.ones_initializer())
#
# # f = tf.constant(6)
#
# # Definition of condition and body
# def cond(a, b, f):
#
#     # a = tf.Print(a,[a, tf.shape(a)], message="cond a : ")
#
#     return tf.less(a,3)
#
#
# def body(a, b, f):
#     # do some stuff with a, b
#
#     # a = 1
#
#     a = tf.Print(a, [a], message="body a : ")
#
#     add = tf.add(a, 1)
#
#     add = tf.Print(add, [add], message="body add : ")
#
#     with tf.control_dependencies([add]):
#
#         f = tf.cond(tf.less(add,2), lambda :f.write(add, 3.2), lambda : f.write(add,4.1))
#         b = f.read(a)
#         b = tf.Print(b, [b], message="body b : ")
#
#     return add, b, f
#
# # Loop, 返回的tensor while 循环后的 a，b，f
# # f = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True,clear_after_read = False)
# f = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
#
# a, b, f = tf.while_loop(cond, body, [a, b, f])
#
# result = f.stack()
#
# with tf.Session() as sess:
#
#     tf.global_variables_initializer().run()
#
#     a, b, result = sess.run([a, b, result])
#
#     print(result)

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops
import numpy as np

def body(time_var, attention_tracker):

    a = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100)

    b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)

    c = a + b

    attention_tracker = attention_tracker.write(time_var, c)

    # attention_tracker = tf.Print(attention_tracker, [attention_tracker], message="body : ")

    return time_var + 1 ,  attention_tracker

def condition(time_var, attention_tracker):

    time_var = tf.Print(time_var, [time_var], message="condition time_var : ")

    return time_var < 10

x = tf.Variable(tf.constant(0, shape=[2, 2]))

#如果 infer_shape=False 则需要指定element shape大小
attention_tracker = tensor_array_ops.TensorArray(tf.int32, size=1, dynamic_size=True, infer_shape=False, element_shape=[2, 2])

time = tf.Variable(1)

time_new , attention_tracker_result = tf.while_loop(condition, body, [time, attention_tracker])

result = attention_tracker_result.stack()

finally1 = tf.add(result, 1)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    finally2 = sess.run([finally1])

    print(finally2)