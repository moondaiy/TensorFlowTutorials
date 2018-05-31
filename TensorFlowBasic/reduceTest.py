import tensorflow as tf
import numpy as np


a1 = tf.Variable(initial_value=np.array([[1,2],[3,4]]),dtype=tf.float32)
b1 = tf.Variable(initial_value=np.array([[1,2],[3,4]]),dtype=tf.float32)

reduce_sum = tf.reduce_sum(a1)
reduce_mean = tf.reduce_mean(b1)
reduce_sum_mean = tf.reduce_mean(tf.reduce_sum(a1))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(reduce_sum))
    print(sess.run(reduce_mean))
    print(sess.run(reduce_sum_mean))