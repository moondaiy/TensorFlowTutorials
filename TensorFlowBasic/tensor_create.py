import tensorflow as tf
import numpy as np



# source_tensor = np.ones((4,3),dtype=np.int64)

source_tensor = tf.placeholder(tf.float32, shape=[None,None,3])


target = tf.ones(tf.shape(source_tensor),dtype=source_tensor.dtype)

init = tf.global_variables_initializer()

mask = target > source_tensor

# result = tf.where(mask, -1, 1)


with tf.Session() as sess:

    sess.run(init)

    # print(sess.run(target,feed_dict={source_tensor : np.zeros((2,3,3),dtype=np.float32)}))

    print(sess.run(mask,feed_dict={source_tensor:np.zeros((2,3,3),dtype=np.float32)}))

    # print(sess.run(result, feed_dict={source_tensor: np.zeros((2, 3, 3), dtype=np.float32)}))