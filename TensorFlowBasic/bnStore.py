# coding:utf-8
'''
图像旋转
'''

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
import numpy as np

resnet_variables = 'resnet_variables'

update_ops_collection = "bn_ops"
bn_decay = 0.5

def _get_variable(name,shape,initializer,weight_decay=0.0,dtype='float',trainable=True):
    "a little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    collections = [tf.GraphKeys.VARIABLES, resnet_variables]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                            trainable=trainable)

trainflag =  tf.placeholder(dtype=tf.bool,shape=[])

# input = tf.variable([[1,1,1,1],[2,2,2,2]],dtype=tf.float32)
# input = tf.get_variable("test", initializer=tf.constant([[1,1,1,1],[2,2,2,2]],dtype=tf.float32))

input = tf.placeholder(dtype=tf.float32,shape = [2,4])


x_shape = input.get_shape()
params_shape = x_shape[-1:]

axis = list(range(len(x_shape) - 1))


beta = _get_variable('beta',params_shape,initializer=tf.zeros_initializer)
gamma = _get_variable('gamma',params_shape,initializer=tf.ones_initializer)

moving_mean = _get_variable('moving_mean',params_shape,initializer=tf.zeros_initializer,trainable=False)
moving_variance = _get_variable('moving_variance',params_shape,initializer=tf.ones_initializer,trainable=False)


mean_i, variance_i = tf.nn.moments(input, axis)


update_moving_mean = moving_averages.assign_moving_average(moving_mean,mean_i, bn_decay)
update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance_i, bn_decay)


tf.add_to_collection(update_ops_collection, update_moving_mean)
tf.add_to_collection(update_ops_collection, update_moving_variance)


mean, variance = control_flow_ops.cond(trainflag, lambda: (mean_i, variance_i),lambda: (moving_mean, moving_variance))

out = tf.nn.batch_normalization(input, mean, variance, beta, gamma, 0.0001)



bn_up = tf.get_collection(update_ops_collection)

# train_op = [out, update_moving_mean, update_moving_variance, moving_variance]

# with tf.control_dependencies([update_moving_mean, update_moving_variance]):
#     moving_mean     = moving_mean
#     moving_variance = moving_variance

train_op = [update_moving_mean, update_moving_variance, moving_mean, moving_variance]
# train_op = [update_moving_mean, update_moving_variance]

restoreflag = True



ref = tf.Variable([1,1],dtype=tf.int32)
value = tf.Variable([3,3], dtype=tf.int32)

# a = tf.subtract(ref,value)

def sub(ref1, value1):
    return state_ops.assign_sub(ref1, value1)

a = sub(ref, value)


saver = tf.train.Saver(max_to_keep=1)

init = tf.global_variables_initializer()

with tf.Session() as sess:



    # print(sess.run([a,ref]))
    # print(sess.run(ref))

    if  False:

        sess.run(init)

        for i in range(1):

            data = np.array([[i,i,i,i],[i+1,i+1,i+1,i+1]])

            result1, result2,result3, result4 = sess.run(train_op,feed_dict={input: data, trainflag:True})

            print(result1)
            print(result2)
            print(result3)
            print(result4)

            print("---------------------")

            # print(sess.run(moving_mean))

        save_path = saver.save(sess, "my_net/save_net.ckpt")

    else:

        saver.restore(sess, "my_net/save_net.ckpt")

        data = np.array([[1, 1, 1, 1], [4, 4, 4, 4]])

        print(sess.run([mean,variance,mean_i,variance_i], feed_dict={input: data, trainflag:False}))