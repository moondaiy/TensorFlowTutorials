
# -*- coding: utf-8 -*-
import numpy as np

import tensorflow as tf;

a = [1,5,3]

box1 = np.zeros((5,4),dtype=np.int64)
box2 = np.array([[1,2,3,4],[5,6,7,8]])

box3 = np.zeros((3,3,3))

ymin_1, xmin_1, ymax_1, xmax_1 = tf.split(box1, 4, axis=1)

#为了后续的broast cast 操作. 则需要进行tf.unstack操作(得到的是2个数字).而不能进行tf.split操作(得到的是矩阵)
#unstack会降低维度, split不会降低维度
ymin_2, xmin_2, ymax_2, xmax_2 = tf.unstack(box2, axis=1)

max_xmin = tf.maximum(ymin_1, ymin_2)

unstack_resulut = tf.unstack(box3, axis=0)

with tf.Session() as sess:

    # print("-----------------ymin2 \n")
    # print(sess.run(ymin_2))
    #
    # print("-----------------ymin1 \n")
    # print(sess.run(ymin_1))
    #
    # print("-----------------max_xmin \n")
    # print (sess.run(max_xmin))

    # print (sess.run(ymin_2))
    print(sess.run(unstack_resulut))
