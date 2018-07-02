# -*- coding: utf-8 -*-
import tensorflow as tf

v = tf.Variable(1., name="v")
v2 = tf.Variable(1., name="v1")

# v3 = tf.Variable(1., name="v2")

# 滑动模型的参数的大小并不会影响v的值
ema = tf.train.ExponentialMovingAverage(0.99)

#variables_to_restore 需要给定需要被指定成滑动平均的变量,因为此时 并不知道那些变量是需要承担滑动平均的
#如不指定,则会将在这个函数以上的所有变量都指定成滑动平均的变量,在实际进行回复的时候会发生错误.
a = ema.variables_to_restore([v])

# {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}

sess = tf.Session()

#这个是使用显示的方式进行变量回复
saver = tf.train.Saver({"v/ExponentialMovingAverage":v, "v1":v2})

#这个是使用隐式方式进行 ,但是 variables_to_restore 函数要注意
saver = tf.train.Saver(a)

saver.restore(sess, "./model.ckpt")
print(sess.run(v))
# print(sess.run(v2))