import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


"""
完成一个简单的网络,但是利用函数进行添加Layer
"""

#定义数据

#构建训练用数据
xdata = np.linspace(-1,1,30)
noise = np.random.normal(0, 0.05, 30)
ydata = np.square(xdata) - 0.5 + noise


#画图
# plt.plot(xdata, ydata)
# plt.show()

#配合数据的使用[-1, 1] 矩阵行表示数据 一个batch有多少个数据 列表示数据维度
xdata = np.reshape(xdata,[-1, 1])
ydata = np.reshape(ydata,[-1, 1])

xs = tf.placeholder(dtype=tf.float32,shape=[None,1])
ys = tf.placeholder(dtype=tf.float32,shape=[None,1])

def addLayer(input,input_Size = 1, hidenSize=10,activation=None):

    #构造权重和偏执
    weight = tf.Variable(initial_value=tf.random_normal([input_Size, hidenSize]), name="weight")
    bias   = tf.Variable(initial_value=tf.zeros([hidenSize]), name="bias")

    out1 = tf.matmul(input, weight) + bias

    #如果有激活函数则使用激活函数进行操作,没有则使用线性函数输出
    if activation is None:
        outputs = out1
    else:
        outputs = activation(out1)

    return outputs


#构造网络形式
resultLayer1 = addLayer(xs, 1, 10, tf.nn.relu)
result  = addLayer(resultLayer1, 10, 1)

# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - result)))

#####################################这个地方有点问题,如果使用loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - result))) ,则会发散
#如果按照这样的方式,则不会loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - result),axis=1))
#不清楚tensorflow为什么会这样

#也是学习下 使用tf.Print调试的方法
se = tf.square(ys - result)
# se = tf.Print(se, [tf.shape(se)], summarize=10000,message="Se Info ")
se_sum = tf.reduce_sum(se,axis=1)
# se_sum = tf.Print(se_sum, [se_sum], summarize=10000,message="Se_Sum Info")

# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - result)))
loss = tf.reduce_mean(se_sum)


train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#tensorflow的版本信息,调用不同的init函数
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for i in range(500):

        sess.run(train_step,feed_dict={xs:xdata, ys:ydata})

        if i % 20 == 0:
            total_loss = sess.run(loss, feed_dict={xs:xdata, ys:ydata})
            print("After %d trainint step(s),loss on all data is %g" % (i, total_loss))








