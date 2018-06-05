import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math

print("Tensorflow version " + tf.__version__)

tf.set_random_seed(1)


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



#研究BN层对整体神经网络的影响

#我们要构建的神经网络层次如下, 这部分代码参考https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-mnist-tutorial/

# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid+BN)   W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid+BN)   W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (sigmoid+BN)   W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (sigmoid+BN)   W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax+BN)   W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]

#图的输入
# X = tf.placeholder(tf.float32, [None, 28, 28, 1])
X = tf.placeholder(tf.float32, [None, 28*28])
Y_ = tf.placeholder(tf.float32, [None, 10])

tst = tf.placeholder(tf.bool)

iter = tf.placeholder(tf.int32)

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 200
M = 100
N = 60
P = 30
Q = 10

W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
S1 = tf.Variable(tf.ones([L]))
O1 = tf.Variable(tf.zeros([L]))

W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
S2 = tf.Variable(tf.ones([M]))
O2 = tf.Variable(tf.zeros([M]))

W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
S3 = tf.Variable(tf.ones([N]))
O3 = tf.Variable(tf.zeros([N]))

W4 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1))
S4 = tf.Variable(tf.ones([P]))
O4 = tf.Variable(tf.zeros([P]))

W5 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1))
B5 = tf.Variable(tf.zeros([Q]))

def batchnorm(Ylogits, Offset, Scale, is_test, iteration, convolutional=False):

    exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, iteration) # adding the iteration prevents from averaging across non-existing iterations

    bnepsilon = 1e-5

    #因为是类似全连接层进行的.所以 moment [0] 就可以

    if convolutional: #针对卷积的BN 简单理解 有多少个卷积核 就有多少个mean 和 variance
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])

    # mean, variance = tf.nn.moments(Ylogits, [0])

    #apply 是实际进行计算的函数
    #更新的操作是需要额外进行的
    update_moving_averages = exp_moving_avg.apply([mean, variance])

    #测试阶段直接使用滑动平均来计算 ,在训练阶段则是直接使用 上面的 mean, variance = tf.nn.moments(Ylogits, [0])
    #为什么测试阶段需要滑动呢

    #exp_moving_avg.average(mean) 只是根据key mean 得到 数值
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)

    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)

    Ybn = tf.nn.batch_normalization(Ylogits, m, v, Offset, Scale, bnepsilon)

    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    return Ylogits, tf.no_op()

# The model
XX = tf.reshape(X, [-1, 784])

Y1l = tf.matmul(XX, W1)
Y1bn, update_ema1 = batchnorm(Y1l, O1, S1, tst, iter)
Y1 = tf.nn.sigmoid(Y1bn)

Y2l = tf.matmul(Y1, W2)
Y2bn, update_ema2 = batchnorm(Y2l, O2, S2, tst, iter)
Y2 = tf.nn.sigmoid(Y2bn)

Y3l = tf.matmul(Y2, W3)
Y3bn, update_ema3 = batchnorm(Y3l, O3, S3, tst, iter)
Y3 = tf.nn.sigmoid(Y3bn)

Y4l = tf.matmul(Y3, W4)
Y4bn, update_ema4 = batchnorm(Y4l, O4, S4, tst, iter)
Y4 = tf.nn.sigmoid(Y4bn)

Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)


#把多个需要操作的op 进行打包处理, 最终只要进行一次 sess.run 就可以全体进行更新啦
update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

# Y1l = tf.matmul(XX, W1)
# # Y1bn, update_ema1 = batchnorm(Y1l, O1, S1, tst, iter)
# Y1 = tf.nn.sigmoid(Y1l)
#
# Y2l = tf.matmul(Y1, W2)
# # Y2bn, update_ema2 = batchnorm(Y2l, O2, S2, tst, iter)
# Y2 = tf.nn.sigmoid(Y2l)
#
# Y3l = tf.matmul(Y2, W3)
# # Y3bn, update_ema3 = batchnorm(Y3l, O3, S3, tst, iter)
# Y3 = tf.nn.sigmoid(Y3l)
#
# Y4l = tf.matmul(Y3, W4)
# # Y4bn, update_ema4 = batchnorm(Y4l, O4, S4, tst, iter)
# Y4 = tf.nn.sigmoid(Y4l)
#
# Ylogits = tf.matmul(Y4, W5) + B5
# Y = tf.nn.softmax(Ylogits)



cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
# cross_entropy = tf.reduce_mean(cross_entropy)*100
cross_entropy = tf.reduce_mean(cross_entropy)*100 # loss 和100相乘表示 可以增加学习信号...


correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#这个学习率调整是很重要的
lr = 0.0001 +  tf.train.exponential_decay(0.03, iter, 1000, 1/math.e)
# lr = 0.001
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # compute training values for visualisation
    if update_train_data:
        a, c, l = sess.run([accuracy, cross_entropy, lr], feed_dict={X: batch_X, Y_: batch_Y, iter: i, tst: False})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(l) + ")")
        # datavis.append_training_curves_data(i, a, c)
        # datavis.update_image1(im)
        # datavis.append_data_histograms(i, al, ac)

    # compute test values for visualisation
    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, tst: True})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        # datavis.append_test_curves_data(i, a, c)
        # datavis.update_image2(im)

    # the backpropagation training step, also updates exponential moving averages for batch norm
    sess.run([train_step, update_ema], feed_dict={X: batch_X, Y_: batch_Y, iter: i, tst: False})
    # sess.run([train_step], feed_dict={X: batch_X, Y_: batch_Y, iter: i, tst: False})


with tf.Session() as sess:

    sess.run(init)

    for i in range(1000):
        training_step(i, True, False)
