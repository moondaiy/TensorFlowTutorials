import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


tf.app.flags.DEFINE_integer("batch_size",32,"batch size")


FLAGS = tf.app.flags.FLAGS
#读取mnist数据集,但是该数据集是一个class,同时提供了相关方法
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#训练样本数据   [None,784]  拉平的数据
#训练样本label [None,10]    one hot编码

def addLayer(input, inputSize, outputSize, keep_prob, activationFun=None,nameScope="Layer"):

    with tf.variable_scope(nameScope):

        with tf.variable_scope("weight"):

            weight = tf.Variable(initial_value=tf.random_normal([inputSize, outputSize], dtype=tf.float32), name="weight")
            tf.summary.histogram('weight', weight)

        with tf.variable_scope("bias"):

            #关于bias的shape
            bias = tf.Variable(initial_value=tf.zeros([outputSize], dtype=tf.float32),name="bias")
            tf.summary.histogram('bias', bias)

        with tf.variable_scope("calcUnit"):

            Wx_plus_b = tf.add(tf.matmul(input, weight), bias)

        with tf.variable_scope("dropout"):

            Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

        with tf.variable_scope("activation_function"):

            if activationFun is not None:

                out = activationFun(Wx_plus_b)

            else:
                out = Wx_plus_b


    return out


def computeAccuracy(inputs, labels, prediction, sess):
# def computeAccuracy(inputs, labels, sess):

    # global prediction
    prediction = tf.nn.softmax(prediction)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = sess.run(accuracy,feed_dict={trainX: inputs, trainY: labels, keep_prob: 1.0})

    return result

def printFunction(loss):

    print(loss)

    return loss

def generateNet(inputs):

    input  = inputs[0]
    labels = inputs[1]
    keep_prob = inputs[2]

    # labels = tf.Print(labels, [labels], message="label info", summarize=1000)

    outLayer1 = addLayer(input, 28 * 28, 20, keep_prob, activationFun=tf.nn.relu, nameScope="LayerInput")

    # outLayer2 = addLayer(outLayer1, 20, 20, keep_prob, activationFun=tf.nn.relu, nameScope="Layer_Hidden")

    prediction = addLayer(outLayer1, 20, 10, keep_prob, activationFun=None, nameScope="LayerOutput")


    loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=labels)

    #要有tf.reduce_mean,否则会有问题
    loss = tf.reduce_mean(loss)

    tf.summary.scalar('loss', loss)

    return prediction, loss



def generateTrainOptimizer(loss):

    #加入DropOut后,学习率需要调整,否则会收敛很慢很慢
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    return train_step


with tf.name_scope("traindata"):

    #trainX 和 trainY
    trainX =tf.placeholder(dtype=tf.float32,shape=[None, 28 * 28], name="trainX")
    trainY = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="trainY")
    keep_prob = tf.placeholder(tf.float32)


#_参数 是当前py文件运行路径 ,这地方是由于tf.app.run()的原因,所以必须的需要一个_参数
def main(_):

    prediction, loss =  generateNet((trainX,trainY,keep_prob))

    train_step = generateTrainOptimizer(loss)

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()

    merged = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=4)

    with tf.Session() as sess:

        sess.run(init)

        if int((tf.__version__).split('.')[1]) < 12 and int(
                (tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
            writer = tf.train.SummaryWriter('logs/', sess.graph)
        else:  # tensorflow version >= 0.12
            writer = tf.summary.FileWriter("logs/", sess.graph)

        for i in range(5000):

            batch_xs, batch_ys = mnist.train.next_batch(100)  # 每次只取100张图片，免得数据太多训练太慢。#在每次循环中我们都随机抓取训练数据中 100 个数据点

            #虽然sess.run([fetch1,fetch2]....)里面有2个,只不过应该是计算了一次 取出了2个值.
            _ , currentLoss,summary = sess.run([train_step, loss,merged], feed_dict={trainX: batch_xs, trainY: batch_ys, keep_prob: 0.5})

            writer.add_summary(summary, i)

            # print(float(currentLoss))
            #
            if i % 50 == 0:
                # 注意，这里改成了测试集
                print(computeAccuracy(mnist.test.images, mnist.test.labels , prediction , sess))

        save_path = saver.save(sess, "my_net/save_net.ckpt")

        writer.close()


    return

if __name__ == '__main__':

    tf.app.run()





