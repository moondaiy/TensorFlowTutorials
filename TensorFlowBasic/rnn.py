import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)



flag = 1

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

lr = 0.001
training_iters = 1000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

new = 64


#对于rnn来说 tensorflow要求的输入格式为 batch step input
x = tf.placeholder(dtype=tf.float32, shape = [None, n_steps, n_inputs])
y = tf.placeholder(dtype=tf.float32,shape=[None, n_classes])

weights = {"in":tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),    #(28 * 128 相当于一个维度提升)
           'out': tf.Variable(tf.random_normal([new, n_classes]))} #(64, 10 ) 输出维度降维

biases = {'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])), # (128, )
          'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))   # (10, )
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################
    X = tf.reshape(X, [-1, n_inputs])  #这个是按照图像的每行作为一个 time steps 进行reshape的

    x_in = tf.matmul(X, weights["in"]) + biases["in"] #28 -> 128

    X_in = tf.reshape(x_in, [-1, n_steps, n_hidden_units])  # 128 * 28 *  128

    #
    cell = tf.contrib.rnn.BasicLSTMCell(new)  #new 才是真正的hiden Layer的个数

    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    #outputs 的输出 应该是 batch  step  out
    #dynamic_rnn 这个函数 和 static_rnn 一个不同是, dynamic_rnn 允许batch和batch之间的seqn lengh ,
    #但是对于同一个batch中的数据 必须先padding到同一个长度才行, 但是 tf.nn.dynamic_rnn 函数中一个参数sequence_length 表示这个batch中
    #各个数据的有效长度, 可以在计算中进行相关的优化
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    #需要transpose的原因是由于 接下来会进行tf.matmul操作
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

    #result中保存了每个batch 中最后一个step 中的结果 输出是10个
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)

    return results

pred = RNN(x, weights, biases)

#tf.reduce_mean要有.否则loss会发散
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#优化器
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

#这个做比较,计算准确率
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#计算准确率 tf.cast 是类型转换函数
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        #初始化参数
        init = tf.global_variables_initializer()

    sess.run(init)

    step = 0

    while step  < training_iters:

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])

        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys,})

        if step % 20 == 0:

            print(sess.run(accuracy, feed_dict={x: batch_xs,y: batch_ys,}))

        step += 1

