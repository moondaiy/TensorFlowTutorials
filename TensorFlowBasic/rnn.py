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

weights = {"in":tf.Variable(tf.random_normal[n_inputs, n_hidden_units]),    #(28 * 128 相当于一个维度提升)
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

    cell = tf.contrib.rnn.BasicLSTMCell(new)  #new 才是真正的hiden Layer的个数

    init_state = cell.zero_state(batch_size, dtype=tf.float32)

